
import csv
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from threading import RLock
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="CloudPulse",
    description="Intelligent Predictive Cloud Scaling and Dynamic Load Distribution",
    version="0.3.0",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "results" / "raw"
CSV_FILE = DATA_DIR / "request_logs.csv"
RUNNER_FILE = BASE_DIR / "testing" / "run_load_test.py"
DASHBOARD_FILE = Path(__file__).resolve().parent / "dashboard.html"

lock = RLock()
total_requests = 0
endpoint_counts = Counter()
request_history = []
experiment_process = None
experiment_log_file = None
experiment_state = {
    "status": "idle",
    "profile": None,
    "phase": None,
    "run_id": None,
    "started_at": None,
    "duration_seconds": None,
    "users": None,
    "message": "No experiment is running.",
    "return_code": None,
}

CSV_HEADERS = [
    "timestamp",
    "method",
    "path",
    "status_code",
    "duration_ms",
]
REQUEST_CSV_HEADERS = {
    "timestamp_utc",
    "response_time_ms",
    "status_code",
    "exception",
}
RUN_ID_PATTERN = re.compile(
    r"locust_(steady|spike|ramp|periodic)_\d{8}T\d{6}_\d{6}Z(?:_[A-Za-z0-9-]+)?"
)


def initialize_csv():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0:
        with CSV_FILE.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
            writer.writeheader()


initialize_csv()


class ExperimentRequest(BaseModel):
    profile: Literal["steady", "spike"]
    users: int = Field(default=10, ge=1, le=1000)
    duration_seconds: int = Field(default=60, ge=1, le=3600)
    spawn_rate: int = Field(default=5, ge=1, le=1000)


@app.middleware("http")
async def measure_requests(request: Request, call_next):
    global total_requests

    start = datetime.now(timezone.utc)
    response = None

    try:
        response = await call_next(request)
        return response
    finally:
        end = datetime.now(timezone.utc)

        record = {
            "timestamp": end.isoformat(),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code if response else 500,
            "duration_ms": round(
                (end - start).total_seconds() * 1000, 3
            ),
        }

        with lock:
            total_requests += 1
            endpoint_counts[request.url.path] += 1
            request_history.append(record)

            with CSV_FILE.open(
                "a", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
                writer.writerow(record)


@app.get("/")
def home():
    return {
        "project": "CloudPulse",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/dashboard")
def dashboard():
    return FileResponse(DASHBOARD_FILE, media_type="text/html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/workload")
def workload():
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": "CloudPulse workload endpoint is ready",
    }


@app.get("/metrics")
def metrics():
    with lock:
        return {
            "total_requests": total_requests,
            "requests_by_endpoint": dict(endpoint_counts),
            "recent_requests": request_history[-10:],
            "csv_file": str(CSV_FILE),
        }


def _run_phase(profile, elapsed, duration):
    if profile == "steady":
        return "Steady load"
    warmup = max(1, duration // 4)
    peak = max(1, duration // 2)
    if elapsed < warmup:
        return "Warm-up"
    if elapsed < warmup + peak:
        return "Peak"
    return "Recovery"


def _refresh_process_state():
    global experiment_log_file

    if experiment_process is None or experiment_process.poll() is None:
        return
    if experiment_state["status"] in {"stopping", "stopped"}:
        experiment_state["status"] = "stopped"
        experiment_state["message"] = "Experiment stopped."
    elif experiment_process.returncode == 0:
        experiment_state["status"] = "completed"
        experiment_state["message"] = "Experiment completed."
    else:
        experiment_state["status"] = "failed"
        experiment_state["message"] = (
            f"Experiment runner exited with code {experiment_process.returncode}."
        )
    experiment_state["phase"] = None
    experiment_state["return_code"] = experiment_process.returncode
    if experiment_log_file is not None:
        experiment_log_file.close()
        experiment_log_file = None


def _read_experiment_requests(path):
    records = []
    try:
        with path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                try:
                    timestamp = datetime.fromisoformat(row["timestamp_utc"])
                    latency = float(row["response_time_ms"])
                    status_code = int(row["status_code"]) if row["status_code"] else 0
                except (KeyError, TypeError, ValueError):
                    continue
                if timestamp.tzinfo is None or not math.isfinite(latency):
                    continue
                records.append(
                    {
                        "timestamp": timestamp,
                        "latency_ms": latency,
                        "failed": bool(row.get("exception", "").strip())
                        or status_code >= 400,
                    }
                )
    except (OSError, UnicodeDecodeError, csv.Error):
        return []
    return records


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _load_saved_experiment(run_id):
    match = RUN_ID_PATTERN.fullmatch(run_id)
    if match is None:
        return None

    metadata_path = DATA_DIR / f"{run_id}_metadata.json"
    request_path = DATA_DIR / f"{run_id}_requests.csv"
    if not metadata_path.is_file() or not request_path.is_file():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        configuration = metadata["configuration"]
        if not isinstance(configuration, dict):
            return None
        scenario = configuration["scenario"]
        started_at = datetime.fromisoformat(metadata["started_at_utc"])
        duration_seconds = int(configuration["duration_seconds"])
        status = metadata["status"]
        if (
            scenario != match.group(1)
            or started_at.tzinfo is None
            or duration_seconds < 1
            or status not in {"running", "completed", "interrupted", "failed"}
            or not isinstance(metadata.get("exit_code"), (int, type(None)))
        ):
            return None
        with request_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if not REQUEST_CSV_HEADERS.issubset(reader.fieldnames or set()):
                return None
            records = _read_experiment_requests(request_path)
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        csv.Error,
    ):
        return None

    return {
        "run_id": run_id,
        "scenario": scenario,
        "status": status,
        "started_at": started_at,
        "duration_seconds": duration_seconds,
        "users": configuration.get("peak_users"),
        "metadata": metadata,
        "records": records,
    }


def _saved_experiment_summary(saved):
    records = saved["records"]
    latencies = [record["latency_ms"] for record in records]
    return {
        "run_id": saved["run_id"],
        "scenario": saved["scenario"],
        "status": saved["status"],
        "started_at": saved["started_at"].isoformat(),
        "duration_seconds": saved["duration_seconds"],
        "users": saved["users"],
        "request_count": len(records),
        "failed_requests": sum(record["failed"] for record in records),
        "average_latency_ms": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "median_latency_ms": _percentile(latencies, 0.5),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "average_throughput_rps": (
            len(records) / saved["duration_seconds"]
        ),
    }


def _saved_experiment_time_series(saved):
    started_at = saved["started_at"]
    request_bins = {}
    latency_bins = {}
    for record in saved["records"]:
        elapsed_seconds = (record["timestamp"] - started_at).total_seconds()
        if elapsed_seconds < 0:
            continue
        second = math.floor(elapsed_seconds)
        request_bins[second] = request_bins.get(second, 0) + 1
        latency_bins.setdefault(second, []).append(record["latency_ms"])

    chart_end = max(request_bins, default=-1) + 1
    return {
        "request_rate": [
            {"time_seconds": second, "value": request_bins.get(second, 0)}
            for second in range(chart_end)
        ],
        "latency": [
            {
                "time_seconds": second,
                "value": (
                    sum(latency_bins[second]) / len(latency_bins[second])
                    if latency_bins.get(second)
                    else None
                ),
            }
            for second in range(chart_end)
        ],
    }


@app.get("/api/experiments/history")
def experiment_history():
    experiments = []
    for metadata_path in DATA_DIR.glob("locust_*_metadata.json"):
        run_id = metadata_path.name[: -len("_metadata.json")]
        saved = _load_saved_experiment(run_id)
        if saved is not None:
            experiments.append(_saved_experiment_summary(saved))
    experiments.sort(key=lambda run: run["started_at"], reverse=True)
    return {"experiments": experiments}


@app.get("/api/experiments/history/{run_id}")
def saved_experiment_data(run_id: str):
    saved = _load_saved_experiment(run_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="Saved experiment not found.")

    summary = _saved_experiment_summary(saved)
    records = saved["records"]
    started_at = saved["started_at"]
    series = _saved_experiment_time_series(saved)
    if series["request_rate"]:
        chart_end = len(series["request_rate"])
        chart_start = max(0, chart_end - 60)
    else:
        chart_start = chart_end = 0
    request_rate = series["request_rate"][chart_start:chart_end]
    latency = series["latency"][chart_start:chart_end]

    latest_timestamp = max(
        (record["timestamp"] for record in records), default=None
    )
    if latest_timestamp is None:
        live_window_seconds = 1.0
        live_requests = 0
    else:
        elapsed = max(0.0, (latest_timestamp - started_at).total_seconds())
        live_window_seconds = min(10.0, max(1.0, elapsed))
        window_start = latest_timestamp.timestamp() - 10
        live_requests = sum(
            record["timestamp"].timestamp() >= window_start for record in records
        )

    metadata = saved["metadata"]
    with lock:
        _refresh_process_state()
        current_state = dict(experiment_state)
        current_process = experiment_process
        current_experiment_active = (
            current_process is not None and current_process.poll() is None
        )
    return {
        "historical": True,
        "current_experiment": {
            "status": current_state["status"],
            "active": current_experiment_active,
        },
        "experiment": {
            "status": saved["status"],
            "profile": saved["scenario"],
            "phase": None,
            "run_id": saved["run_id"],
            "started_at": started_at.isoformat(),
            "duration_seconds": saved["duration_seconds"],
            "users": saved["users"],
            "message": "Saved historical experiment; read-only.",
            "return_code": metadata.get("exit_code"),
        },
        "metrics": {
            "live_window_seconds": round(live_window_seconds, 1),
            "live_requests_per_second": round(
                live_requests / live_window_seconds, 2
            ),
            "total_requests": summary["request_count"],
            "failed_requests": summary["failed_requests"],
            "average_latency_ms": summary["average_latency_ms"],
            "median_latency_ms": summary["median_latency_ms"],
            "p95_latency_ms": summary["p95_latency_ms"],
        },
        "request_rate": request_rate,
        "latency": latency,
        "kubernetes": {
            "replicas": None,
            "cpu_percent": None,
            "memory_percent": None,
            "available": False,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/experiments/compare/{run_id_a}/{run_id_b}")
def compare_saved_experiments(run_id_a: str, run_id_b: str):
    saved_runs = []
    for run_id in (run_id_a, run_id_b):
        saved = _load_saved_experiment(run_id)
        if saved is None:
            raise HTTPException(status_code=404, detail="Saved experiment not found.")
        saved_runs.append(
            {
                "summary": _saved_experiment_summary(saved),
                **_saved_experiment_time_series(saved),
            }
        )
    return {"runs": saved_runs}


@app.get("/api/dashboard")
def dashboard_data():
    now = datetime.now(timezone.utc)
    with lock:
        _refresh_process_state()
        state = dict(experiment_state)
        process = experiment_process

    if not state["run_id"]:
        return {
            "experiment": state,
            "metrics": {
                "live_window_seconds": 10,
                "live_requests_per_second": 0,
                "total_requests": 0,
                "failed_requests": 0,
                "average_latency_ms": None,
                "median_latency_ms": None,
                "p95_latency_ms": None,
            },
            "request_rate": [],
            "latency": [],
            "kubernetes": {
                "replicas": None,
                "cpu_percent": None,
                "memory_percent": None,
                "available": False,
            },
            "updated_at": now.isoformat(),
        }

    request_path = DATA_DIR / f"{state['run_id']}_requests.csv"
    records = _read_experiment_requests(request_path)
    latencies = [record["latency_ms"] for record in records]
    failures = sum(record["failed"] for record in records)
    started_at = datetime.fromisoformat(state["started_at"])
    elapsed = max(0.0, (now - started_at).total_seconds())
    window_start = now.timestamp() - 10
    live_requests = sum(
        record["timestamp"].timestamp() >= window_start for record in records
    )
    active_window_seconds = min(10.0, max(1.0, elapsed))
    request_bins = {}
    latency_bins = {}
    for record in records:
        second = int((record["timestamp"] - started_at).total_seconds())
        if second < 0:
            continue
        request_bins[second] = request_bins.get(second, 0) + 1
        latency_bins.setdefault(second, []).append(record["latency_ms"])
    if state["status"] in {"completed", "stopped"}:
        if request_bins:
            latest_request_second = max(request_bins)
            chart_end = latest_request_second + 1
            chart_start = max(0, chart_end - 60)
        else:
            chart_start = chart_end = 0
    else:
        chart_start = max(0, int(elapsed) - 59)
        chart_end = max(chart_start, int(elapsed)) + 1
    request_rate = [
        {"time_seconds": second, "value": request_bins.get(second, 0)}
        for second in range(chart_start, chart_end)
    ]
    latency = [
        {
            "time_seconds": second,
            "value": (
                sum(latency_bins[second]) / len(latency_bins[second])
                if latency_bins.get(second)
                else None
            ),
        }
        for second in range(chart_start, chart_end)
    ]

    if process is not None and process.poll() is None:
        with lock:
            experiment_state["phase"] = _run_phase(
                state["profile"], elapsed, state["duration_seconds"]
            )
            if state["status"] == "starting" and (records or elapsed >= 3):
                experiment_state["status"] = "running"
                experiment_state["message"] = "Experiment is running."
            state = dict(experiment_state)

    return {
        "experiment": state,
        "metrics": {
            "live_window_seconds": round(active_window_seconds, 1),
            "live_requests_per_second": round(live_requests / active_window_seconds, 2),
            "total_requests": len(records),
            "failed_requests": failures,
            "average_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "median_latency_ms": _percentile(latencies, 0.5),
            "p95_latency_ms": _percentile(latencies, 0.95),
        },
        "request_rate": request_rate,
        "latency": latency,
        "kubernetes": {
            "replicas": None,
            "cpu_percent": None,
            "memory_percent": None,
            "available": False,
        },
        "updated_at": now.isoformat(),
    }


@app.post("/api/experiments")
def start_experiment(request: ExperimentRequest):
    global experiment_process, experiment_log_file

    if request.profile == "spike" and request.duration_seconds < 3:
        raise HTTPException(
            status_code=422,
            detail="Spike experiments require a duration of at least 3 seconds.",
        )
    if request.profile == "spike" and request.users < 2:
        raise HTTPException(
            status_code=422,
            detail="Spike experiments require at least 2 peak users.",
        )

    with lock:
        _refresh_process_state()
        if experiment_process is not None and experiment_process.poll() is None:
            raise HTTPException(
                status_code=409,
                detail="An experiment is already running; stop it before starting another.",
            )

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        run_id = f"locust_{request.profile}_{timestamp}_{uuid4().hex[:6]}"
        while any(DATA_DIR.glob(f"{run_id}_*")):
            run_id = f"locust_{request.profile}_{timestamp}_{uuid4().hex[:6]}"

        command = [
            sys.executable,
            str(RUNNER_FILE),
            "--scenario",
            request.profile,
            "--users",
            str(request.users),
            "--duration",
            str(request.duration_seconds),
            "--spawn-rate",
            str(request.spawn_rate),
            "--host",
            "http://127.0.0.1:8000",
            "--output-prefix",
            run_id,
        ]
        log_path = DATA_DIR / f"dashboard_{run_id}.log"
        experiment_log_file = log_path.open("x", encoding="utf-8")
        try:
            creation_flags = (
                subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )
            experiment_process = subprocess.Popen(
                command,
                cwd=BASE_DIR,
                stdout=experiment_log_file,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
        except OSError as error:
            experiment_log_file.close()
            experiment_log_file = None
            raise HTTPException(
                status_code=500, detail=f"Could not start Locust runner: {error}"
            ) from error

        experiment_state.update(
            {
                "status": "starting",
                "profile": request.profile,
                "phase": "Starting",
                "run_id": run_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": request.duration_seconds,
                "users": request.users,
                "message": "Locust runner is starting.",
                "return_code": None,
            }
        )
        return {"experiment": dict(experiment_state)}


@app.post("/api/experiments/stop")
def stop_experiment():
    with lock:
        _refresh_process_state()
        process = experiment_process
        if process is None or process.poll() is not None:
            raise HTTPException(status_code=409, detail="No experiment is running.")
        if experiment_state["status"] == "stopping":
            return {"experiment": dict(experiment_state)}
        experiment_state["status"] = "stopping"
        experiment_state["phase"] = None
        experiment_state["message"] = "Stopping experiment gracefully."
        stop_path = DATA_DIR / f"{experiment_state['run_id']}_stop.request"

    try:
        with stop_path.open("x", encoding="utf-8"):
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            else:
                process.terminate()
            process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        with lock:
            experiment_state["status"] = "running"
            experiment_state["message"] = f"Could not stop experiment safely: {error}"
        raise HTTPException(
            status_code=500, detail=f"Could not stop experiment safely: {error}"
        ) from error

    with lock:
        _refresh_process_state()
        experiment_state["status"] = "stopped"
        experiment_state["phase"] = None
        experiment_state["message"] = "Experiment stopped."
        return {"experiment": dict(experiment_state)}
