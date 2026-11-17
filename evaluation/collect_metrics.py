import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import time


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "results" / "raw"
OUTPUT_DIR = BASE_DIR / "results" / "processed"
INTERVAL_SECONDS = 10
CONTRACT_COLUMNS = (
    "timestamp",
    "request_count",
    "request_rate_rps",
    "cpu_utilization_pct",
    "memory_usage_bytes",
    "replica_count",
    "average_latency_ms",
    "p95_latency_ms",
    "error_count",
)
REQUEST_COLUMNS = (
    "timestamp_utc",
    "response_time_ms",
    "status_code",
    "exception",
)
RUN_ID_PATTERN = re.compile(r"locust_[A-Za-z0-9][A-Za-z0-9_-]*")


class MetricsCollectionError(ValueError):
    pass


def _parse_timestamp(value, source, line):
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise MetricsCollectionError(
            f"Invalid timestamp in {source} line {line}: {value!r}"
        ) from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise MetricsCollectionError(
            f"Timestamp must include a timezone in {source} line {line}."
        )
    return timestamp.astimezone(timezone.utc)


def _percentile(values, percentile):
    if not values:
        return ""
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _run_bins(run_id, raw_dir):
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise MetricsCollectionError(f"Invalid run ID: {run_id!r}")
    request_path = Path(raw_dir) / f"{run_id}_requests.csv"
    if not request_path.is_file():
        raise FileNotFoundError(f"Request CSV not found: {request_path}")

    with request_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise MetricsCollectionError(f"Request CSV is empty: {request_path}")
        missing = [field for field in REQUEST_COLUMNS if field not in reader.fieldnames]
        if missing:
            raise MetricsCollectionError(
                f"Request CSV is missing required columns: {', '.join(missing)}"
            )
        bins = {}
        for line, record in enumerate(reader, start=2):
            timestamp = _parse_timestamp(
                record["timestamp_utc"], request_path, line
            )
            bucket = math.floor(timestamp.timestamp() / INTERVAL_SECONDS) * INTERVAL_SECONDS
            stats = bins.setdefault(
                bucket,
                {"request_count": 0, "latencies": [], "errors": 0, "known_errors": True},
            )
            stats["request_count"] += 1
            latency_value = (record["response_time_ms"] or "").strip()
            if latency_value:
                try:
                    latency = float(latency_value)
                except ValueError as error:
                    raise MetricsCollectionError(
                        f"Invalid response_time_ms in {request_path} line {line}: "
                        f"{latency_value!r}"
                    ) from error
                if not math.isfinite(latency) or latency < 0:
                    raise MetricsCollectionError(
                        f"Invalid response_time_ms in {request_path} line {line}: "
                        f"{latency_value!r}"
                    )
                stats["latencies"].append(latency)
            status_value = (record["status_code"] or "").strip()
            exception = (record["exception"] or "").strip()
            if exception:
                stats["errors"] += 1
            elif status_value:
                try:
                    if int(status_value) >= 400:
                        stats["errors"] += 1
                except ValueError as error:
                    raise MetricsCollectionError(
                        f"Invalid status_code in {request_path} line {line}: "
                        f"{status_value!r}"
                    ) from error
            else:
                stats["known_errors"] = False
    if not bins:
        raise MetricsCollectionError(f"Request CSV has no request records: {request_path}")
    return bins


def _csv_row(timestamp, stats):
    count = stats["request_count"]
    latencies = stats["latencies"]
    return {
        "timestamp": timestamp.isoformat(),
        "request_count": count,
        "request_rate_rps": count / INTERVAL_SECONDS,
        "cpu_utilization_pct": "",
        "memory_usage_bytes": "",
        "replica_count": "",
        "average_latency_ms": (
            sum(latencies) / len(latencies) if latencies else ""
        ),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "error_count": stats["errors"] if stats["known_errors"] else "",
    }


def _write_contract(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CONTRACT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def collect_run(run_id, raw_dir=RAW_DIR, output_dir=OUTPUT_DIR):
    bins = _run_bins(run_id, raw_dir)
    first, last = min(bins), max(bins)
    rows = []
    for bucket in range(first, last + INTERVAL_SECONDS, INTERVAL_SECONDS):
        stats = bins.get(
            bucket,
            {
                "request_count": 0,
                "latencies": [],
                "errors": 0,
                "known_errors": True,
            },
        )
        rows.append(_csv_row(datetime.fromtimestamp(bucket, timezone.utc), stats))
    return _write_contract(
        Path(output_dir) / f"{run_id}_10s_contract.csv", rows
    )


def _parse_cpu(value):
    if value.endswith("n"):
        return float(value[:-1]) / 1_000_000_000
    if value.endswith("u"):
        return float(value[:-1]) / 1_000_000
    if value.endswith("m"):
        return float(value[:-1]) / 1000
    return float(value)


def _parse_bytes(value):
    units = {
        "Ki": 1024,
        "Mi": 1024**2,
        "Gi": 1024**3,
        "Ti": 1024**4,
        "K": 1000,
        "M": 1000**2,
        "G": 1000**3,
        "T": 1000**4,
    }
    for suffix in sorted(units, key=len, reverse=True):
        if value.endswith(suffix):
            return int(float(value[: -len(suffix)]) * units[suffix])
    return int(value)


def sample_kubernetes_metrics():
    deployment = subprocess.run(
        [
            "kubectl",
            "get",
            "deployment",
            "cloudpulse-service",
            "-n",
            "cloudpulse",
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    deployment_data = json.loads(deployment.stdout)
    replica_count = deployment_data.get("status", {}).get("availableReplicas")
    cpu_limits = []
    for container in (
        deployment_data.get("spec", {}).get("template", {})
        .get("spec", {})
        .get("containers", [])
    ):
        cpu_limit = (
            container.get("resources", {})
            .get("limits", {})
            .get("cpu")
        )
        if cpu_limit:
            cpu_limits.append(_parse_cpu(str(cpu_limit)))

    top = subprocess.run(
        [
            "kubectl",
            "top",
            "pods",
            "-n",
            "cloudpulse",
            "-l",
            "app=cloudpulse-service",
            "--no-headers",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    cpu_used = 0.0
    memory_used = 0
    pod_count = 0
    for line in top.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        cpu_used += _parse_cpu(parts[1])
        memory_used += _parse_bytes(parts[2])
        pod_count += 1
    cpu_limit_total = sum(cpu_limits) * pod_count
    cpu_pct = (
        cpu_used / cpu_limit_total * 100
        if cpu_limit_total > 0 and pod_count
        else ""
    )
    return {
        "cpu_utilization_pct": cpu_pct,
        "memory_usage_bytes": memory_used if pod_count else "",
        "replica_count": replica_count if replica_count is not None else "",
    }


def collect_live_k8s(output_path, samples=None):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CONTRACT_COLUMNS)
        writer.writeheader()
        output.flush()
        count = 0
        next_sample = time.monotonic()
        while samples is None or count < samples:
            metrics = sample_kubernetes_metrics()
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_count": "",
                "request_rate_rps": "",
                **metrics,
                "average_latency_ms": "",
                "p95_latency_ms": "",
                "error_count": "",
            }
            writer.writerow(row)
            output.flush()
            count += 1
            next_sample += INTERVAL_SECONDS
            if samples is not None and count >= samples:
                break
            time.sleep(max(0, next_sample - time.monotonic()))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export honest 10-second CloudPulse workload or Kubernetes metrics."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-id", help="Run ID whose per-request CSV is aggregated")
    mode.add_argument(
        "--live-k8s",
        action="store_true",
        help="Continuously sample CloudPulse Kubernetes deployment/pod metrics",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Optional number of 10-second live Kubernetes samples, useful for bounded checks",
    )
    args = parser.parse_args(argv)
    if args.samples is not None and (not args.live_k8s or args.samples < 1):
        parser.error("--samples requires --live-k8s and must be positive")
    try:
        if args.run_id:
            output = collect_run(args.run_id)
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = collect_live_k8s(
                OUTPUT_DIR / f"live_k8s_{stamp}_10s_contract.csv",
                samples=args.samples,
            )
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, MetricsCollectionError) as error:
        parser.exit(1, f"Metrics collection failed: {error}\n")
    print(f"10-second metrics saved to {output}")


if __name__ == "__main__":
    main()
