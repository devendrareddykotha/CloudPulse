import argparse
import csv
from datetime import datetime, timezone
import ipaddress
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent.parent
LOCUST_FILE = BASE_DIR / "testing" / "locustfile.py"
RAW_RESULTS_DIR = BASE_DIR / "results" / "raw"
SCENARIOS = ("steady", "spike", "ramp", "periodic")
REQUEST_FIELDS = (
    "timestamp_utc",
    "method",
    "name",
    "url",
    "status_code",
    "response_time_ms",
    "response_length",
    "exception",
)


def positive_int(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a whole number") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def unique_csv_prefix(scenario):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    prefix = RAW_RESULTS_DIR / f"locust_{scenario}_{timestamp}"
    suffix = 1
    while any(RAW_RESULTS_DIR.glob(f"{prefix.name}_*")):
        prefix = RAW_RESULTS_DIR / f"locust_{scenario}_{timestamp}_{suffix}"
        suffix += 1
    return prefix


def write_metadata(path, metadata):
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")


def is_loopback_host(host):
    try:
        parsed = urlsplit(host)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        if parsed.username is not None or parsed.password is not None:
            return False
        if parsed.hostname.lower() == "localhost":
            return True
        return ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False


def target_user_count(args, elapsed, duration):
    if args.scenario == "steady":
        return args.users
    if args.scenario == "spike":
        if elapsed < args.warmup_duration:
            return args.low_users
        if elapsed < args.warmup_duration + args.peak_duration:
            return args.users
        return args.low_users
    if args.scenario == "ramp":
        return min(args.users, math.ceil(args.users * elapsed / duration))

    cycle_duration = duration / 4
    high_phase = elapsed % cycle_duration >= cycle_duration / 2
    return args.users if high_phase else args.users // 2


def _record_local_request(
    base_url, request_writer, request_file, request_lock, worker_state
):
    worker_state["iteration"] += 1
    endpoint = (
        "/health" if worker_state["iteration"] % 6 == 0 else "/workload"
    )
    url = f"{base_url}{endpoint}"
    started_at = datetime.now(timezone.utc)
    start_clock = time.perf_counter()
    status_code = ""
    response_length = ""
    exception = ""
    try:
        request = Request(url, method="GET")
        try:
            with urlopen(request, timeout=5) as response:
                status_code = response.status
                response_length = len(response.read())
        except HTTPError as error:
            status_code = error.code
            response_length = len(error.read())
    except (OSError, URLError, TimeoutError) as error:
        exception = str(error)
    response_time_ms = (time.perf_counter() - start_clock) * 1000
    row = {
        "timestamp_utc": started_at.isoformat(),
        "method": "GET",
        "name": endpoint,
        "url": url,
        "status_code": status_code,
        "response_time_ms": f"{response_time_ms:.3f}",
        "response_length": response_length,
        "exception": exception,
    }
    with request_lock:
        request_writer.writerow(row)
        request_file.flush()


def run_standard_library_load(args, duration, request_csv_path, stop_file):
    """Run a bounded, loopback-only experiment without importing native Locust dependencies."""
    parsed_host = urlsplit(args.host)
    base_url = args.host.rstrip("/")
    if parsed_host.path not in ("", "/") or parsed_host.query or parsed_host.fragment:
        raise ValueError("The standard-library runner requires a loopback base URL without a path.")
    if not is_loopback_host(args.host):
        raise ValueError("The standard-library runner is restricted to loopback hosts.")

    stop_event = threading.Event()
    condition = threading.Condition()
    request_lock = threading.Lock()
    worker_errors = []
    active_users = 0
    started_clock = time.monotonic()
    next_tick = started_clock
    previous_tick = started_clock
    spawn_credit = 0.0

    def worker(worker_id):
        state = {"iteration": 0}
        try:
            while not stop_event.is_set():
                with condition:
                    while worker_id >= active_users and not stop_event.is_set():
                        condition.wait(timeout=0.1)
                if stop_event.is_set():
                    return

                request_started = time.monotonic()
                _record_local_request(
                    base_url, writer, request_file, request_lock, state
                )
                stop_event.wait(max(0.0, 1.0 - (time.monotonic() - request_started)))
        except Exception as error:
            with condition:
                worker_errors.append(error)
                stop_event.set()
                condition.notify_all()

    with request_csv_path.open("x", newline="", encoding="utf-8") as request_file:
        writer = csv.DictWriter(request_file, fieldnames=REQUEST_FIELDS)
        writer.writeheader()
        request_file.flush()
        workers = [
            threading.Thread(target=worker, args=(worker_id,), daemon=True)
            for worker_id in range(args.users)
        ]
        for thread in workers:
            thread.start()

        try:
            while time.monotonic() - started_clock < duration:
                if stop_file.exists():
                    break
                now = time.monotonic()
                elapsed = now - started_clock
                desired_users = target_user_count(args, elapsed, duration)
                spawn_credit += (now - previous_tick) * args.spawn_rate
                previous_tick = now
                user_steps = int(spawn_credit)
                if user_steps:
                    spawn_credit -= user_steps
                with condition:
                    if active_users < desired_users:
                        active_users = min(
                            desired_users, active_users + user_steps
                        )
                    elif active_users > desired_users:
                        active_users = max(
                            desired_users, active_users - user_steps
                        )
                    condition.notify_all()
                if worker_errors:
                    break
                next_tick = min(started_clock + duration, now + 0.05)
                stop_event.wait(max(0, next_tick - time.monotonic()))
        except KeyboardInterrupt:
            stop_file.touch(exist_ok=True)
        finally:
            stop_event.set()
            with condition:
                condition.notify_all()
            for thread in workers:
                thread.join(timeout=6)

    if worker_errors:
        raise RuntimeError(f"Local load worker failed: {worker_errors[0]}") from worker_errors[0]
    return stop_file.exists()


def run_locust_subprocess(command, environment, stop_file):
    exit_code = 1
    stop_requested = False
    stop_requested_at = None
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with subprocess.Popen(
        command,
        cwd=BASE_DIR,
        env=environment,
        creationflags=creation_flags,
    ) as locust_process:
        while True:
            try:
                exit_code = locust_process.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if stop_file.exists() and not stop_requested:
                    stop_requested = True
                    stop_requested_at = time.monotonic()
                    print(
                        "Stop requested; Locust workload will stop at its next phase tick.",
                        flush=True,
                    )
                if (
                    stop_requested_at is not None
                    and time.monotonic() - stop_requested_at > 15
                ):
                    print(
                        "Locust did not stop after the request; terminating its process tree.",
                        flush=True,
                    )
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(locust_process.pid), "/T", "/F"],
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                    else:
                        locust_process.terminate()
                    exit_code = locust_process.wait(timeout=5)
                    break
    return exit_code, stop_requested


def main():
    parser = argparse.ArgumentParser(description="Run a CloudPulse workload scenario")
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--users", type=positive_int, default=10, help="Peak concurrent users (maximum 100)")
    parser.add_argument("--duration", type=positive_int, help="Total duration in seconds")
    parser.add_argument("--spawn-rate", type=positive_int, default=5, help="Users started or stopped per second")
    parser.add_argument("--host", default="http://127.0.0.1:8000", help="FastAPI base URL")
    parser.add_argument(
        "--output-prefix",
        help="Optional unique filename prefix under results/raw (used by the dashboard)",
    )
    parser.add_argument("--low-users", type=positive_int, help="Spike warm-up and recovery user count")
    parser.add_argument("--warmup-duration", type=positive_int)
    parser.add_argument("--peak-duration", type=positive_int)
    parser.add_argument("--recovery-duration", type=positive_int)
    parser.add_argument("--warmup-spawn-rate", type=positive_int)
    parser.add_argument("--spike-spawn-rate", type=positive_int)
    parser.add_argument("--recovery-spawn-rate", type=positive_int)
    args = parser.parse_args()

    if args.users > 100:
        parser.error("--users cannot exceed 100")

    phase_duration_args = (
        args.warmup_duration,
        args.peak_duration,
        args.recovery_duration,
    )
    phase_rate_args = (
        args.warmup_spawn_rate,
        args.spike_spawn_rate,
        args.recovery_spawn_rate,
    )
    has_spike_options = args.low_users is not None or any(
        value is not None for value in (*phase_duration_args, *phase_rate_args)
    )
    if args.scenario != "spike" and has_spike_options:
        parser.error("Spike phase options can only be used with --scenario spike")

    if args.scenario == "spike":
        if any(value is not None for value in phase_duration_args) and not all(
            value is not None for value in phase_duration_args
        ):
            parser.error("Set all three spike phase durations, or none of them")
        if all(value is not None for value in phase_duration_args):
            warmup_duration, peak_duration, recovery_duration = phase_duration_args
            duration = sum(phase_duration_args)
            if args.duration is not None and args.duration != duration:
                parser.error("--duration must equal the sum of the three spike phase durations")
        else:
            duration = args.duration or 120
            if duration < 3:
                parser.error("Spike --duration must be at least 3 seconds")
            warmup_duration = max(1, duration // 4)
            peak_duration = max(1, duration // 2)
            recovery_duration = duration - warmup_duration - peak_duration

        low_users = args.low_users or max(1, args.users // 2)
        if low_users >= args.users:
            parser.error("--low-users must be less than peak --users")
        warmup_spawn_rate = args.warmup_spawn_rate or args.spawn_rate
        spike_spawn_rate = args.spike_spawn_rate or args.spawn_rate
        recovery_spawn_rate = args.recovery_spawn_rate or args.spawn_rate
    else:
        duration = args.duration or 120

    if args.scenario == "periodic" and duration < 8:
        parser.error("--duration must be at least 8 seconds for the periodic scenario")

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output_prefix:
        if (
            Path(args.output_prefix).name != args.output_prefix
            or not args.output_prefix.startswith(f"locust_{args.scenario}_")
        ):
            parser.error("--output-prefix must be a filename beginning with locust_<scenario>_")
        output_prefix = RAW_RESULTS_DIR / args.output_prefix
        if any(RAW_RESULTS_DIR.glob(f"{output_prefix.name}_*")):
            parser.error(f"Output prefix already exists: {output_prefix.name}")
    else:
        output_prefix = unique_csv_prefix(args.scenario)
    request_csv_path = output_prefix.with_name(f"{output_prefix.name}_requests.csv")
    metadata_path = output_prefix.with_name(f"{output_prefix.name}_metadata.json")
    stop_file = output_prefix.with_name(f"{output_prefix.name}_stop.request")

    configuration = {
        "scenario": args.scenario,
        "host": args.host,
        "peak_users": args.users,
        "duration_seconds": duration,
        "spawn_rate": args.spawn_rate,
    }
    if args.scenario == "spike":
        configuration["phases"] = {
            "warmup": {
                "users": low_users,
                "duration_seconds": warmup_duration,
                "spawn_rate": warmup_spawn_rate,
            },
            "peak": {
                "users": args.users,
                "duration_seconds": peak_duration,
                "spawn_rate": spike_spawn_rate,
            },
            "recovery": {
                "users": low_users,
                "duration_seconds": recovery_duration,
                "spawn_rate": recovery_spawn_rate,
            },
        }
    metadata = {
        "configuration": configuration,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "exit_code": None,
        "status": "running",
    }
    write_metadata(metadata_path, metadata)

    environment = os.environ.copy()
    environment.update(
        {
            "CLOUDPULSE_SCENARIO": args.scenario,
            "CLOUDPULSE_USERS": str(args.users),
            "CLOUDPULSE_DURATION": str(duration),
            "CLOUDPULSE_SPAWN_RATE": str(args.spawn_rate),
            "CLOUDPULSE_REQUESTS_CSV": str(request_csv_path),
            "CLOUDPULSE_STOP_FILE": str(stop_file),
        }
    )
    if args.scenario == "spike":
        environment.update(
            {
                "CLOUDPULSE_LOW_USERS": str(low_users),
                "CLOUDPULSE_WARMUP_DURATION": str(warmup_duration),
                "CLOUDPULSE_PEAK_DURATION": str(peak_duration),
                "CLOUDPULSE_RECOVERY_DURATION": str(recovery_duration),
                "CLOUDPULSE_WARMUP_SPAWN_RATE": str(warmup_spawn_rate),
                "CLOUDPULSE_SPIKE_SPAWN_RATE": str(spike_spawn_rate),
                "CLOUDPULSE_RECOVERY_SPAWN_RATE": str(recovery_spawn_rate),
            }
        )

    command = [
        sys.executable,
        "-m",
        "locust",
        "--locustfile",
        str(LOCUST_FILE),
        "--headless",
        "--host",
        args.host,
        "--csv",
        str(output_prefix),
        "--csv-full-history",
    ]
    print(f"Locust CSV prefix: {output_prefix}", flush=True)
    print(f"Request records: {request_csv_path}", flush=True)
    print(f"Run metadata: {metadata_path}", flush=True)
    exit_code = 1
    stop_requested = False
    try:
        if is_loopback_host(args.host):
            metadata["configuration"]["engine"] = "stdlib"
            write_metadata(metadata_path, metadata)
            print(
                "Using the standard-library loopback runner; Locust native modules are not imported.",
                flush=True,
            )
            stop_requested = run_standard_library_load(
                args, duration, request_csv_path, stop_file
            )
            exit_code = 130 if stop_requested else 0
        else:
            metadata["configuration"]["engine"] = "locust"
            write_metadata(metadata_path, metadata)
            exit_code, stop_requested = run_locust_subprocess(
                command, environment, stop_file
            )
    except KeyboardInterrupt:
        stop_file.touch(exist_ok=True)
        stop_requested = True
        exit_code = 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Experiment runner failed: {error}", file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        if stop_file.exists():
            stop_file.unlink()
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["exit_code"] = exit_code
        metadata["status"] = (
            "interrupted"
            if stop_requested and exit_code in {0, 130}
            else "completed"
            if exit_code == 0
            else "interrupted"
            if exit_code == 130
            else "failed"
        )
        write_metadata(metadata_path, metadata)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
