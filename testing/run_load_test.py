import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


BASE_DIR = Path(__file__).resolve().parent.parent
LOCUST_FILE = BASE_DIR / "testing" / "locustfile.py"
RAW_RESULTS_DIR = BASE_DIR / "results" / "raw"
SCENARIOS = ("steady", "spike", "ramp", "periodic")


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


def main():
    parser = argparse.ArgumentParser(description="Run a CloudPulse Locust workload scenario")
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--users", type=positive_int, default=10, help="Peak concurrent Locust users")
    parser.add_argument("--duration", type=positive_int, help="Total duration in seconds")
    parser.add_argument("--spawn-rate", type=positive_int, default=5, help="Users started per second")
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
    environment = os.environ.copy()
    environment.update(
        {
            "CLOUDPULSE_SCENARIO": args.scenario,
            "CLOUDPULSE_USERS": str(args.users),
            "CLOUDPULSE_DURATION": str(duration),
            "CLOUDPULSE_SPAWN_RATE": str(args.spawn_rate),
            "CLOUDPULSE_REQUESTS_CSV": str(request_csv_path),
            "CLOUDPULSE_STOP_FILE": str(
                output_prefix.with_name(f"{output_prefix.name}_stop.request")
            ),
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
    stop_forced = False
    stop_file = output_prefix.with_name(f"{output_prefix.name}_stop.request")
    stop_requested_at = None
    try:
        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        )
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
                        stop_forced = True
                        print("Locust did not stop after the request; terminating its process tree.", flush=True)
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
    except KeyboardInterrupt:
        stop_file.touch(exist_ok=True)
        exit_code = 130
    finally:
        if stop_file.exists():
            stop_file.unlink()
        if stop_requested and not stop_forced and exit_code == 0:
            exit_code = 130
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
