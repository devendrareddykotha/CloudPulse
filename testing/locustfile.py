
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from threading import Lock

from locust import HttpUser, LoadTestShape, constant_throughput, events, task


REQUEST_FIELDS = [
    "timestamp_utc",
    "method",
    "name",
    "url",
    "status_code",
    "response_time_ms",
    "response_length",
    "exception",
]
request_log_file = None
request_writer = None
request_log_lock = Lock()


@events.test_start.add_listener
def open_request_log(environment, **kwargs):
    global request_log_file, request_writer

    request_log_path = os.getenv("CLOUDPULSE_REQUESTS_CSV")
    if not request_log_path:
        return

    request_log_file = Path(request_log_path).open(
        "w", newline="", encoding="utf-8"
    )
    request_writer = csv.DictWriter(request_log_file, fieldnames=REQUEST_FIELDS)
    request_writer.writeheader()


@events.request.add_listener
def record_request(
    request_type,
    name,
    response_time,
    response_length,
    response,
    context,
    exception,
    start_time,
    url,
    **kwargs,
):
    if request_writer is None:
        return

    timestamp = (
        datetime.fromtimestamp(start_time, timezone.utc).isoformat()
        if start_time is not None
        else datetime.now(timezone.utc).isoformat()
    )
    record = {
        "timestamp_utc": timestamp,
        "method": request_type,
        "name": name,
        "url": url,
        "status_code": getattr(response, "status_code", ""),
        "response_time_ms": response_time,
        "response_length": response_length,
        "exception": str(exception) if exception else "",
    }
    with request_log_lock:
        request_writer.writerow(record)
        request_log_file.flush()


@events.test_stop.add_listener
def close_request_log(environment, **kwargs):
    global request_log_file, request_writer

    if request_log_file is not None:
        request_log_file.close()
    request_log_file = None
    request_writer = None


class CloudPulseUser(HttpUser):
    wait_time = constant_throughput(1)

    def on_start(self):
        self.iteration = 0

    @task
    def exercise_service(self):
        self.iteration += 1
        if self.iteration % 6 == 0:
            self.client.get("/health", name="/health")
        else:
            self.client.get("/workload", name="/workload")


class CloudPulseLoadShape(LoadTestShape):
    def __init__(self):
        super().__init__()
        self.scenario = os.getenv("CLOUDPULSE_SCENARIO", "steady")
        self.users = int(os.getenv("CLOUDPULSE_USERS", "10"))
        self.duration = int(os.getenv("CLOUDPULSE_DURATION", "120"))
        self.spawn_rate = int(os.getenv("CLOUDPULSE_SPAWN_RATE", "5"))
        self.low_users = int(
            os.getenv("CLOUDPULSE_LOW_USERS", str(max(1, self.users // 2)))
        )
        default_warmup = max(1, self.duration // 4)
        default_peak = max(1, self.duration // 2)
        default_recovery = self.duration - default_warmup - default_peak
        self.warmup_duration = int(
            os.getenv("CLOUDPULSE_WARMUP_DURATION", str(default_warmup))
        )
        self.peak_duration = int(
            os.getenv("CLOUDPULSE_PEAK_DURATION", str(default_peak))
        )
        self.recovery_duration = int(
            os.getenv("CLOUDPULSE_RECOVERY_DURATION", str(default_recovery))
        )
        self.warmup_spawn_rate = int(
            os.getenv("CLOUDPULSE_WARMUP_SPAWN_RATE", str(self.spawn_rate))
        )
        self.spike_spawn_rate = int(
            os.getenv("CLOUDPULSE_SPIKE_SPAWN_RATE", str(self.spawn_rate))
        )
        self.recovery_spawn_rate = int(
            os.getenv("CLOUDPULSE_RECOVERY_SPAWN_RATE", str(self.spawn_rate))
        )

        if self.scenario not in {"steady", "spike", "ramp", "periodic"}:
            raise ValueError(f"Unsupported CloudPulse scenario: {self.scenario}")
        if self.users < 1 or self.duration < 1 or self.spawn_rate < 1:
            raise ValueError("Users, duration, and spawn rate must be positive")
        if self.scenario == "spike":
            phase_durations = (
                self.warmup_duration,
                self.peak_duration,
                self.recovery_duration,
            )
            phase_spawn_rates = (
                self.warmup_spawn_rate,
                self.spike_spawn_rate,
                self.recovery_spawn_rate,
            )
            if self.low_users < 1 or self.low_users >= self.users:
                raise ValueError("Spike low users must be positive and below peak users")
            if min(phase_durations) < 1 or min(phase_spawn_rates) < 1:
                raise ValueError("Spike phase durations and spawn rates must be positive")
            self.duration = sum(phase_durations)
        if self.scenario == "periodic" and self.duration < 8:
            raise ValueError("The periodic scenario requires a duration of at least 8 seconds")

    def tick(self):
        stop_file = os.getenv("CLOUDPULSE_STOP_FILE")
        if stop_file and Path(stop_file).is_file():
            return None

        elapsed = self.get_run_time()
        if elapsed >= self.duration:
            return None

        if self.scenario == "spike":
            warmup_end = self.warmup_duration
            peak_end = warmup_end + self.peak_duration
            if elapsed < warmup_end:
                return self.low_users, self.warmup_spawn_rate
            if elapsed < peak_end:
                return self.users, self.spike_spawn_rate
            return self.low_users, self.recovery_spawn_rate

        low_users = self.users // 2
        if self.scenario == "steady":
            user_count = self.users
        elif self.scenario == "ramp":
            user_count = math.ceil(self.users * elapsed / self.duration)
        else:
            cycle_duration = self.duration / 4
            high_phase = elapsed % cycle_duration >= cycle_duration / 2
            user_count = self.users if high_phase else low_users

        return user_count, self.spawn_rate
