import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

METRICS_URL = "http://localhost:8093/metrics/request-rate"
SAMPLE_COUNT = 15
SAMPLE_INTERVAL_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 10

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "data" / "live"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def collect_metrics():
    request = Request(
        METRICS_URL,
        headers={"Accept": "application/json"},
    )

    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP status: {response.status}")

        payload = json.loads(response.read().decode("utf-8"))

    count = payload["requests_last_60_seconds"]
    rate = payload["request_rate_rps"]
    window = payload["window_seconds"]
    metric_timestamp = payload["timestamp"]

    if not isinstance(count, int) or count < 0:
        raise ValueError("Invalid request count")

    if not isinstance(rate, (int, float)) or rate < 0:
        raise ValueError("Invalid request rate")

    if window != 60:
        raise ValueError(f"Unexpected metrics window: {window}")

    return {
        "metric_timestamp_utc": metric_timestamp,
        "requests_last_60_seconds": count,
        "request_rate_rps": rate,
        "window_seconds": window,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect live CloudPulse request-rate metrics.")
    parser.add_argument(
        "--samples",
        type=int,
        default=SAMPLE_COUNT,
        help=f"number of observations to collect (default: {SAMPLE_COUNT})",
    )
    arguments = parser.parse_args()
    if arguments.samples < 1:
        parser.error("--samples must be at least 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename_time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_file = OUTPUT_DIR / f"live_metrics_{filename_time}.csv"

    suffix = 1
    while output_file.exists():
        output_file = OUTPUT_DIR / f"live_metrics_{filename_time}_{suffix}.csv"
        suffix += 1

    columns = [
        "sample_number",
        "collected_at_utc",
        "metric_timestamp_utc",
        "requests_last_60_seconds",
        "request_rate_rps",
        "window_seconds",
        "source_url",
        "status",
    ]

    print(f"Collecting {arguments.samples} observations.")
    print(f"Interval: {SAMPLE_INTERVAL_SECONDS} seconds")
    print(f"Output: {output_file}")

    with output_file.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for sample_number in range(1, arguments.samples + 1):
            row = {
                "sample_number": sample_number,
                "collected_at_utc": utc_now(),
                "metric_timestamp_utc": "",
                "requests_last_60_seconds": "",
                "request_rate_rps": "",
                "window_seconds": "",
                "source_url": METRICS_URL,
                "status": "",
            }

            try:
                metrics = collect_metrics()
                row.update(metrics)
                row["status"] = "success"

                print(
                    f"Sample {sample_number}/{arguments.samples}: "
                    f"{metrics['request_rate_rps']} RPS, "
                    f"{metrics['requests_last_60_seconds']} requests"
                )

            except (
                URLError,
                HTTPError,
                TimeoutError,
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
                json.JSONDecodeError,
            ) as error:
                row["status"] = f"error: {error}"
                print(f"Sample {sample_number} failed: {error}")

            writer.writerow(row)
            file.flush()

            if sample_number < arguments.samples:
                print(f"Waiting {SAMPLE_INTERVAL_SECONDS} seconds...")
                time.sleep(SAMPLE_INTERVAL_SECONDS)

    print("\nCollection finished.")
    print(f"CSV saved: {output_file}")


if __name__ == "__main__":
    main()
