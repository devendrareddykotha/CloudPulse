
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "results" / "raw" / "request_logs.csv"
OUTPUT_FILE = BASE_DIR / "results" / "processed" / "workload_10s.csv"

INTERVAL = "10s"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    with INPUT_FILE.open(newline="", encoding="utf-8") as source:
        records = list(csv.DictReader(source))

    if not records:
        print("No request records found.")
        return

    interval_seconds = 10
    buckets = defaultdict(lambda: {"request_count": 0, "latencies": [], "error_count": 0})
    bucket_timestamps = []

    for record in records:
        timestamp = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamp = timestamp.astimezone(timezone.utc)
        bucket = int(timestamp.timestamp()) // interval_seconds * interval_seconds
        bucket_timestamps.append(bucket)

        values = buckets[bucket]
        if record.get("path") not in (None, ""):
            values["request_count"] += 1
        try:
            values["latencies"].append(float(record["duration_ms"]))
        except (KeyError, TypeError, ValueError):
            pass
        try:
            if float(record["status_code"]) >= 400:
                values["error_count"] += 1
        except (KeyError, TypeError, ValueError):
            pass

    rows = []
    for bucket in range(min(bucket_timestamps), max(bucket_timestamps) + interval_seconds, interval_seconds):
        values = buckets[bucket]
        latencies = values["latencies"]
        rows.append({
            "timestamp": datetime.fromtimestamp(bucket, timezone.utc).isoformat(),
            "request_count": values["request_count"],
            "average_latency_ms": sum(latencies) / len(latencies) if latencies else "",
            "error_count": values["error_count"],
            "request_rate_rps": values["request_count"] / interval_seconds,
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Input records: {len(records)}")
    print(f"Workload intervals: {len(rows)}")
    print(f"Output saved to: {OUTPUT_FILE}")
    for row in rows:
        print(" ".join(str(value) for value in row.values()))


if __name__ == "__main__":
    main()
