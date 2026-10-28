
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "results" / "raw" / "request_logs.csv"
OUTPUT_FILE = BASE_DIR / "results" / "processed" / "workload_10s.csv"

INTERVAL = "10s"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        print("No request records found.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")

    workload = df.resample(INTERVAL).agg(
        request_count=("path", "count"),
        average_latency_ms=("duration_ms", "mean"),
        error_count=("status_code", lambda s: (s >= 400).sum()),
    )

    workload["request_rate_rps"] = (
        workload["request_count"] / 10
    )

    workload = workload.reset_index()
    workload.to_csv(OUTPUT_FILE, index=False)

    print(f"Input records: {len(df)}")
    print(f"Workload intervals: {len(workload)}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print(workload.to_string(index=False))


if __name__ == "__main__":
    main()
