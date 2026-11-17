import argparse
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import re
import statistics


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "results" / "raw"
PROCESSED_DIR = BASE_DIR / "results" / "processed"
GRAPH_DIR = BASE_DIR / "results" / "graphs"
REQUEST_COLUMNS = (
    "timestamp_utc",
    "response_time_ms",
    "status_code",
    "exception",
)
RUN_ID_PATTERN = re.compile(r"locust_[A-Za-z0-9][A-Za-z0-9_-]*")
INTERVAL_SECONDS = 10
SUMMARY_COLUMNS = (
    "run_id",
    "request_count",
    "failed_requests",
    "duration_seconds",
    "throughput_rps",
    "average_latency_ms",
    "median_latency_ms",
    "p95_latency_ms",
)
COLORS = ("#087e8b", "#bd5b19", "#5368d5", "#9d3bb5", "#39854a", "#c23b55")


class ComparisonError(ValueError):
    pass


def _percentile(values, percentile):
    if not values:
        return ""
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def load_run(run_id, raw_dir=RAW_DIR):
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ComparisonError(f"Invalid run ID: {run_id!r}")
    request_path = Path(raw_dir) / f"{run_id}_requests.csv"
    if not request_path.is_file():
        raise FileNotFoundError(f"Request CSV not found: {request_path}")
    with request_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ComparisonError(f"Request CSV is empty: {request_path}")
        missing = [field for field in REQUEST_COLUMNS if field not in reader.fieldnames]
        if missing:
            raise ComparisonError(
                f"Request CSV is missing columns: {', '.join(missing)}"
            )
        latencies = []
        failures = 0
        failures_known = True
        requests = []
        for line, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.fromisoformat(
                    row["timestamp_utc"].replace("Z", "+00:00")
                )
            except (AttributeError, ValueError) as error:
                raise ComparisonError(
                    f"Malformed timestamp in {request_path} line {line}."
                ) from error
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ComparisonError(
                    f"Timestamp lacks timezone in {request_path} line {line}."
                )
            timestamp = timestamp.astimezone(timezone.utc)
            latency = None
            if row["response_time_ms"]:
                try:
                    latency = float(row["response_time_ms"])
                except ValueError as error:
                    raise ComparisonError(
                        f"Invalid response latency in {request_path} line {line}."
                    ) from error
                if not math.isfinite(latency) or latency < 0:
                    raise ComparisonError(
                        f"Invalid response latency in {request_path} line {line}."
                    )
                latencies.append(latency)
            status = (row["status_code"] or "").strip()
            exception = (row["exception"] or "").strip()
            if exception:
                failures += 1
            elif status:
                try:
                    failures += int(int(status) >= 400)
                except ValueError as error:
                    raise ComparisonError(
                        f"Invalid status code in {request_path} line {line}."
                    ) from error
            else:
                failures_known = False
            requests.append((timestamp, latency))
    if not requests:
        raise ComparisonError(f"Request CSV has no records: {request_path}")

    start = min(timestamp for timestamp, _ in requests)
    end = max(timestamp for timestamp, _ in requests)
    bins = {}
    for timestamp, latency in requests:
        elapsed = max(0.0, (timestamp - start).total_seconds())
        second = math.floor(elapsed / INTERVAL_SECONDS) * INTERVAL_SECONDS
        bucket = bins.setdefault(second, {"latencies": [], "requests": 0})
        bucket["requests"] += 1
        if latency is not None:
            bucket["latencies"].append(latency)
    elapsed = (end - start).total_seconds()
    metadata_path = Path(raw_dir) / f"{run_id}_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            configured_duration = metadata.get("configuration", {}).get(
                "duration_seconds"
            )
            if (
                isinstance(configured_duration, (int, float))
                and not isinstance(configured_duration, bool)
                and configured_duration > 0
            ):
                elapsed = float(configured_duration)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    duration = max(elapsed, 1.0)
    summary = {
        "run_id": run_id,
        "request_count": len(requests),
        "failed_requests": failures if failures_known else "",
        "duration_seconds": duration,
        "throughput_rps": len(requests) / duration,
        "average_latency_ms": (
            statistics.fmean(latencies) if latencies else ""
        ),
        "median_latency_ms": statistics.median(latencies) if latencies else "",
        "p95_latency_ms": _percentile(latencies, 0.95),
    }
    last_second = max(bins)
    elapsed_seconds = list(range(0, last_second + INTERVAL_SECONDS, INTERVAL_SECONDS))
    rate = []
    latency = []
    for second in elapsed_seconds:
        bucket = bins.get(second, {"requests": 0, "latencies": []})
        rate.append(bucket["requests"] / INTERVAL_SECONDS)
        samples = bucket["latencies"]
        latency.append(
            sum(samples) / len(samples) if samples else None
        )
    return {
        "summary": summary,
        "elapsed_seconds": elapsed_seconds,
        "request_rate": rate,
        "latency": latency,
    }


def compare_runs(run_ids, raw_dir=RAW_DIR):
    if len(run_ids) < 2:
        raise ComparisonError("Provide at least two run IDs to compare.")
    if len(set(run_ids)) != len(run_ids):
        raise ComparisonError("Run IDs must be unique.")
    return [load_run(run_id, raw_dir) for run_id in run_ids]


def _svg_chart(results):
    width, height = 1100, 760
    left, right = 95, 1060
    top1, bottom1 = 90, 350
    top2, bottom2 = 455, 715
    max_time = max(
        (max(run["elapsed_seconds"], default=0) for run in results),
        default=10,
    )
    max_time = max(max_time, 10)
    max_rate = max(
        (max(run["request_rate"], default=0) for run in results), default=0
    )
    max_latency = max(
        (
            max((value for value in run["latency"] if value is not None), default=0)
            for run in results
        ),
        default=0,
    )
    max_rate = max(max_rate * 1.1, 0.1)
    max_latency = max(max_latency * 1.1, 1)

    def x_pos(seconds):
        return left + seconds / max_time * (right - left)

    def y_pos(value, minimum, maximum, top, bottom):
        return bottom - (value / maximum) * (bottom - top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<rect width='100%' height='100%' fill='white'/>",
        "<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17212b}.grid{stroke:#d7dee5}.axis{stroke:#465563;stroke-width:1.4}.title{font-size:23px;font-weight:700}.label{font-size:14px;font-weight:600}.tick{font-size:12px}</style>",
        "<text class='title' x='95' y='38'>Saved CloudPulse run comparison</text>",
    ]
    for panel_index, (title, top, bottom, maximum, series_key) in enumerate(
        (
            ("Request rate (requests/second, 10-second bins)", top1, bottom1, max_rate, "request_rate"),
            ("Mean response latency (milliseconds, 10-second bins)", top2, bottom2, max_latency, "latency"),
        )
    ):
        parts.append(
            f"<text class='label' x='{left}' y='{top - 22}'>{html.escape(title)}</text>"
        )
        for tick in range(6):
            value = maximum * tick / 5
            y = bottom - (bottom - top) * tick / 5
            parts.append(
                f"<line class='grid' x1='{left}' y1='{y:.1f}' x2='{right}' y2='{y:.1f}'/>"
            )
            parts.append(
                f"<text class='tick' x='{left - 10}' y='{y + 4:.1f}' text-anchor='end'>{value:.1f}</text>"
            )
        parts.append(
            f"<line class='axis' x1='{left}' y1='{bottom}' x2='{right}' y2='{bottom}'/>"
        )
        for run_index, run in enumerate(results):
            values = run[series_key]
            points = []
            for index, value in enumerate(values):
                if value is None:
                    continue
                time_second = run["elapsed_seconds"][index]
                y = bottom - value / maximum * (bottom - top)
                points.append(f"{x_pos(time_second):.1f},{y:.1f}")
            if len(points) > 1:
                parts.append(
                    f"<polyline points='{' '.join(points)}' fill='none' stroke='{COLORS[run_index % len(COLORS)]}' stroke-width='2.5'/>"
                )
            elif points:
                x, y = points[0].split(",")
                parts.append(
                    f"<circle cx='{x}' cy='{y}' r='4' fill='{COLORS[run_index % len(COLORS)]}'/>"
                )
        parts.append(
            f"<text class='label' x='{(left + right) / 2}' y='{bottom + 35}' text-anchor='middle'>Elapsed time from each run start (seconds)</text>"
        )
    legend_y = 63
    legend_x = left
    for index, run in enumerate(results):
        label = html.escape(run["summary"]["run_id"])
        parts.append(
            f"<line x1='{legend_x}' y1='{legend_y}' x2='{legend_x + 24}' y2='{legend_y}' stroke='{COLORS[index % len(COLORS)]}' stroke-width='3'/>"
        )
        parts.append(
            f"<text class='tick' x='{legend_x + 30}' y='{legend_y + 4}'>{label}</text>"
        )
        legend_x += min(400, 40 + len(run["summary"]["run_id"]) * 7)
        if legend_x > 900:
            legend_x = left
            legend_y += 18
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_comparison(results, processed_dir=PROCESSED_DIR, graph_dir=GRAPH_DIR, stamp=None):
    processed_dir = Path(processed_dir)
    graph_dir = Path(graph_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = processed_dir / f"comparison_{stamp}_summary.csv"
    graph_path = graph_dir / f"comparison_{stamp}.svg"
    with summary_path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(result["summary"] for result in results)
    with graph_path.open("x", encoding="utf-8") as output:
        output.write(_svg_chart(results))
    return summary_path, graph_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare two or more saved CloudPulse request CSV runs."
    )
    parser.add_argument("run_ids", nargs="+")
    args = parser.parse_args(argv)
    try:
        results = compare_runs(args.run_ids)
        summary_path, graph_path = write_comparison(results)
    except (ComparisonError, OSError) as error:
        parser.exit(1, f"Comparison failed: {error}\n")
    print(f"Summary CSV: {summary_path}")
    print(f"Comparison chart: {graph_path}")
    for result in results:
        print(result["summary"])


if __name__ == "__main__":
    main()
