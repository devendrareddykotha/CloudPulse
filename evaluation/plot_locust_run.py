import argparse
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "results" / "raw"
GRAPH_DIR = BASE_DIR / "results" / "graphs"
PROCESSED_DIR = BASE_DIR / "results" / "processed"


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_float(value, field, source):
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field} value in {source}: {value!r}") from error


def svg_chart(title, subtitle, ylabel, values, origin, duration, phases, output_path):
    width = 1080
    height = 540
    left = 88
    right = 1030
    top = 100
    bottom = 445
    plot_width = right - left
    plot_height = bottom - top
    last_second = max(values, default=0)
    x_max = max(duration, last_second + 1, 1)
    numeric_values = [value for value in values.values() if value is not None]
    y_max = max(numeric_values, default=0)
    y_max = max(1, math.ceil(y_max / 5) * 5)

    def x_position(seconds):
        return left + seconds / x_max * plot_width

    def y_position(value):
        return bottom - value / y_max * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<rect width='100%' height='100%' fill='#ffffff'/>",
        "<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17212b}.grid{stroke:#d7dee5;stroke-width:1}.axis{stroke:#465563;stroke-width:1.4}.tick{font-size:12px}.title{font-size:23px;font-weight:700}.subtitle{font-size:13px;fill:#52616f}.phase{font-size:12px;font-weight:600}.axislabel{font-size:13px;font-weight:600}</style>",
        f'<text class="title" x="{left}" y="38">{html.escape(title)}</text>',
        f'<text class="subtitle" x="{left}" y="63">{html.escape(subtitle)}</text>',
    ]

    phase_colors = {
        "warmup": "#dceef8",
        "spike": "#f8c9a9",
        "sustained_peak": "#fce2df",
        "recovery": "#e3f2e4",
    }
    phase_labels = {
        "warmup": "Warm-up",
        "spike": "Spike",
        "sustained_peak": "Sustained peak",
        "recovery": "Recovery",
    }
    for phase_name, start, end in phases:
        x1 = x_position(start)
        x2 = x_position(min(end, x_max))
        if x2 > x1:
            parts.append(
                f'<rect x="{x1:.2f}" y="{top}" width="{x2 - x1:.2f}" height="{plot_height}" fill="{phase_colors[phase_name]}" opacity="0.78"/>'
            )
            if x2 - x1 >= 70:
                parts.append(
                    f'<text class="phase" x="{(x1 + x2) / 2:.2f}" y="{top + 18}" text-anchor="middle">{phase_labels[phase_name]}</text>'
                )
    if x_max > duration:
        x1 = x_position(duration)
        parts.append(
            f'<rect x="{x1:.2f}" y="{top}" width="{right - x1:.2f}" height="{plot_height}" fill="#edf0f2" opacity="0.9"/>'
        )
        parts.append(
            f'<text class="phase" x="{(x1 + right) / 2:.2f}" y="{top + 18}" text-anchor="middle">Drain</text>'
        )

    legend_x = left
    for phase_name, _, _ in phases:
        parts.append(
            f'<rect x="{legend_x}" y="80" width="12" height="12" fill="{phase_colors[phase_name]}"/>'
        )
        parts.append(
            f'<text class="phase" x="{legend_x + 18}" y="90">{phase_labels[phase_name]}</text>'
        )
        legend_x += 110 if phase_name in {"warmup", "recovery"} else 170

    for tick in range(6):
        value = y_max * tick / 5
        y = y_position(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{value:.0f}</text>')

    x_ticks = sorted(set([0, duration, *range(0, x_max + 1, 5)]))
    for tick in x_ticks:
        if tick > x_max:
            continue
        x = x_position(tick)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{bottom + 22}" text-anchor="middle">{tick}</text>')

    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
            f'<text class="axislabel" x="{(left + right) / 2}" y="{height - 28}" text-anchor="middle">Elapsed time from {origin.strftime("%H:%M:%S UTC")} (seconds)</text>',
            f'<text class="axislabel" transform="translate(24 {(top + bottom) / 2}) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>',
        ]
    )

    segments = []
    current_segment = []
    for second in sorted(values):
        value = values[second]
        if value is None:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            continue
        current_segment.append(f"{x_position(second + 0.5):.2f},{y_position(value):.2f}")
    if current_segment:
        segments.append(current_segment)
    color = "#087e8b" if "requests" in ylabel.lower() else "#bd5b19" if "latency" in ylabel.lower() else "#bd2d2d"
    for segment in segments:
        if len(segment) == 1:
            x, y = segment[0].split(",")
            parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}"/>')
        else:
            parts.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Create graphs and a report from a saved Locust run")
    parser.add_argument("run_prefix", help="Run filename prefix without the _stats.csv suffix")
    parser.add_argument(
        "--output-tag",
        help="Optional filename tag, useful for preserving and creating a second analysis of a run",
    )
    args = parser.parse_args()
    if Path(args.run_prefix).name != args.run_prefix:
        parser.error("run_prefix must be a filename prefix, not a path")
    if args.output_tag and not all(
        character.isascii() and (character.isalnum() or character in "-_")
        for character in args.output_tag
    ):
        parser.error("output-tag may contain only ASCII letters, numbers, hyphens, and underscores")

    prefix = args.run_prefix
    output_prefix = f"{prefix}_{args.output_tag}" if args.output_tag else prefix
    sources = {
        "metadata": RAW_DIR / f"{prefix}_metadata.json",
        "stats": RAW_DIR / f"{prefix}_stats.csv",
        "history": RAW_DIR / f"{prefix}_stats_history.csv",
        "requests": RAW_DIR / f"{prefix}_requests.csv",
        "failures": RAW_DIR / f"{prefix}_failures.csv",
        "exceptions": RAW_DIR / f"{prefix}_exceptions.csv",
    }
    missing = [path for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing run input(s): " + ", ".join(str(path) for path in missing))

    metadata = json.loads(sources["metadata"].read_text(encoding="utf-8"))
    if metadata.get("status") != "completed" or metadata.get("exit_code") != 0:
        raise ValueError("The selected Locust run did not complete successfully")
    if metadata.get("configuration", {}).get("scenario") != "spike":
        raise ValueError("The selected run is not a spike scenario")

    stats_rows = read_csv(sources["stats"])
    summary = next((row for row in stats_rows if row.get("Name") == "Aggregated"), None)
    if summary is None:
        raise ValueError("The Locust statistics file has no Aggregated row")
    requests = read_csv(sources["requests"])
    history_rows = read_csv(sources["history"])
    failure_rows = read_csv(sources["failures"])
    exception_rows = read_csv(sources["exceptions"])
    if not requests:
        raise ValueError("The per-request CSV contains no request records")

    for row in requests:
        row["parsed_timestamp"] = datetime.fromisoformat(row["timestamp_utc"])
        row["elapsed_seconds"] = None
        row["response_time_value"] = parse_float(
            row["response_time_ms"], "response_time_ms", sources["requests"]
        )
        row["is_failure"] = bool(row["exception"].strip()) or int(row["status_code"] or 0) >= 400

    aggregate_history = [
        row for row in history_rows
        if row.get("Name") == "Aggregated" and row.get("Timestamp")
    ]
    if not aggregate_history:
        raise ValueError("The history CSV has no aggregated time-series rows")
    first_history_timestamp = min(
        parse_float(row["Timestamp"], "Timestamp", sources["history"])
        for row in aggregate_history
    )
    origin = datetime.fromtimestamp(first_history_timestamp, timezone.utc)
    for row in requests:
        row["elapsed_seconds"] = (row["parsed_timestamp"] - origin).total_seconds()

    config = metadata["configuration"]
    phases_config = config["phases"]
    warmup_duration = int(phases_config["warmup"]["duration_seconds"])
    peak_duration = int(phases_config["peak"]["duration_seconds"])
    recovery_duration = int(phases_config["recovery"]["duration_seconds"])
    duration = warmup_duration + peak_duration + recovery_duration
    if duration != int(config["duration_seconds"]):
        raise ValueError("Metadata phase durations do not add up to the recorded total duration")
    spike_duration = min(1, peak_duration)
    phases = [
        ("warmup", 0, warmup_duration),
        ("spike", warmup_duration, warmup_duration + spike_duration),
    ]
    if peak_duration > spike_duration:
        phases.append(
            (
                "sustained_peak",
                warmup_duration + spike_duration,
                warmup_duration + peak_duration,
            )
        )
    phases.append(("recovery", warmup_duration + peak_duration, duration))
    phase_users = {
        "warmup": phases_config["warmup"]["users"],
        "spike": phases_config["peak"]["users"],
        "sustained_peak": phases_config["peak"]["users"],
        "recovery": phases_config["recovery"]["users"],
    }

    phase_counts = {name: 0 for name, _, _ in phases}
    tail_count = 0
    pre_run_count = 0
    per_second = {}
    for row in requests:
        elapsed = row["elapsed_seconds"]
        second = math.floor(elapsed)
        bucket = per_second.setdefault(second, {"count": 0, "latencies": [], "failures": 0})
        bucket["count"] += 1
        bucket["latencies"].append(row["response_time_value"])
        bucket["failures"] += int(row["is_failure"])
        if elapsed < 0:
            pre_run_count += 1
        elif elapsed >= duration:
            tail_count += 1
        else:
            for name, start, end in phases:
                if start <= elapsed < end:
                    phase_counts[name] += 1
                    break

    stats_requests = int(summary["Request Count"])
    stats_failures = int(summary["Failure Count"])
    request_failures = sum(int(row["is_failure"]) for row in requests)
    if stats_requests != len(requests):
        raise ValueError(
            f"Request count mismatch: stats={stats_requests}, request records={len(requests)}"
        )
    if stats_failures != request_failures:
        raise ValueError(
            f"Failure count mismatch: stats={stats_failures}, request records={request_failures}"
        )

    last_second = max(per_second)
    request_rates = {second: per_second.get(second, {}).get("count", 0) for second in range(last_second + 1)}
    latency_by_second = {
        second: (
            sum(per_second[second]["latencies"]) / len(per_second[second]["latencies"])
            if second in per_second else None
        )
        for second in range(last_second + 1)
    }
    failures_by_second = {second: per_second.get(second, {}).get("failures", 0) for second in range(last_second + 1)}

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    graph_paths = {
        "requests_per_second": GRAPH_DIR / f"{output_prefix}_requests_per_second.svg",
        "latency_over_time": GRAPH_DIR / f"{output_prefix}_latency_over_time.svg",
        "failures_over_time": GRAPH_DIR / f"{output_prefix}_failures_over_time.svg",
    }
    report_path = PROCESSED_DIR / f"{output_prefix}_report.md"
    output_paths = [*graph_paths.values(), report_path]
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing output(s): " + ", ".join(str(path) for path in existing))

    chart_subtitle = (
        "One-second bins; shaded bands distinguish warm-up, the first peak second (spike), "
        "sustained peak, and recovery."
    )
    chart_specs = [
        ("Requests per second", "Requests / second", request_rates, graph_paths["requests_per_second"]),
        ("Average response latency per second", "Latency (ms)", latency_by_second, graph_paths["latency_over_time"]),
        ("Failures per second", "Failures / second", failures_by_second, graph_paths["failures_over_time"]),
    ]
    for title, ylabel, values, path in chart_specs:
        content = svg_chart(title, chart_subtitle, ylabel, values, origin, duration, phases, path)
        with path.open("x", encoding="utf-8", newline="") as file:
            file.write(content)

    started = datetime.fromisoformat(metadata["started_at_utc"])
    finished = datetime.fromisoformat(metadata["finished_at_utc"])
    wall_duration = (finished - started).total_seconds()
    failure_percentage = 100 * stats_failures / stats_requests if stats_requests else None
    summary_values = {
        "total_requests": stats_requests,
        "failures": stats_failures,
        "failure_percentage": failure_percentage,
        "throughput": parse_float(summary["Requests/s"], "Requests/s", sources["stats"]),
        "average_latency": parse_float(summary["Average Response Time"], "Average Response Time", sources["stats"]),
        "median_latency": parse_float(summary["Median Response Time"], "Median Response Time", sources["stats"]),
        "p95_latency": parse_float(summary["95%"], "95%", sources["stats"]),
        "p99_latency": parse_float(summary["99%"], "99%", sources["stats"]) if summary.get("99%") else None,
        "min_latency": parse_float(summary["Min Response Time"], "Min Response Time", sources["stats"]),
        "max_latency": parse_float(summary["Max Response Time"], "Max Response Time", sources["stats"]),
    }

    phase_lines = [
        "| Phase | Scheduled users | Scheduled duration | Requests | Mean requests/s |",
        "|---|---:|---:|---:|---:|",
    ]
    phase_labels = {
        "warmup": "Warm-up",
        "spike": "Spike",
        "sustained_peak": "Sustained peak",
        "recovery": "Recovery",
    }
    for name, start, end in phases:
        count = phase_counts[name]
        phase_lines.append(
            f"| {phase_labels[name]} | {phase_users[name]} | {end - start} s | "
            f"{count} | {count / (end - start):.2f} |"
        )
    phase_lines.append(f"| Post-schedule drain | N/A | N/A | {tail_count} | N/A |")
    if pre_run_count:
        phase_lines.append(f"| Before history origin | N/A | N/A | {pre_run_count} | N/A |")

    transitions = []
    last_users = None
    for row in sorted(aggregate_history, key=lambda item: float(item["Timestamp"])):
        user_count = int(float(row["User Count"]))
        if user_count != last_users:
            elapsed = int(float(row["Timestamp"]) - first_history_timestamp)
            transitions.append(f"{elapsed}s: {user_count} users")
            last_users = user_count

    source_lines = [f"- `{path.relative_to(BASE_DIR).as_posix()}`" for path in sources.values()]
    graph_lines = [f"- `{path.relative_to(BASE_DIR).as_posix()}`" for path in graph_paths.values()]
    report = f"""# Spike experiment report

Run prefix: `{prefix}`
Scenario: `{config['scenario']}`
Status: `{metadata['status']}` (exit code {metadata['exit_code']})

## Configuration and timing

- Started: {metadata['started_at_utc']}
- Finished: {metadata['finished_at_utc']}
- Scheduled duration: {duration} seconds
- Runner wall-clock duration: {wall_duration:.2f} seconds (includes Locust startup and shutdown)
- Phase clock origin: first aggregate history timestamp, {origin.isoformat()}

## Measured summary

| Metric | Value |
|---|---:|
| Total requests | {summary_values['total_requests']} |
| Failed requests | {summary_values['failures']} ({summary_values['failure_percentage']:.2f}%) |
| Average throughput | {summary_values['throughput']:.2f} requests/second |
| Average latency | {summary_values['average_latency']:.2f} ms |
| Median latency | {summary_values['median_latency']:.2f} ms |
| 95th-percentile latency | {summary_values['p95_latency']:.2f} ms |
| Scheduled duration | {duration} seconds |
| Runner wall-clock duration | {wall_duration:.2f} seconds |
| 99th-percentile latency | {f"{summary_values['p99_latency']:.2f} ms" if summary_values['p99_latency'] is not None else "Unavailable in Locust summary CSV"} |
| Minimum / maximum latency | {summary_values['min_latency']:.2f} / {summary_values['max_latency']:.2f} ms |
| Failure CSV rows | {len(failure_rows)} |
| Exception CSV rows | {len(exception_rows)} |

## Request activity by phase

{chr(10).join(phase_lines)}

Phase rates divide request records by each phase's displayed duration. Metadata defines warm-up, peak, and recovery; because it does not specify a separate spike duration, the spike is shown as the first one-second bin at the start of the configured peak, with the remaining peak time labeled sustained peak. Request timestamps are bucketed relative to the first aggregate history timestamp. {tail_count} request record(s) occurred at or after the scheduled end and are shown separately as post-schedule drain. {pre_run_count} request record(s) occurred before the history origin.

Observed user-count changes in Locust history: {"; ".join(transitions)}.

## Graphs

{chr(10).join(graph_lines)}

The initial spike interval is a visualization convention aligned to the observed transition from warm-up to peak; it is not a separately configured phase duration. Requests per second and failures are counted in one-second bins. Latency is the mean of per-request response times in each one-second bin; bins without requests are left blank. Failure records are identified by an HTTP status of 400 or higher or a recorded exception.

## Source files read

{chr(10).join(source_lines)}

This is a local service experiment. It does not measure Kubernetes autoscaling.
"""
    with report_path.open("x", encoding="utf-8", newline="") as file:
        file.write(report)

    print("Created outputs:")
    for path in [*graph_paths.values(), report_path]:
        print(path.relative_to(BASE_DIR).as_posix())
    print(f"Requests: {stats_requests}; failures: {stats_failures}; request rows: {len(requests)}")
    print(f"Phase counts: {phase_counts}; post-schedule drain: {tail_count}")


if __name__ == "__main__":
    main()
