"""Plot saved CloudPulse comparison, telemetry, and pod-distribution data."""

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .validate_data import validate_csv
except ImportError:
    from validate_data import validate_csv


BASE_DIR = Path(__file__).resolve().parents[1]
GRAPH_DIR = BASE_DIR / "results" / "graphs"
COLORS = ("#087e8b", "#bd5b19", "#5368d5", "#9d3bb5", "#39854a", "#c23b55")


class PlotError(ValueError):
    pass


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                raise PlotError(f"CSV has no header: {path}")
            return reader.fieldnames, list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise PlotError(f"Could not read {path}: {error}") from error


def _float_or_none(value: Optional[str], column: str) -> Optional[float]:
    if value is None or not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise PlotError(f"{column} contains a non-numeric value: {value!r}") from error
    if not math.isfinite(parsed):
        raise PlotError(f"{column} contains a non-finite value")
    return parsed


def _write_svg(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(content)
    return path


def _line_svg(
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[Tuple[str, Sequence[Optional[float]]]],
    path: Path,
) -> Path:
    width, height = 1000, 570
    left, right, top, bottom = 90, 960, 82, 455
    values = [value for _, items in series for value in items if value is not None]
    maximum = max(values, default=0.0)
    maximum = max(maximum * 1.12, 1.0)
    count = max((len(items) for _, items in series), default=0)
    count = max(count, 1)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<rect width='100%' height='100%' fill='white'/>",
        "<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17212b}.grid{stroke:#dbe2e8}.axis{stroke:#465563;stroke-width:1.3}.title{font-size:23px;font-weight:700}.label{font-size:14px;font-weight:600}.tick{font-size:12px}</style>",
        f"<text class='title' x='{left}' y='38'>{html.escape(title)}</text>",
        f"<text class='label' x='22' y='{(top + bottom) / 2}' transform='rotate(-90 22 {(top + bottom) / 2})' text-anchor='middle'>{html.escape(y_label)}</text>",
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = bottom - (bottom - top) * tick / 5
        pieces.append(
            f"<line class='grid' x1='{left}' y1='{y:.1f}' x2='{right}' y2='{y:.1f}'/>"
        )
        pieces.append(
            f"<text class='tick' x='{left - 10}' y='{y + 4:.1f}' text-anchor='end'>{value:.2f}</text>"
        )
    pieces.extend(
        [
            f"<line class='axis' x1='{left}' y1='{bottom}' x2='{right}' y2='{bottom}'/>",
        ]
    )
    elapsed_seconds = (count - 1) * 10
    for tick in range(6):
        x = left + (right - left) * tick / 5
        label = f"{elapsed_seconds * tick / 5:.0f}s"
        pieces.append(
            f"<line class='axis' x1='{x:.1f}' y1='{bottom}' x2='{x:.1f}' y2='{bottom + 5}'/>"
        )
        pieces.append(
            f"<text class='tick' x='{x:.1f}' y='{bottom + 20}' text-anchor='middle'>{label}</text>"
        )
    pieces.append(
        f"<text class='label' x='{(left + right) / 2}' y='535' text-anchor='middle'>{html.escape(x_label)}</text>"
    )
    for series_index, (label, items) in enumerate(series):
        color = COLORS[series_index % len(COLORS)]
        points = []
        segments = []
        current_segment = []
        for index, value in enumerate(items):
            if value is None:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                continue
            x = left if count == 1 else left + index / (count - 1) * (right - left)
            y = bottom - value / maximum * (bottom - top)
            points.append((x, y))
            current_segment.append((x, y))
        if current_segment:
            segments.append(current_segment)
        for segment in segments:
            if len(segment) < 2:
                continue
            coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in segment)
            pieces.append(
                f"<polyline points='{coordinates}' fill='none' stroke='{color}' stroke-width='2.5'/>"
            )
        for x, y in points:
            pieces.append(
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.5' fill='{color}'/>"
            )
        legend_x = left + series_index * 190
        pieces.append(
            f"<line x1='{legend_x}' y1='490' x2='{legend_x + 22}' y2='490' stroke='{color}' stroke-width='3'/>"
        )
        pieces.append(
            f"<text class='tick' x='{legend_x + 28}' y='494'>{html.escape(label)}</text>"
        )
        if not any(value is not None for value in items):
            pieces.append(
                f"<text class='tick' x='{(left + right) / 2}' y='{(top + bottom) / 2 + series_index * 18}' text-anchor='middle'>No recorded {html.escape(label.lower())} values</text>"
            )
    pieces.append("</svg>")
    return _write_svg(path, "\n".join(pieces) + "\n")


def _bar_svg(
    title: str,
    y_label: str,
    labels: Sequence[str],
    series: Sequence[Tuple[str, Sequence[Optional[float]]]],
    path: Path,
) -> Path:
    width, height = 1100, 600
    left, right, top, bottom = 95, 1060, 90, 455
    all_values = [value for _, values in series for value in values if value is not None]
    maximum = max(max(all_values, default=0.0) * 1.15, 1.0)
    group_count = max(len(labels), 1)
    group_width = (right - left) / group_count
    bar_count = max(len(series), 1)
    bar_width = min(42.0, group_width * 0.72 / bar_count)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<rect width='100%' height='100%' fill='white'/>",
        "<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17212b}.grid{stroke:#dbe2e8}.axis{stroke:#465563;stroke-width:1.3}.title{font-size:23px;font-weight:700}.label{font-size:14px;font-weight:600}.tick{font-size:12px}</style>",
        f"<text class='title' x='{left}' y='40'>{html.escape(title)}</text>",
        f"<text class='label' x='24' y='{(top + bottom) / 2}' transform='rotate(-90 24 {(top + bottom) / 2})' text-anchor='middle'>{html.escape(y_label)}</text>",
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = bottom - (bottom - top) * tick / 5
        pieces.append(
            f"<line class='grid' x1='{left}' y1='{y:.1f}' x2='{right}' y2='{y:.1f}'/>"
        )
        pieces.append(
            f"<text class='tick' x='{left - 9}' y='{y + 4:.1f}' text-anchor='end'>{value:.2f}</text>"
        )
    if not all_values:
        pieces.append(
            f"<text class='tick' x='{(left + right) / 2}' y='{(top + bottom) / 2}' text-anchor='middle'>No recorded values</text>"
        )
    for index, label in enumerate(labels):
        center = left + (index + 0.5) * group_width
        pieces.append(
            f"<text class='tick' x='{center:.1f}' y='{bottom + 24}' text-anchor='middle'>{html.escape(label)}</text>"
        )
        for series_index, (series_name, values) in enumerate(series):
            value = values[index] if index < len(values) else None
            if value is None:
                continue
            x = center + (series_index - (bar_count - 1) / 2) * bar_width
            bar_height = value / maximum * (bottom - top)
            pieces.append(
                f"<rect x='{x - bar_width * 0.42:.1f}' y='{bottom - bar_height:.1f}' width='{bar_width * 0.84:.1f}' height='{bar_height:.1f}' fill='{COLORS[series_index % len(COLORS)]}'><title>{html.escape(series_name)}: {value:g}</title></rect>"
            )
    for index, (name, _) in enumerate(series):
        x = left + index * 190
        pieces.append(
            f"<rect x='{x}' y='535' width='14' height='14' fill='{COLORS[index % len(COLORS)]}'/>"
        )
        pieces.append(
            f"<text class='tick' x='{x + 21}' y='547'>{html.escape(name)}</text>"
        )
    pieces.append("</svg>")
    return _write_svg(path, "\n".join(pieces) + "\n")


def _save_png_from_svg_data(
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[Tuple[str, Sequence[Optional[float]]]],
    path: Path,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise PlotError("PNG output requires matplotlib; use --format svg instead") from error
    figure, axis = plt.subplots(figsize=(11, 5.5))
    for label, values in series:
        axis.plot(
            range(len(values)),
            [float("nan") if value is None else value for value in values],
            marker="o",
            label=label,
        )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    if series:
        axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=150)
    except OSError:
        plt.close(figure)
        raise
    plt.close(figure)
    return path


def _save_png_bar_chart(
    title: str,
    y_label: str,
    labels: Sequence[str],
    series: Sequence[Tuple[str, Sequence[Optional[float]]]],
    path: Path,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise PlotError("PNG output requires matplotlib; use --format svg instead") from error
    figure, axis = plt.subplots(figsize=(11, 5.5))
    positions = list(range(len(labels)))
    bar_count = max(len(series), 1)
    width = 0.8 / bar_count
    for series_index, (name, values) in enumerate(series):
        centers = [
            position + (series_index - (bar_count - 1) / 2) * width
            for position in positions
        ]
        axis.bar(
            centers,
            [float("nan") if value is None else value for value in values],
            width=width,
            label=name,
        )
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.grid(axis="y", alpha=0.3)
    if series:
        axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=150)
    except OSError:
        plt.close(figure)
        raise
    plt.close(figure)
    return path


def _plot_contract(path: Path, output_dir: Path, output_format: str) -> List[Path]:
    validation = validate_csv(path)
    if not validation["valid"] or validation["format"] != "contract_10s":
        raise PlotError("Input is not a valid 10-second contract CSV: " + "; ".join(validation["errors"]))
    _, rows = _read_csv(path)
    stem = path.stem
    plots = (
        (
            "latency",
            "CloudPulse response latency",
            "Elapsed 10-second samples",
            "Latency (ms)",
            (
                (
                    "Average latency",
                    [_float_or_none(row["average_latency_ms"], "average_latency_ms") for row in rows],
                ),
                (
                    "P95 latency",
                    [_float_or_none(row["p95_latency_ms"], "p95_latency_ms") for row in rows],
                ),
            ),
        ),
        (
            "throughput",
            "CloudPulse request throughput",
            "Elapsed 10-second samples",
            "Requests/second",
            (
                (
                    "Request rate",
                    [_float_or_none(row["request_rate_rps"], "request_rate_rps") for row in rows],
                ),
            ),
        ),
        (
            "replicas",
            "CloudPulse replica count",
            "Elapsed 10-second samples",
            "Replicas",
            (
                (
                    "Replica count",
                    [_float_or_none(row["replica_count"], "replica_count") for row in rows],
                ),
            ),
        ),
    )
    outputs = []
    for suffix, title, x_label, y_label, series in plots:
        svg_path = output_dir / f"{stem}_{suffix}.svg"
        if output_format in ("svg", "both"):
            outputs.append(_line_svg(title, x_label, y_label, series, svg_path))
        if output_format in ("png", "both"):
            png_path = output_dir / f"{stem}_{suffix}.png"
            outputs.append(
                _save_png_from_svg_data(title, x_label, y_label, series, png_path)
            )
    return outputs


def _plot_comparison(path: Path, output_dir: Path, output_format: str) -> List[Path]:
    fields, rows = _read_csv(path)
    required = (
        "run_id",
        "throughput_rps",
        "average_latency_ms",
        "p95_latency_ms",
    )
    missing = [field for field in required if field not in fields]
    if missing or not rows:
        raise PlotError(
            "Comparison summary is missing required columns or rows: "
            + ", ".join(missing)
        )
    labels = [row["run_id"] for row in rows]
    latency = (
        (
            "Average latency",
            [_float_or_none(row["average_latency_ms"], "average_latency_ms") for row in rows],
        ),
        (
            "P95 latency",
            [_float_or_none(row["p95_latency_ms"], "p95_latency_ms") for row in rows],
        ),
    )
    throughput = (
        ("Throughput", [_float_or_none(row["throughput_rps"], "throughput_rps") for row in rows]),
    )
    outputs = []
    if output_format in ("svg", "both"):
        outputs.append(
            _bar_svg(
                "Saved-run latency comparison",
                "Latency (ms)",
                labels,
                latency,
                output_dir / f"{path.stem}_latency.svg",
            )
        )
        outputs.append(
            _bar_svg(
                "Saved-run throughput comparison",
                "Requests/second",
                labels,
                throughput,
                output_dir / f"{path.stem}_throughput.svg",
            )
        )
    if output_format in ("png", "both"):
        outputs.extend(
            _save_png_bar_chart(
                title,
                ylabel,
                labels,
                series,
                output_dir / f"{path.stem}_{suffix}.png",
            )
            for suffix, title, ylabel, series in (
                ("latency", "Saved-run latency comparison", "Latency (ms)", latency),
                ("throughput", "Saved-run throughput comparison", "Requests/second", throughput),
            )
        )
    return outputs


def _plot_distribution(
    path: Path, output_dir: Path, output_format: str
) -> List[Path]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlotError(f"Could not read distribution JSON: {error}") from error
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not results:
        raise PlotError("Distribution JSON must contain a nonempty results list")
    modes: List[str] = []
    pod_names = set()
    counts_by_mode = {}
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("pod_counts"), dict):
            raise PlotError("Every distribution result must contain pod_counts")
        mode = str(result.get("mode", "unknown"))
        modes.append(mode)
        counts = result["pod_counts"]
        counts_by_mode[mode] = counts
        for pod, count in counts.items():
            if isinstance(count, bool) or not isinstance(count, (int, float)) or count < 0:
                raise PlotError(f"Invalid request count for pod {pod!r}")
            pod_names.add(str(pod))
    labels = sorted(pod_names)
    series = [
        (mode, [float(counts_by_mode[mode].get(pod, 0)) for pod in labels])
        for mode in modes
    ]
    outputs = []
    if output_format in ("svg", "both"):
        outputs.append(
            _bar_svg(
                "Per-pod request distribution",
                "Actual response count",
                labels,
                series,
                output_dir / f"{path.stem}_distribution.svg",
            )
        )
    if output_format in ("png", "both"):
        outputs.append(
            _save_png_bar_chart(
                "Per-pod request distribution",
                "Actual response count",
                labels,
                series,
                output_dir / f"{path.stem}_distribution.png",
            )
        )
    return outputs


def plot_results(
    input_path: str | Path,
    output_dir: str | Path = GRAPH_DIR,
    output_format: str = "svg",
) -> List[Path]:
    source = Path(input_path)
    destination = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")
    if output_format not in {"svg", "png", "both"}:
        raise ValueError("output_format must be svg, png, or both")
    if source.suffix.lower() == ".json":
        return _plot_distribution(source, destination, output_format)
    fields, _ = _read_csv(source)
    if "timestamp" in fields and "request_rate_rps" in fields:
        return _plot_contract(source, destination, output_format)
    return _plot_comparison(source, destination, output_format)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot saved CloudPulse contract, comparison, or distribution data."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=GRAPH_DIR)
    parser.add_argument("--format", choices=("svg", "png", "both"), default="svg")
    arguments = parser.parse_args(argv)
    try:
        outputs = plot_results(arguments.input, arguments.output_dir, arguments.format)
    except (FileNotFoundError, OSError, PlotError, ValueError) as error:
        parser.exit(1, f"Plot generation failed: {error}\n")
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
