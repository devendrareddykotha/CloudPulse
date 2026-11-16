"""Prepare a documented single-service MCR proxy from the Alibaba archive."""

import argparse
import math
import zipfile
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = Path.home() / "Downloads" / "UpolishedData.zip"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "prediction"
    / "data"
    / "processed"
    / "alibaba_consumer_mcr_service_proxy.csv"
)
MCR_FILES = ("dataset-mcr0.csv", "dataset-mcr1.csv", "dataset-mcr2.csv")
RAW_COLUMNS = ["field1", "timestamp_ms", "msname", "msinstanceid", "metrics", "value"]
SERVICE_COLUMNS = ["timestamp_ms", "msname", "msinstanceid", "metrics", "value"]
EXPECTED_METRIC = "consumerMQ_MCR"


def aggregate_service_records(
    records: pd.DataFrame,
    service_id: str,
    expected_time_points: Optional[int] = 90,
    expected_interval_ms: Optional[int] = 60_000,
) -> pd.DataFrame:
    """Sum one service's per-instance MCR values at each timestamp.

    The returned ``timestamp`` is an elapsed-millisecond offset represented as
    UTC time from the Unix epoch solely to fit the existing CSV loader. It is
    not a claim about the trace's real wall-clock start date.
    """
    missing_columns = [column for column in SERVICE_COLUMNS if column not in records]
    if missing_columns:
        raise ValueError(f"Missing required source fields: {missing_columns}")
    if not isinstance(service_id, str) or not service_id.strip():
        raise ValueError("service_id must be a non-empty string")
    if expected_time_points is not None and expected_time_points <= 0:
        raise ValueError("expected_time_points must be positive or None")
    if expected_interval_ms is not None and expected_interval_ms <= 0:
        raise ValueError("expected_interval_ms must be positive or None")

    selected = records.loc[records["msname"] == service_id, SERVICE_COLUMNS].copy()
    if selected.empty:
        raise ValueError(f"No records found for service id: {service_id}")

    observed_metrics = set(selected["metrics"].dropna().unique())
    if observed_metrics != {EXPECTED_METRIC}:
        raise ValueError(
            f"Expected only {EXPECTED_METRIC!r} for the selected service; "
            f"found {sorted(observed_metrics)}"
        )

    selected["timestamp_ms"] = pd.to_numeric(
        selected["timestamp_ms"], errors="coerce"
    )
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    if selected[["timestamp_ms", "value"]].isna().any().any():
        raise ValueError("Selected records contain missing or nonnumeric timestamps/values")
    if not selected["timestamp_ms"].map(math.isfinite).all():
        raise ValueError("Selected records contain non-finite timestamps")
    if not selected["value"].map(math.isfinite).all():
        raise ValueError("Selected records contain non-finite MCR values")
    if (selected["timestamp_ms"] % 1 != 0).any():
        raise ValueError("Timestamp offsets must be whole milliseconds")
    if (selected["value"] < 0).any():
        raise ValueError("MCR values must not be negative")
    if selected["msinstanceid"].isna().any():
        raise ValueError("Selected records contain missing instance identifiers")

    duplicate_keys = ["timestamp_ms", "msname", "msinstanceid", "metrics"]
    if selected.duplicated(subset=duplicate_keys).any():
        raise ValueError("Duplicate service-instance metric records were found")

    offsets = sorted(selected["timestamp_ms"].astype("int64").unique().tolist())
    if expected_time_points is not None and len(offsets) != expected_time_points:
        raise ValueError(
            f"Expected {expected_time_points} timestamps for the selected service; "
            f"found {len(offsets)}"
        )
    if expected_interval_ms is not None and len(offsets) > 1:
        differences = pd.Series(offsets).diff().dropna()
        if not (differences == expected_interval_ms).all():
            raise ValueError(
                f"Timestamp offsets are not continuous at {expected_interval_ms} ms"
            )

    grouped = (
        selected.groupby("timestamp_ms", sort=True)
        .agg(
            request_rate_rps=("value", "sum"),
            instance_count=("msinstanceid", "nunique"),
        )
        .reset_index()
    )
    grouped["timestamp"] = pd.to_datetime(
        grouped["timestamp_ms"], unit="ms", origin="unix", utc=True
    )
    grouped["service_id"] = service_id
    grouped["metric"] = EXPECTED_METRIC
    return grouped[
        [
            "timestamp",
            "request_rate_rps",
            "timestamp_ms",
            "service_id",
            "metric",
            "instance_count",
        ]
    ]


def prepare_workload_csv(
    archive_path: Union[str, Path],
    service_id: str,
    output_csv: Union[str, Path] = DEFAULT_OUTPUT,
    expected_time_points: Optional[int] = 90,
    expected_interval_ms: Optional[int] = 60_000,
    chunk_size: int = 100_000,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Read the archive, validate a selected service, and save its proxy series.

    Only the explicitly selected service is aggregated. Its MCR values are
    summed across instance IDs at each timestamp; this is a service-level proxy,
    not a validated cluster-wide incoming request rate. The unexplained leading
    source field is retained in the archive and omitted from this derived file.
    """
    archive = Path(archive_path).expanduser().resolve()
    output = Path(output_csv).expanduser().resolve()
    if archive == output:
        raise ValueError("Output path must not overwrite the source archive")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {output}. "
            "Choose another path or pass overwrite=True explicitly."
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    selected_parts = []
    with zipfile.ZipFile(archive) as zip_file:
        available_names = set(zip_file.namelist())
        missing_members = [name for name in MCR_FILES if name not in available_names]
        if missing_members:
            raise ValueError(f"Archive is missing expected MCR files: {missing_members}")

        for name in MCR_FILES:
            with zip_file.open(name) as raw:
                chunks = pd.read_csv(
                    raw,
                    header=None,
                    names=RAW_COLUMNS,
                    chunksize=chunk_size,
                )
                for chunk in chunks:
                    matched = chunk.loc[
                        chunk["msname"] == service_id, SERVICE_COLUMNS
                    ]
                    if not matched.empty:
                        selected_parts.append(matched)

    if not selected_parts:
        raise ValueError(f"No records found for service id: {service_id}")

    selected_records = pd.concat(selected_parts, ignore_index=True)
    workload = aggregate_service_records(
        selected_records,
        service_id=service_id,
        expected_time_points=expected_time_points,
        expected_interval_ms=expected_interval_ms,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    workload.to_csv(output, index=False)
    return workload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare one selected service's Alibaba consumer-MCR proxy series."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--service-id",
        required=True,
        help="The selected id1/msname candidate to aggregate; no service is chosen implicitly.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-time-points", type=int, default=90)
    parser.add_argument("--expected-interval-ms", type=int, default=60_000)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing the specified processed output file.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        workload = prepare_workload_csv(
            archive_path=arguments.archive,
            service_id=arguments.service_id,
            output_csv=arguments.output,
            expected_time_points=arguments.expected_time_points,
            expected_interval_ms=arguments.expected_interval_ms,
            chunk_size=arguments.chunk_size,
            overwrite=arguments.overwrite,
        )
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))

    print(f"Prepared rows: {len(workload)}")
    print(f"Selected service id: {arguments.service_id}")
    print("Aggregation: sum consumerMQ_MCR across instances per timestamp")
    print("Interpretation: selected-service proxy, not cluster-wide demand")
    print(f"Output CSV: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())