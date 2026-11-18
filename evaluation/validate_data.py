"""Validate CloudPulse shared-contract and per-second telemetry CSV files."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


CONTRACT_COLUMNS = (
    "timestamp",
    "request_count",
    "request_rate_rps",
    "cpu_utilization_pct",
    "memory_usage_bytes",
    "replica_count",
    "average_latency_ms",
    "p95_latency_ms",
    "error_count",
)
PER_SECOND_COLUMNS = (
    "run_id",
    "scenario",
    "timestamp_utc",
    "elapsed_second",
    "requests_per_second",
    "request_count",
    "mean_response_latency_ms",
    "latency_sample_count",
    "failures",
    "error_rate",
    "failure_sample_count",
    "configured_peak_user_count",
    "test_duration_seconds",
)
def _parse_aware_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _number(value: str, column: str, line_number: int, errors: List[str]) -> Optional[float]:
    if value is None or not value.strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(f"line {line_number}: {column} is not numeric")
        return None
    if not math.isfinite(parsed):
        errors.append(f"line {line_number}: {column} must be finite")
        return None
    if parsed < 0:
        errors.append(f"line {line_number}: {column} must not be negative")
        return None
    return parsed


def validate_csv(csv_path: str | Path) -> Dict[str, object]:
    """Return validation findings without filling, sorting, or changing source data."""
    path = Path(csv_path)
    errors: List[str] = []
    warnings: List[str] = []
    missing_value_counts: Dict[str, int] = {}
    report: Dict[str, object] = {
        "path": str(path),
        "valid": False,
        "format": None,
        "row_count": 0,
        "errors": errors,
        "warnings": warnings,
        "missing_value_counts": missing_value_counts,
    }
    if not path.is_file():
        errors.append(f"CSV file not found: {path}")
        return report

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            fieldnames = reader.fieldnames or []
            if len(fieldnames) != len(set(fieldnames)):
                errors.append("CSV contains duplicate column names")
            if all(column in fieldnames for column in CONTRACT_COLUMNS):
                csv_format = "contract_10s"
                required = CONTRACT_COLUMNS
                timestamp_column = "timestamp"
                count_columns = ("request_count", "error_count", "replica_count")
                rate_columns = ("request_rate_rps",)
                numeric_columns = (
                    "cpu_utilization_pct",
                    "memory_usage_bytes",
                    "average_latency_ms",
                    "p95_latency_ms",
                )
            elif all(column in fieldnames for column in PER_SECOND_COLUMNS):
                csv_format = "per_second"
                required = PER_SECOND_COLUMNS
                timestamp_column = "timestamp_utc"
                count_columns = (
                    "request_count",
                    "failures",
                    "latency_sample_count",
                    "failure_sample_count",
                    "configured_peak_user_count",
                    "test_duration_seconds",
                )
                rate_columns = ("requests_per_second", "error_rate")
                numeric_columns = ("mean_response_latency_ms",)
            else:
                errors.append(
                    "CSV columns do not match the 9-column contract or the "
                    "evaluation/generate_dataset.py per_second schema"
                )
                return report
            report["format"] = csv_format
            missing_columns = [column for column in required if column not in fieldnames]
            if missing_columns:
                errors.append("missing required columns: " + ", ".join(missing_columns))
                return report

            previous_timestamp = None
            rows = 0
            for line_number, row in enumerate(reader, start=2):
                rows += 1
                if None in row:
                    errors.append(f"line {line_number}: extra values beyond the header")
                raw_timestamp = row.get(timestamp_column)
                try:
                    if raw_timestamp is None or not raw_timestamp.strip():
                        raise ValueError("timestamp is missing")
                    timestamp = _parse_aware_timestamp(raw_timestamp)
                except (TypeError, ValueError) as error:
                    errors.append(f"line {line_number}: invalid {timestamp_column}: {error}")
                    timestamp = None
                if timestamp is not None:
                    if previous_timestamp is not None and timestamp <= previous_timestamp:
                        errors.append(
                            f"line {line_number}: timestamps are not strictly chronological"
                        )
                    previous_timestamp = timestamp

                for column in (*count_columns, *rate_columns, *numeric_columns):
                    raw_value = row.get(column)
                    if raw_value is None or not raw_value.strip():
                        missing_value_counts[column] = (
                            missing_value_counts.get(column, 0) + 1
                        )
                        if column in count_columns or column in rate_columns:
                            warnings.append(
                                f"line {line_number}: {column} is missing; value remains unknown"
                            )
                        continue
                    value = _number(raw_value, column, line_number, errors)
                    if value is not None and column in count_columns and not value.is_integer():
                        errors.append(f"line {line_number}: {column} must be an integer")
                if csv_format == "per_second":
                    missing_request_rate = row.get("requests_per_second")
                    if missing_request_rate and missing_request_rate.strip():
                        request_count = _number(
                            row.get("request_count", ""),
                            "request_count",
                            line_number,
                            errors,
                        )
                        request_rate = _number(
                            missing_request_rate,
                            "requests_per_second",
                            line_number,
                            errors,
                        )
                        if (
                            request_count is not None
                            and request_rate is not None
                            and request_rate != request_count
                        ):
                            warnings.append(
                                f"line {line_number}: requests_per_second differs "
                                "from request_count; both source values are preserved"
                            )
            report["row_count"] = rows
            if rows == 0:
                errors.append("CSV contains no data rows")
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        errors.append(f"could not read CSV: {error}")

    for column, count in sorted(missing_value_counts.items()):
        if count:
            warnings.insert(
                0,
                f"{column}: {count} missing value(s) left unchanged",
            )
    report["valid"] = not errors
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a CloudPulse 10-second contract or per_second CSV."
    )
    parser.add_argument("csv_path", type=Path)
    arguments = parser.parse_args(argv)
    report = validate_csv(arguments.csv_path)
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
