from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "request_rate_rps",
]


def _parse_timestamps(values: pd.Series, timezone: Optional[str]) -> pd.Series:
    """Parse timestamps without imposing a timezone when none was configured."""
    parsed_values = []
    for value in values:
        timestamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(timestamp):
            parsed_values.append(pd.NaT)
            continue

        if timezone is not None:
            try:
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize(timezone)
                else:
                    timestamp = timestamp.tz_convert(timezone)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Could not apply timezone '{timezone}': {error}") from error
        parsed_values.append(timestamp)

    return pd.Series(parsed_values, index=values.index)


def validate_workload_quality(
    df: pd.DataFrame,
    expected_sampling_interval: Optional[Union[str, pd.Timedelta]] = None,
    timezone: Optional[str] = None,
) -> Dict[str, Any]:
    """Return workload quality counts and the CSV row numbers behind each issue.

    This function inspects the input order and does not remove or repair rows.
    Row numbers are one-based CSV line numbers (the header is line 1).
    """
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    interval = None
    if expected_sampling_interval is not None:
        interval = pd.Timedelta(expected_sampling_interval)
        if interval <= pd.Timedelta(0):
            raise ValueError("expected_sampling_interval must be greater than zero")

    parsed_timestamps = _parse_timestamps(df["timestamp"], timezone)
    numeric_rates = pd.to_numeric(df["request_rate_rps"], errors="coerce")
    missing_timestamp_rows = []
    invalid_timestamp_rows = []
    missing_rate_rows = []
    non_numeric_rate_rows = []
    negative_rate_rows = []
    duplicate_timestamp_rows = []
    out_of_order_rows = []
    sampling_gap_rows = []

    previous_timestamp = None
    seen_timestamps = set()
    for position, (original_timestamp, timestamp, original_rate, rate) in enumerate(
        zip(df["timestamp"], parsed_timestamps, df["request_rate_rps"], numeric_rates)
    ):
        row_number = position + 2
        if pd.isna(original_timestamp) or str(original_timestamp).strip() == "":
            missing_timestamp_rows.append(row_number)
        elif pd.isna(timestamp):
            invalid_timestamp_rows.append(row_number)
        else:
            timestamp_key = timestamp.value
            if timestamp_key in seen_timestamps:
                duplicate_timestamp_rows.append(row_number)
            seen_timestamps.add(timestamp_key)

            if previous_timestamp is not None:
                try:
                    delta = timestamp - previous_timestamp
                    if delta < pd.Timedelta(0):
                        out_of_order_rows.append(row_number)
                    elif interval is not None and delta > interval:
                        sampling_gap_rows.append(row_number)
                except TypeError:
                    out_of_order_rows.append(row_number)
            previous_timestamp = timestamp

        if pd.isna(original_rate) or str(original_rate).strip() == "":
            missing_rate_rows.append(row_number)
        elif pd.isna(rate):
            non_numeric_rate_rows.append(row_number)
        elif rate < 0:
            negative_rate_rows.append(row_number)

    row_issues = {
        "missing_timestamps": missing_timestamp_rows,
        "invalid_timestamps": invalid_timestamp_rows,
        "missing_request_rates": missing_rate_rows,
        "non_numeric_request_rates": non_numeric_rate_rows,
        "negative_request_rates": negative_rate_rows,
        "duplicate_timestamps": duplicate_timestamp_rows,
        "out_of_order_timestamps": out_of_order_rows,
        "unexpected_sampling_gaps": sampling_gap_rows,
    }
    report = {name: len(rows) for name, rows in row_issues.items()}
    report["total_rows"] = int(len(df))
    report["problem_rows"] = row_issues
    report["expected_sampling_interval"] = str(interval) if interval is not None else None
    report["timezone"] = timezone
    return report


def load_workload_csv(
    csv_path: Union[str, Path],
    timezone: Optional[str] = None,
    expected_sampling_interval: Optional[Union[str, pd.Timedelta]] = None,
) -> pd.DataFrame:
    """Load workload data and validate data quality.

    Timezone and sampling interval are optional configuration. With no timezone,
    parsed timestamps keep their supplied timezone (or remain timezone-naive).
    With no sampling interval, no sampling-gap checks are performed.

    Parameters
    ----------
    csv_path : Union[str, Path]
        Path to the workload CSV file.

    Returns
    -------
    pd.DataFrame
        A parsed DataFrame sorted chronologically by timestamp, with invalid
        timestamps last. All input rows are retained; invalid timestamps become
        NaT and missing or non-numeric rates become NaN. A validation report is
        attached as ``df.attrs["validation_report"]``. The source CSV is unchanged.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the CSV is empty or required columns are missing.
    """
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"CSV is empty: {path}") from error

    if df.empty:
        raise ValueError(f"CSV contains no workload rows: {path}")

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.copy()
    report = validate_workload_quality(
        df,
        expected_sampling_interval=expected_sampling_interval,
        timezone=timezone,
    )
    df["timestamp"] = _parse_timestamps(df["timestamp"], timezone)
    df["request_rate_rps"] = pd.to_numeric(df["request_rate_rps"], errors="coerce")

    # Keep the established sorted return value while retaining input-order findings.
    df["_cloudpulse_sort_key"] = df["timestamp"].map(
        lambda value: value.value if not pd.isna(value) else float("inf")
    )
    df = df.sort_values("_cloudpulse_sort_key", kind="stable").drop(
        columns="_cloudpulse_sort_key"
    ).reset_index(drop=True)
    df.attrs["validation_report"] = report

    print("=== CloudPulse Data Validation ===")
    print(f"Total rows: {report['total_rows']}")
    for issue_name, row_numbers in report["problem_rows"].items():
        if row_numbers:
            print(f"{issue_name}: {len(row_numbers)} (CSV rows {row_numbers})")
    if not any(report["problem_rows"].values()):
        print("No data-quality problems detected.")
    print(f"Timezone: {timezone if timezone is not None else 'not configured'}")
    print(
        "Expected sampling interval: "
        f"{report['expected_sampling_interval'] or 'not configured'}"
    )

    return df
