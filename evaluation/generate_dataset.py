import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
import tempfile


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "results" / "raw"
OUTPUT_DIR = BASE_DIR / "results" / "processed" / "datasets"
SCHEMA_VERSION = "1.1"
RUN_ID_PATTERN = re.compile(
    r"locust_(steady|spike|ramp|periodic)_\d{8}T\d{6}_\d{6}Z(?:_[A-Za-z0-9-]+)?"
)
REQUEST_COLUMNS = (
    "timestamp_utc",
    "method",
    "name",
    "url",
    "status_code",
    "response_time_ms",
    "response_length",
    "exception",
)
REQUEST_OUTPUT_COLUMNS = (
    "run_id",
    "scenario",
    "timestamp_utc",
    "elapsed_seconds",
    "method",
    "name",
    "url",
    "status_code",
    "response_time_ms",
    "response_length",
    "exception",
    "failed",
    "configured_peak_user_count",
    "test_duration_seconds",
)
AGGREGATE_COLUMNS = (
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


class DatasetValidationError(ValueError):
    pass


def _parse_aware_timestamp(value, field, source):
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise DatasetValidationError(
            f"Invalid {field} timestamp in {source}: {value!r}"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DatasetValidationError(
            f"{field} timestamp in {source} must include a timezone."
        )
    return parsed.astimezone(timezone.utc)


def _read_run(run_id, raw_dir):
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise DatasetValidationError(f"Invalid CloudPulse run ID: {run_id!r}")

    metadata_path = raw_dir / f"{run_id}_metadata.json"
    requests_path = raw_dir / f"{run_id}_requests.csv"
    if not metadata_path.is_file():
        raise DatasetValidationError(f"Metadata file not found: {metadata_path}")
    if not requests_path.is_file():
        raise DatasetValidationError(f"Request CSV not found: {requests_path}")

    try:
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetValidationError(
            f"Could not read valid metadata from {metadata_path}: {error}"
        ) from error
    if not isinstance(metadata, dict) or not isinstance(
        metadata.get("configuration"), dict
    ):
        raise DatasetValidationError(
            f"Metadata must contain a configuration object: {metadata_path}"
        )

    configuration = metadata["configuration"]
    scenario = configuration.get("scenario")
    if scenario != RUN_ID_PATTERN.fullmatch(run_id).group(1):
        raise DatasetValidationError(
            f"Metadata scenario does not match run ID: {metadata_path}"
        )
    if metadata.get("status") not in ("completed", "interrupted", "failed"):
        raise DatasetValidationError(
            f"Run must be finished before dataset generation: {metadata_path}"
        )
    try:
        started_at_value = metadata["started_at_utc"]
        duration_seconds = configuration["duration_seconds"]
        users = configuration["peak_users"]
    except KeyError as error:
        raise DatasetValidationError(
            f"Metadata is missing valid duration, user count, or start time: {metadata_path}"
        ) from error
    if (
        not isinstance(duration_seconds, int)
        or isinstance(duration_seconds, bool)
        or not isinstance(users, int)
        or isinstance(users, bool)
        or duration_seconds < 1
        or users < 1
    ):
        raise DatasetValidationError(
            f"Metadata duration and user count must be positive: {metadata_path}"
        )
    started_at = _parse_aware_timestamp(
        started_at_value, "started_at_utc", metadata_path
    )

    try:
        requests_bytes = requests_path.read_bytes()
        if not requests_bytes:
            raise DatasetValidationError(f"Request CSV is empty: {requests_path}")
        decoded_requests = requests_bytes.decode("utf-8-sig")
        reader = csv.DictReader(StringIO(decoded_requests, newline=""))
        fields = reader.fieldnames or []
        missing_columns = [column for column in REQUEST_COLUMNS if column not in fields]
        if missing_columns:
            raise DatasetValidationError(
                f"Request CSV {requests_path} is missing required columns: "
                + ", ".join(missing_columns)
            )
        if len(fields) != len(set(fields)):
            raise DatasetValidationError(
                f"Request CSV contains duplicate column names: {requests_path}"
            )
        input_rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise DatasetValidationError(
            f"Could not read request CSV {requests_path}: {error}"
        ) from error
    if not input_rows:
        raise DatasetValidationError(
            f"Request CSV contains a header but no request records: {requests_path}"
        )

    records = []
    seen_records = set()
    warnings = []
    errors = []
    for line_number, row in enumerate(input_rows, start=2):
        if None in row:
            errors.append(f"line {line_number}: extra values beyond the CSV header")
            continue
        raw_signature = tuple(row.get(column, "") or "" for column in REQUEST_COLUMNS)
        if raw_signature in seen_records:
            errors.append(f"line {line_number}: duplicate request record")
            continue
        seen_records.add(raw_signature)

        try:
            timestamp = _parse_aware_timestamp(
                row["timestamp_utc"], "timestamp_utc", f"{requests_path}:{line_number}"
            )
        except DatasetValidationError as error:
            errors.append(str(error))
            continue
        elapsed_seconds = (timestamp - started_at).total_seconds()
        if elapsed_seconds < 0:
            errors.append(
                f"line {line_number}: request timestamp precedes run start"
            )
            continue

        latency_value = (row["response_time_ms"] or "").strip()
        if latency_value:
            try:
                latency = float(latency_value)
                if not math.isfinite(latency) or latency < 0:
                    raise ValueError("must be finite and nonnegative")
            except ValueError as error:
                errors.append(
                    f"line {line_number}: invalid response_time_ms {latency_value!r}"
                )
                continue
        else:
            latency = None
            warnings.append(
                f"line {line_number}: response_time_ms is missing; latency is unknown"
            )

        status_value = (row["status_code"] or "").strip()
        if status_value:
            try:
                status_code = int(status_value)
            except ValueError:
                errors.append(
                    f"line {line_number}: invalid status_code {status_value!r}"
                )
                continue
        else:
            status_code = None
            warnings.append(
                f"line {line_number}: status_code is missing; failure outcome may be unknown"
            )

        exception = row["exception"] or ""
        failed = (
            True
            if exception.strip()
            else status_code >= 400
            if status_code is not None
            else None
        )
        records.append(
            {
                "timestamp": timestamp,
                "elapsed_seconds": elapsed_seconds,
                "elapsed_second": math.floor(elapsed_seconds),
                "method": row["method"] or "",
                "name": row["name"] or "",
                "url": row["url"] or "",
                "status_code": status_code,
                "response_time_ms": latency,
                "response_length": row["response_length"] or "",
                "exception": exception,
                "failed": failed,
            }
        )

    if errors:
        raise DatasetValidationError(
            f"Invalid request records in {requests_path}:\n- "
            + "\n- ".join(errors)
        )
    if not records:
        raise DatasetValidationError(
            f"Request CSV contains no valid request records: {requests_path}"
        )

    records.sort(
        key=lambda record: (
            record["timestamp"],
            record["method"],
            record["name"],
            record["url"],
            record["status_code"] if record["status_code"] is not None else -1,
            record["response_time_ms"]
            if record["response_time_ms"] is not None
            else -1,
            record["response_length"],
            record["exception"],
        )
    )
    return {
        "run_id": run_id,
        "scenario": scenario,
        "status": metadata["status"],
        "started_at": started_at,
        "duration_seconds": duration_seconds,
        "users": users,
        "records": records,
        "warnings": sorted(warnings),
        "metadata_bytes": metadata_bytes,
        "requests_bytes": requests_bytes,
    }


def _csv_content(fieldnames, rows):
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _dataset_contents(run):
    run_id = run["run_id"]
    scenario = run["scenario"]
    started_at = run["started_at"]
    users = run["users"]
    duration = run["duration_seconds"]
    request_rows = []
    bins = {}
    for record in run["records"]:
        request_rows.append(
            {
                "run_id": run_id,
                "scenario": scenario,
                "timestamp_utc": record["timestamp"].isoformat(),
                "elapsed_seconds": record["elapsed_seconds"],
                "method": record["method"],
                "name": record["name"],
                "url": record["url"],
                "status_code": record["status_code"],
                "response_time_ms": record["response_time_ms"],
                "response_length": record["response_length"],
                "exception": record["exception"],
                "failed": record["failed"],
                "configured_peak_user_count": users,
                "test_duration_seconds": duration,
            }
        )
        bucket = bins.setdefault(
            record["elapsed_second"],
            {"requests": [], "latencies": [], "failures": [], "unknown_failure": 0},
        )
        bucket["requests"].append(record)
        if record["response_time_ms"] is not None:
            bucket["latencies"].append(record["response_time_ms"])
        if record["failed"] is None:
            bucket["unknown_failure"] += 1
        else:
            bucket["failures"].append(record["failed"])

    aggregate_rows = []
    for elapsed_second in sorted(bins):
        bucket = bins[elapsed_second]
        request_count = len(bucket["requests"])
        latency_complete = len(bucket["latencies"]) == request_count
        failure_complete = bucket["unknown_failure"] == 0
        failure_count = (
            sum(bucket["failures"]) if failure_complete else None
        )
        aggregate_rows.append(
            {
                "run_id": run_id,
                "scenario": scenario,
                "timestamp_utc": (
                    started_at + timedelta(seconds=elapsed_second)
                ).isoformat(),
                "elapsed_second": elapsed_second,
                "requests_per_second": request_count,
                "request_count": request_count,
                "mean_response_latency_ms": (
                    sum(bucket["latencies"]) / len(bucket["latencies"])
                    if latency_complete and bucket["latencies"]
                    else None
                ),
                "latency_sample_count": len(bucket["latencies"]),
                "failures": failure_count,
                "error_rate": (
                    failure_count / request_count if failure_complete else None
                ),
                "failure_sample_count": (
                    request_count if failure_complete else request_count - bucket["unknown_failure"]
                ),
                "configured_peak_user_count": users,
                "test_duration_seconds": duration,
            }
        )

    fingerprint = hashlib.sha256()
    fingerprint.update(SCHEMA_VERSION.encode("ascii"))
    fingerprint.update(run["metadata_bytes"])
    fingerprint.update(run["requests_bytes"])
    digest = fingerprint.hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "scenario": scenario,
        "status": run["status"],
        "started_at_utc": started_at.isoformat(),
        "configured_peak_user_count": users,
        "test_duration_seconds": duration,
        "request_record_count": len(request_rows),
        "per_second_record_count": len(aggregate_rows),
        "source_sha256": digest,
        "validation_warnings": run["warnings"],
        "files": {
            "request_records": "request_records.csv",
            "per_second": "per_second.csv",
        },
    }
    return digest, {
        "request_records.csv": _csv_content(REQUEST_OUTPUT_COLUMNS, request_rows),
        "per_second.csv": _csv_content(AGGREGATE_COLUMNS, aggregate_rows),
        "manifest.json": (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }


def _write_dataset(output_root, run_id, digest, contents):
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{run_id}_{digest[:12]}"
    if output_path.exists():
        for filename, content in contents.items():
            path = output_path / filename
            if not path.is_file() or path.read_bytes() != content:
                raise FileExistsError(
                    f"Existing dataset differs from generated output; refusing to overwrite: {output_path}"
                )
        return output_path, True

    with tempfile.TemporaryDirectory(prefix=f".{run_id}.", dir=output_root) as temp:
        staging_path = Path(temp)
        for filename, content in contents.items():
            (staging_path / filename).write_bytes(content)
        try:
            os.rename(staging_path, output_path)
        except FileExistsError:
            for filename, content in contents.items():
                path = output_path / filename
                if not path.is_file() or path.read_bytes() != content:
                    raise FileExistsError(
                        f"Concurrent dataset output differs; refusing to overwrite: {output_path}"
                    )
            return output_path, True
    return output_path, False


def generate_dataset(run_id, raw_dir=RAW_DIR, output_root=OUTPUT_DIR):
    run = _read_run(run_id, Path(raw_dir))
    digest, contents = _dataset_contents(run)
    output_path, reused = _write_dataset(
        Path(output_root), run_id, digest, contents
    )
    return output_path, reused, run["warnings"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate deterministic request-level and per-second datasets from a saved CloudPulse run."
    )
    parser.add_argument("--run-id", required=True, help="Saved Locust run ID")
    args = parser.parse_args(argv)
    try:
        output_path, reused, warnings = generate_dataset(args.run_id)
    except (DatasetValidationError, OSError) as error:
        parser.error(str(error))
    print(f"Dataset directory: {output_path}")
    print("Existing identical output reused." if reused else "Dataset generated.")
    if warnings:
        print("Validation warnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("Validation warnings: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
