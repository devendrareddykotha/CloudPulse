"""Tests for the explicitly labelled single-service MCR workload proxy."""

import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from prediction.src.prepare_workload import (
    MCR_FILES,
    aggregate_service_records,
    prepare_workload_csv,
)


SERVICE_ID = "service-hash-a"
METRIC = "consumerMQ_MCR"


def sample_records():
    return pd.DataFrame(
        [
            (100, 0, SERVICE_ID, "instance-a", METRIC, 1.25),
            (101, 0, SERVICE_ID, "instance-b", METRIC, 2.25),
            (102, 60_000, SERVICE_ID, "instance-a", METRIC, 3.0),
            (103, 60_000, SERVICE_ID, "instance-b", METRIC, 4.0),
            (104, 0, "other-service", "instance-c", METRIC, 100.0),
            (105, 60_000, "other-service", "instance-c", METRIC, 200.0),
        ],
        columns=["field1", "timestamp_ms", "msname", "msinstanceid", "metrics", "value"],
    )


def create_test_archive(path: Path) -> None:
    records = sample_records()
    partitions = [
        records[records["field1"].isin([100, 101])],
        records[records["field1"].isin([102, 103])],
        records[records["field1"].isin([104, 105])],
    ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, partition in zip(MCR_FILES, partitions):
            rows = partition.to_csv(index=False, header=False, lineterminator="\n")
            archive.writestr(name, rows)


class AggregateServiceRecordsTests(unittest.TestCase):
    def test_sums_selected_instances_at_each_timestamp(self) -> None:
        result = aggregate_service_records(
            sample_records(), SERVICE_ID, expected_time_points=2
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result["request_rate_rps"].tolist(), [3.5, 7.0])
        self.assertEqual(result["instance_count"].tolist(), [2, 2])
        self.assertEqual(result["timestamp_ms"].tolist(), [0, 60_000])
        self.assertEqual(result["service_id"].tolist(), [SERVICE_ID, SERVICE_ID])
        self.assertTrue(
            result["timestamp"].equals(
                pd.Series(pd.to_datetime([0, 60_000], unit="ms", utc=True))
            )
        )

    def test_rejects_non_mcr_metric(self) -> None:
        records = sample_records()
        records.loc[0, "metrics"] = "consumerMQ_RT"

        with self.assertRaisesRegex(ValueError, "Expected only"):
            aggregate_service_records(records, SERVICE_ID, expected_time_points=2)

    def test_rejects_duplicate_service_instance_metric_rows(self) -> None:
        records = sample_records()
        duplicate = records.iloc[[0]]
        records = pd.concat([records, duplicate], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "Duplicate service-instance"):
            aggregate_service_records(records, SERVICE_ID, expected_time_points=2)

    def test_rejects_invalid_values(self) -> None:
        invalid_cases = ["not-a-number", float("inf"), float("nan"), -1.0]
        for value in invalid_cases:
            with self.subTest(value=value):
                records = sample_records()
                records["value"] = records["value"].astype(object)
                records.loc[0, "value"] = value
                with self.assertRaises(ValueError):
                    aggregate_service_records(
                        records, SERVICE_ID, expected_time_points=2
                    )

    def test_rejects_incomplete_or_irregular_timestamp_coverage(self) -> None:
        records = sample_records()
        with self.assertRaisesRegex(ValueError, "Expected 3 timestamps"):
            aggregate_service_records(
                records, SERVICE_ID, expected_time_points=3
            )

        records.loc[records["timestamp_ms"] == 60_000, "timestamp_ms"] = 120_000
        with self.assertRaisesRegex(ValueError, "not continuous"):
            aggregate_service_records(
                records, SERVICE_ID, expected_time_points=2
            )


class PrepareWorkloadCsvTests(unittest.TestCase):
    def test_writes_separate_csv_and_preserves_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "synthetic_fixture.zip"
            output_path = root / "processed" / "service_proxy.csv"
            create_test_archive(archive_path)
            original_archive = archive_path.read_bytes()

            workload = prepare_workload_csv(
                archive_path=archive_path,
                service_id=SERVICE_ID,
                output_csv=output_path,
                expected_time_points=2,
                chunk_size=3,
            )
            saved = pd.read_csv(output_path)

            self.assertEqual(len(workload), 2)
            self.assertEqual(saved["request_rate_rps"].tolist(), [3.5, 7.0])
            self.assertEqual(saved["timestamp_ms"].tolist(), [0, 60_000])
            self.assertEqual(archive_path.read_bytes(), original_archive)

    def test_refuses_to_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "synthetic_fixture.zip"
            output_path = root / "service_proxy.csv"
            create_test_archive(archive_path)
            output_path.write_text("preserve me", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                prepare_workload_csv(
                    archive_path=archive_path,
                    service_id=SERVICE_ID,
                    output_csv=output_path,
                    expected_time_points=2,
                )
            self.assertEqual(output_path.read_text(encoding="utf-8"), "preserve me")


if __name__ == "__main__":
    unittest.main()