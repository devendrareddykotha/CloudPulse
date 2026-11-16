"""Unit tests for CloudPulse workload validation using synthetic CSV inputs."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from prediction.src.preprocessing import (
    REQUIRED_COLUMNS,
    load_workload_csv,
    validate_workload_quality,
)


class WorkloadPreprocessingTests(unittest.TestCase):
    def create_csv(self, directory: str, name: str, content: str) -> Path:
        path = Path(directory) / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_five_valid_rows_are_returned_in_chronological_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_content = (
                "timestamp,request_rate_rps\n"
                "2026-10-12T10:00:00Z,20\n"
                "2026-10-12T10:00:10Z,25\n"
                "2026-10-12T10:00:20Z,23\n"
                "2026-10-12T10:00:30Z,30\n"
                "2026-10-12T10:00:40Z,35\n"
            )
            path = self.create_csv(
                directory,
                "valid.csv",
                csv_content,
            )
            result = load_workload_csv(path)
            original_csv = path.read_text(encoding="utf-8")

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(column in result.columns for column in REQUIRED_COLUMNS))
        self.assertEqual(result["request_rate_rps"].tolist(), [20, 25, 23, 30, 35])
        self.assertEqual(result.attrs["validation_report"]["total_rows"], 5)
        self.assertEqual(result.attrs["validation_report"]["out_of_order_timestamps"], 0)
        self.assertEqual(result.attrs["validation_report"]["timezone"], None)
        self.assertEqual(original_csv, csv_content)

    def test_missing_file_raises_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            with self.assertRaises(FileNotFoundError):
                load_workload_csv(path)

    def test_empty_file_raises_clear_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_csv(directory, "empty.csv", "")
            with self.assertRaisesRegex(ValueError, "CSV is empty"):
                load_workload_csv(path)

    def test_header_without_data_rows_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_csv(
                directory, "header_only.csv", "timestamp,request_rate_rps\n"
            )
            with self.assertRaisesRegex(ValueError, "no workload rows"):
                load_workload_csv(path)

    def test_missing_required_column_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_csv(
                directory,
                "missing_column.csv",
                "timestamp,other_metric\n2026-10-12T10:00:00Z,50\n",
            )
            with self.assertRaisesRegex(ValueError, "request_rate_rps"):
                load_workload_csv(path)

    def test_invalid_timestamp_is_reported_without_dropping_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_csv(
                directory,
                "invalid_timestamp.csv",
                "timestamp,request_rate_rps\nnot-a-time,50\n",
            )
            result = load_workload_csv(path)

        report = result.attrs["validation_report"]
        self.assertEqual(len(result), 1)
        self.assertEqual(report["invalid_timestamps"], 1)
        self.assertEqual(report["problem_rows"]["invalid_timestamps"], [2])

    def test_missing_and_non_numeric_rates_are_distinguished(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_csv(
                directory,
                "invalid_rates.csv",
                "timestamp,request_rate_rps\n"
                "2026-10-12T10:00:00Z,\n"
                "2026-10-12T10:01:00Z,not-a-number\n",
            )
            result = load_workload_csv(path)

        report = result.attrs["validation_report"]
        self.assertEqual(len(result), 2)
        self.assertEqual(report["missing_request_rates"], 1)
        self.assertEqual(report["non_numeric_request_rates"], 1)

    def test_negative_rates_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_csv(
                directory,
                "negative_rate.csv",
                "timestamp,request_rate_rps\n2026-10-12T10:00:00Z,-1\n",
            )
            result = load_workload_csv(path)

        self.assertEqual(
            result.attrs["validation_report"]["negative_request_rates"], 1
        )

    def test_duplicate_timestamps_are_reported_without_dropping_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_csv(
                directory,
                "duplicates.csv",
                "timestamp,request_rate_rps\n"
                "2026-10-12T10:00:00Z,10\n"
                "2026-10-12T10:00:00Z,12\n",
            )
            result = load_workload_csv(path)

        self.assertEqual(len(result), 2)
        self.assertEqual(
            result.attrs["validation_report"]["duplicate_timestamps"], 1
        )

    def test_sampling_gaps_are_only_checked_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_csv(
                directory,
                "sampling_gap.csv",
                "timestamp,request_rate_rps\n"
                "2026-10-12T10:00:00Z,10\n"
                "2026-10-12T10:02:00Z,12\n",
            )
            without_interval = load_workload_csv(path)
            with_interval = load_workload_csv(
                path, expected_sampling_interval="1min"
            )

        self.assertEqual(
            without_interval.attrs["validation_report"]["unexpected_sampling_gaps"],
            0,
        )
        self.assertEqual(
            with_interval.attrs["validation_report"]["unexpected_sampling_gaps"], 1
        )

    def test_timezone_can_be_configured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_csv(
                directory,
                "local_time.csv",
                "timestamp,request_rate_rps\n2026-10-12 10:00:00,10\n",
            )
            result = load_workload_csv(path, timezone="America/New_York")

        self.assertEqual(str(result.iloc[0]["timestamp"].tz), "America/New_York")
        self.assertEqual(
            result.attrs["validation_report"]["timezone"], "America/New_York"
        )

    def test_quality_report_finds_all_anomalies(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": [
                    "2026-10-12T10:01:00Z",
                    "2026-10-12T10:00:00Z",
                    "2026-10-12T10:00:00Z",
                ],
                "request_rate_rps": ["-2", "bad", None],
            }
        )

        report = validate_workload_quality(frame)

        self.assertEqual(report["negative_request_rates"], 1)
        self.assertEqual(report["non_numeric_request_rates"], 1)
        self.assertEqual(report["missing_request_rates"], 1)
        self.assertEqual(report["duplicate_timestamps"], 1)
        self.assertEqual(report["out_of_order_timestamps"], 1)


if __name__ == "__main__":
    unittest.main()
