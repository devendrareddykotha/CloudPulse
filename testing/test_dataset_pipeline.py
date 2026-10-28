import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from evaluation.generate_dataset import (
    REQUEST_COLUMNS,
    DatasetValidationError,
    generate_dataset,
)


RUN_ID = "locust_steady_20261028T061519_500234Z_dataset01"
SPIKE_RUN_ID = "locust_spike_20261028T061017_704805Z_dataset02"
STARTED_AT = datetime(2026, 10, 28, 6, 15, 19, tzinfo=timezone.utc)


class DatasetPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.raw_dir = self.root / "raw"
        self.output_dir = self.root / "processed" / "datasets"
        self.raw_dir.mkdir()
        self.addCleanup(self.temp_dir.cleanup)

    def write_metadata(
        self,
        run_id=RUN_ID,
        scenario="steady",
        started_at=STARTED_AT,
        duration=30,
        users=10,
        status="completed",
    ):
        metadata = {
            "configuration": {
                "scenario": scenario,
                "peak_users": users,
                "duration_seconds": duration,
            },
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": (started_at + timedelta(seconds=duration)).isoformat(),
            "exit_code": 0 if status == "completed" else None,
            "status": status,
        }
        (self.raw_dir / f"{run_id}_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    def write_requests(self, run_id=RUN_ID, rows=None, started_at=STARTED_AT):
        rows = rows or (
            (0.1, "10", "200", ""),
            (0.8, "30", "500", ""),
            (2.2, "50", "", ""),
        )
        path = self.raw_dir / f"{run_id}_requests.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=REQUEST_COLUMNS)
            writer.writeheader()
            for offset, latency, status_code, exception in rows:
                writer.writerow(
                    {
                        "timestamp_utc": (
                            started_at + timedelta(seconds=offset)
                        ).isoformat(),
                        "method": "GET",
                        "name": "/workload",
                        "url": "http://127.0.0.1:8000/workload",
                        "status_code": status_code,
                        "response_time_ms": latency,
                        "response_length": "98",
                        "exception": exception,
                    }
                )
        return path

    def read_csv(self, path):
        with path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def test_aggregates_real_records_and_preserves_missing_values(self):
        self.write_metadata()
        self.write_requests(
            rows=(
                (0.1, "10", "200", ""),
                (0.8, "", "500", ""),
                (2.2, "50", "", ""),
            )
        )

        output_path, reused, warnings = generate_dataset(
            RUN_ID, self.raw_dir, self.output_dir
        )

        requests = self.read_csv(output_path / "request_records.csv")
        aggregates = self.read_csv(output_path / "per_second.csv")
        self.assertFalse(reused)
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[1]["response_time_ms"], "")
        self.assertTrue(any("response_time_ms is missing" in item for item in warnings))
        self.assertEqual(
            [int(row["elapsed_second"]) for row in aggregates], [0, 2]
        )
        self.assertEqual(int(aggregates[0]["request_count"]), 2)
        self.assertEqual(float(aggregates[0]["requests_per_second"]), 2)
        self.assertEqual(aggregates[0]["mean_response_latency_ms"], "")
        self.assertEqual(int(aggregates[0]["latency_sample_count"]), 1)
        self.assertEqual(int(aggregates[0]["failures"]), 1)
        self.assertEqual(float(aggregates[0]["error_rate"]), 0.5)
        self.assertEqual(aggregates[1]["failures"], "")
        self.assertEqual(aggregates[1]["error_rate"], "")

    def test_runs_align_to_their_own_start_time(self):
        later_start = STARTED_AT + timedelta(hours=4)
        self.write_metadata()
        self.write_requests(rows=((0.2, "11", "200", ""), (2.2, "33", "200", "")))
        self.write_metadata(
            SPIKE_RUN_ID,
            scenario="spike",
            started_at=later_start,
            duration=60,
        )
        self.write_requests(
            SPIKE_RUN_ID,
            rows=((0.2, "101", "200", ""), (2.2, "303", "200", "")),
            started_at=later_start,
        )

        steady_path, _, _ = generate_dataset(
            RUN_ID, self.raw_dir, self.output_dir
        )
        spike_path, _, _ = generate_dataset(
            SPIKE_RUN_ID, self.raw_dir, self.output_dir
        )
        steady = self.read_csv(steady_path / "per_second.csv")
        spike = self.read_csv(spike_path / "per_second.csv")
        self.assertEqual(
            [row["elapsed_second"] for row in steady], ["0", "2"]
        )
        self.assertEqual(
            [row["elapsed_second"] for row in spike], ["0", "2"]
        )
        self.assertEqual(
            [row["timestamp_utc"] for row in steady],
            [
                STARTED_AT.isoformat(),
                (STARTED_AT + timedelta(seconds=2)).isoformat(),
            ],
        )
        self.assertEqual(
            [row["timestamp_utc"] for row in spike],
            [
                later_start.isoformat(),
                (later_start + timedelta(seconds=2)).isoformat(),
            ],
        )

    def test_per_second_mean_latency_uses_only_recorded_values(self):
        self.write_metadata()
        self.write_requests(
            rows=(
                (0.1, "10", "200", ""),
                (0.8, "30", "200", ""),
                (2.2, "50", "200", ""),
            )
        )

        output_path, _, _ = generate_dataset(
            RUN_ID, self.raw_dir, self.output_dir
        )
        aggregates = self.read_csv(output_path / "per_second.csv")

        self.assertEqual(
            [float(row["mean_response_latency_ms"]) for row in aggregates],
            [20.0, 50.0],
        )

    def test_generation_is_repeatable_and_does_not_replace_outputs(self):
        self.write_metadata()
        self.write_requests()
        output_path, reused, _ = generate_dataset(
            RUN_ID, self.raw_dir, self.output_dir
        )
        original_files = {
            path.name: path.read_bytes() for path in output_path.iterdir()
        }

        same_path, reused_again, _ = generate_dataset(
            RUN_ID, self.raw_dir, self.output_dir
        )

        self.assertEqual(same_path, output_path)
        self.assertFalse(reused)
        self.assertTrue(reused_again)
        self.assertEqual(
            original_files,
            {path.name: path.read_bytes() for path in output_path.iterdir()},
        )
        (output_path / "per_second.csv").write_text("preserve me\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            generate_dataset(RUN_ID, self.raw_dir, self.output_dir)
        self.assertEqual(
            (output_path / "per_second.csv").read_text(encoding="utf-8"),
            "preserve me\n",
        )

    def test_empty_request_file_is_rejected(self):
        self.write_metadata()
        (self.raw_dir / f"{RUN_ID}_requests.csv").write_bytes(b"")

        with self.assertRaisesRegex(DatasetValidationError, "is empty"):
            generate_dataset(RUN_ID, self.raw_dir, self.output_dir)

    def test_missing_columns_are_reported(self):
        self.write_metadata()
        (self.raw_dir / f"{RUN_ID}_requests.csv").write_text(
            "timestamp_utc,response_time_ms\n"
            "2026-10-28T06:15:19+00:00,10\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(DatasetValidationError, "missing required columns"):
            generate_dataset(RUN_ID, self.raw_dir, self.output_dir)

    def test_malformed_timestamp_is_reported(self):
        self.write_metadata()
        with (self.raw_dir / f"{RUN_ID}_requests.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=REQUEST_COLUMNS)
            writer.writeheader()
            writer.writerow(
                {
                    "timestamp_utc": "not-a-timestamp",
                    "method": "GET",
                    "name": "/workload",
                    "url": "http://127.0.0.1:8000/workload",
                    "status_code": "200",
                    "response_time_ms": "10",
                    "response_length": "98",
                    "exception": "",
                }
            )

        with self.assertRaisesRegex(DatasetValidationError, "Invalid timestamp_utc"):
            generate_dataset(RUN_ID, self.raw_dir, self.output_dir)

    def test_duplicate_request_records_are_rejected(self):
        self.write_metadata()
        duplicate = (0.1, "10", "200", "")
        self.write_requests(rows=(duplicate, duplicate))

        with self.assertRaisesRegex(DatasetValidationError, "duplicate request record"):
            generate_dataset(RUN_ID, self.raw_dir, self.output_dir)

    def test_all_supported_scenario_names_are_accepted(self):
        for scenario, run_id in (
            ("steady", RUN_ID),
            ("spike", SPIKE_RUN_ID),
            ("ramp", "locust_ramp_20261028T061519_500234Z_dataset03"),
            ("periodic", "locust_periodic_20261028T061519_500234Z_dataset04"),
        ):
            self.write_metadata(run_id=run_id, scenario=scenario)
            self.write_requests(run_id=run_id)
            output_path, _, _ = generate_dataset(
                run_id, self.raw_dir, self.output_dir
            )
            manifest = json.loads(
                (output_path / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["scenario"], scenario)


if __name__ == "__main__":
    unittest.main()
