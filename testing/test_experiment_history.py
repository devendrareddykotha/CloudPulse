import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from service import main


RUN_ID = "locust_steady_20261028T061519_500234Z_test01"
SPIKE_RUN_ID = "locust_spike_20261028T061017_704805Z_test02"
REQUEST_FIELDS = [
    "timestamp_utc",
    "method",
    "name",
    "url",
    "status_code",
    "response_time_ms",
    "response_length",
    "exception",
]


class ExperimentHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.data_dir_patch = patch.object(main, "DATA_DIR", self.data_dir)
        self.data_dir_patch.start()
        self.addCleanup(self.data_dir_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)
        self.started_at = datetime(2026, 10, 28, 6, 15, 19, tzinfo=timezone.utc)

    def write_metadata(
        self,
        run_id=RUN_ID,
        status="completed",
        scenario="steady",
        duration_seconds=30,
        started_at=None,
    ):
        started_at = started_at or self.started_at
        metadata = {
            "configuration": {
                "scenario": scenario,
                "peak_users": 10,
                "duration_seconds": duration_seconds,
            },
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": (started_at + timedelta(seconds=3)).isoformat(),
            "exit_code": 0 if status == "completed" else None,
            "status": status,
        }
        path = self.data_dir / f"{run_id}_metadata.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        return path

    def write_requests(self, run_id=RUN_ID, rows=None, started_at=None):
        started_at = started_at or self.started_at
        rows = rows or ((0.1, 10), (0.8, 30), (2.2, 50))
        path = self.data_dir / f"{run_id}_requests.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=REQUEST_FIELDS)
            writer.writeheader()
            for offset, latency in rows:
                writer.writerow(
                    {
                        "timestamp_utc": (
                            started_at + timedelta(seconds=offset)
                        ).isoformat(),
                        "method": "GET",
                        "name": "/workload",
                        "url": "http://127.0.0.1:8000/workload",
                        "status_code": "200",
                        "response_time_ms": str(latency),
                        "response_length": "10",
                        "exception": "",
                    }
                )
        return path

    def test_history_lists_saved_run_and_real_summary(self):
        self.write_metadata()
        self.write_requests()

        payload = main.experiment_history()

        self.assertEqual(len(payload["experiments"]), 1)
        summary = payload["experiments"][0]
        self.assertEqual(summary["run_id"], RUN_ID)
        self.assertEqual(summary["scenario"], "steady")
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["request_count"], 3)
        self.assertEqual(summary["failed_requests"], 0)
        self.assertEqual(summary["average_latency_ms"], 30)
        self.assertEqual(summary["median_latency_ms"], 30)
        self.assertAlmostEqual(summary["p95_latency_ms"], 48)

    def test_selected_saved_run_returns_csv_backed_time_series(self):
        self.write_metadata()
        self.write_requests()

        payload = main.saved_experiment_data(RUN_ID)

        self.assertTrue(payload["historical"])
        self.assertEqual(payload["experiment"]["status"], "completed")
        self.assertEqual(payload["metrics"]["total_requests"], 3)
        self.assertEqual(
            [point["value"] for point in payload["request_rate"]], [2, 0, 1]
        )
        measured_latency = [
            point["value"] for point in payload["latency"] if point["value"] is not None
        ]
        self.assertEqual(measured_latency, [20, 50])

    def test_missing_request_csv_is_not_listed_or_selectable(self):
        self.write_metadata()

        self.assertEqual(main.experiment_history()["experiments"], [])
        with self.assertRaises(HTTPException) as error:
            main.saved_experiment_data(RUN_ID)
        self.assertEqual(error.exception.status_code, 404)

    def test_malformed_metadata_is_skipped(self):
        metadata_path = self.data_dir / f"{RUN_ID}_metadata.json"
        metadata_path.write_text("{invalid json", encoding="utf-8")
        self.write_requests()

        self.assertEqual(main.experiment_history()["experiments"], [])
        with self.assertRaises(HTTPException) as error:
            main.saved_experiment_data(RUN_ID)
        self.assertEqual(error.exception.status_code, 404)

    def test_run_id_path_traversal_is_rejected(self):
        with self.assertRaises(HTTPException) as error:
            main.saved_experiment_data("..\\request_logs")
        self.assertEqual(error.exception.status_code, 404)

    def test_comparison_uses_csv_metrics_and_relative_time_series(self):
        self.write_metadata()
        self.write_requests()
        self.write_metadata(
            SPIKE_RUN_ID,
            scenario="spike",
            duration_seconds=60,
            started_at=self.started_at + timedelta(minutes=1),
        )
        self.write_requests(
            SPIKE_RUN_ID,
            rows=((0.1, 100), (0.8, 200), (2.2, 300)),
            started_at=self.started_at + timedelta(minutes=1),
        )

        payload = main.compare_saved_experiments(RUN_ID, SPIKE_RUN_ID)

        steady, spike = payload["runs"]
        self.assertEqual(steady["summary"]["request_count"], 3)
        self.assertEqual(steady["summary"]["average_latency_ms"], 30)
        self.assertEqual(steady["summary"]["median_latency_ms"], 30)
        self.assertAlmostEqual(steady["summary"]["p95_latency_ms"], 48)
        self.assertEqual(steady["summary"]["average_throughput_rps"], 0.1)
        self.assertEqual(
            [point["value"] for point in steady["request_rate"]], [2, 0, 1]
        )
        self.assertEqual(
            [point["value"] for point in steady["latency"]], [20, None, 50]
        )
        self.assertEqual(spike["summary"]["scenario"], "spike")
        self.assertEqual(spike["summary"]["average_latency_ms"], 200)
        self.assertEqual(spike["summary"]["median_latency_ms"], 200)
        self.assertAlmostEqual(spike["summary"]["p95_latency_ms"], 290)
        self.assertEqual(spike["summary"]["average_throughput_rps"], 0.05)
        self.assertEqual(
            [point["value"] for point in spike["request_rate"]], [2, 0, 1]
        )
        self.assertEqual(
            [point["value"] for point in spike["latency"]], [150, None, 300]
        )

    def test_comparison_bins_real_csv_records_from_each_run_start(self):
        spike_started_at = self.started_at + timedelta(hours=5)
        self.write_metadata()
        self.write_requests(rows=((-0.2, 5), (0.2, 10), (2.2, 50)))
        self.write_metadata(
            SPIKE_RUN_ID,
            scenario="spike",
            duration_seconds=60,
            started_at=spike_started_at,
        )
        self.write_requests(
            SPIKE_RUN_ID,
            rows=((0.2, 100), (2.2, 300)),
            started_at=spike_started_at,
        )

        payload = main.compare_saved_experiments(RUN_ID, SPIKE_RUN_ID)
        steady, spike = payload["runs"]

        self.assertEqual(steady["summary"]["request_count"], 3)
        self.assertEqual(
            [point["time_seconds"] for point in steady["request_rate"]], [0, 1, 2]
        )
        self.assertEqual(
            [point["value"] for point in steady["request_rate"]], [1, 0, 1]
        )
        self.assertEqual(
            [point["value"] for point in steady["latency"]], [10, None, 50]
        )
        self.assertEqual(
            [point["time_seconds"] for point in spike["request_rate"]], [0, 1, 2]
        )
        self.assertEqual(
            [point["value"] for point in spike["request_rate"]], [1, 0, 1]
        )
        self.assertEqual(
            [point["value"] for point in spike["latency"]], [100, None, 300]
        )

    def test_comparison_rejects_invalid_run_id(self):
        with self.assertRaises(HTTPException) as error:
            main.compare_saved_experiments(RUN_ID, "..\\request_logs")
        self.assertEqual(error.exception.status_code, 404)

    def test_comparison_rejects_run_with_missing_request_csv(self):
        self.write_metadata()
        self.write_metadata(
            SPIKE_RUN_ID,
            scenario="spike",
            duration_seconds=60,
        )
        self.write_requests()

        with self.assertRaises(HTTPException) as error:
            main.compare_saved_experiments(RUN_ID, SPIKE_RUN_ID)
        self.assertEqual(error.exception.status_code, 404)

    def test_comparison_rejects_malformed_request_csv(self):
        self.write_metadata()
        self.write_requests()
        self.write_metadata(
            SPIKE_RUN_ID,
            scenario="spike",
            duration_seconds=60,
        )
        (self.data_dir / f"{SPIKE_RUN_ID}_requests.csv").write_text(
            "not,a,request,csv\n1,2,3,4\n",
            encoding="utf-8",
        )

        with self.assertRaises(HTTPException) as error:
            main.compare_saved_experiments(RUN_ID, SPIKE_RUN_ID)
        self.assertEqual(error.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
