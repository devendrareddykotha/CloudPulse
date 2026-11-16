import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from testing import run_load_test


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return b"measured response"


def profile_args(scenario, **overrides):
    values = {
        "scenario": scenario,
        "users": 6,
        "low_users": 2,
        "warmup_duration": 2,
        "peak_duration": 4,
        "recovery_duration": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class StandardLibraryLoadTests(unittest.TestCase):
    def test_profiles_produce_the_expected_user_targets(self):
        self.assertEqual(
            run_load_test.target_user_count(profile_args("steady"), 2, 8), 6
        )
        spike = profile_args("spike")
        self.assertEqual(run_load_test.target_user_count(spike, 1, 8), 2)
        self.assertEqual(run_load_test.target_user_count(spike, 3, 8), 6)
        self.assertEqual(run_load_test.target_user_count(spike, 7, 8), 2)
        self.assertEqual(
            run_load_test.target_user_count(profile_args("ramp"), 4, 8), 3
        )
        periodic = profile_args("periodic")
        self.assertEqual(run_load_test.target_user_count(periodic, 0, 8), 3)
        self.assertEqual(run_load_test.target_user_count(periodic, 1, 8), 6)
        self.assertEqual(run_load_test.target_user_count(periodic, 2, 8), 3)

    def test_local_run_writes_measured_request_records(self):
        args = profile_args(
            "steady",
            users=2,
            spawn_rate=100,
            host="http://127.0.0.1:8000",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            request_path = Path(temp_dir) / "run_requests.csv"
            stop_file = Path(temp_dir) / "run_stop.request"
            with patch.object(run_load_test, "urlopen", return_value=FakeResponse()):
                stopped = run_load_test.run_standard_library_load(
                    args, 0.25, request_path, stop_file
                )

            with request_path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                rows = list(reader)

        self.assertFalse(stopped)
        self.assertEqual(tuple(reader.fieldnames), run_load_test.REQUEST_FIELDS)
        self.assertGreaterEqual(len(rows), 1)
        for row in rows:
            self.assertEqual(row["status_code"], "200")
            self.assertEqual(row["response_length"], str(len(b"measured response")))
            self.assertGreaterEqual(float(row["response_time_ms"]), 0)
            self.assertIn(row["name"], {"/workload", "/health"})

    def test_standard_library_runner_rejects_non_loopback_hosts(self):
        args = profile_args("steady", users=1, spawn_rate=1, host="http://example.com")
        with tempfile.TemporaryDirectory() as temp_dir:
            request_path = Path(temp_dir) / "run_requests.csv"
            with self.assertRaisesRegex(ValueError, "loopback"):
                run_load_test.run_standard_library_load(
                    args, 0.25, request_path, Path(temp_dir) / "stop.request"
                )
            self.assertFalse(request_path.exists())


if __name__ == "__main__":
    unittest.main()
