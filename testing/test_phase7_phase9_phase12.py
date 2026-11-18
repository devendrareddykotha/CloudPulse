import asyncio
import csv
from collections import Counter, deque
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from threading import Lock
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from fastapi import Response

from evaluation import (
    collect_metrics,
    compare_results,
    measure_distribution,
    plot_results,
    validate_data,
)
from prediction.src import live_loop, live_metrics_collector
from service import main
from testing import run_periodic, run_ramp, run_spike, run_steady


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


class Phase7MetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.processed = self.root / "processed"
        self.raw.mkdir()
        self.run_id = "locust_steady_20261117T120000_000000Z_test"
        self.started = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
        self.source = self.raw / f"{self.run_id}_requests.csv"
        with self.source.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=REQUEST_FIELDS)
            writer.writeheader()
            for offset, latency, status in (
                (1, 10, 200),
                (8, 30, 500),
                (11, 50, 200),
            ):
                writer.writerow(
                    {
                        "timestamp_utc": (
                            self.started + timedelta(seconds=offset)
                        ).isoformat(),
                        "method": "GET",
                        "name": "/workload",
                        "url": "http://127.0.0.1:8000/workload",
                        "status_code": status,
                        "response_time_ms": latency,
                        "response_length": 100,
                        "exception": "",
                    }
                )
        self.addCleanup(self.temp.cleanup)

    def test_run_export_has_exact_contract_and_measured_values(self):
        output_path = collect_metrics.collect_run(
            self.run_id, self.raw, self.processed
        )
        with output_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
        self.assertEqual(tuple(reader.fieldnames), collect_metrics.CONTRACT_COLUMNS)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["request_count"], "2")
        self.assertEqual(rows[0]["request_rate_rps"], "0.2")
        self.assertEqual(rows[0]["average_latency_ms"], "20.0")
        self.assertEqual(rows[0]["p95_latency_ms"], "29.0")
        self.assertEqual(rows[0]["error_count"], "1")
        self.assertEqual(rows[1]["request_count"], "1")
        for field in ("cpu_utilization_pct", "memory_usage_bytes", "replica_count"):
            self.assertEqual(rows[0][field], "")
            self.assertEqual(rows[1][field], "")

    def test_output_is_not_overwritten(self):
        output_path = collect_metrics.collect_run(
            self.run_id, self.raw, self.processed
        )
        output_path.write_text("preserve\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            collect_metrics.collect_run(self.run_id, self.raw, self.processed)
        self.assertEqual(output_path.read_text(encoding="utf-8"), "preserve\n")

    def test_kubernetes_sample_uses_actual_values_and_cpu_limits(self):
        deployment = {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {"resources": {"limits": {"cpu": "500m"}}}
                        ]
                    }
                }
            },
            "status": {"availableReplicas": 2},
        }
        completed_deployment = type(
            "Completed", (), {"stdout": json.dumps(deployment)}
        )()
        completed_top = type(
            "Completed", (), {"stdout": "pod-a 250m 64Mi\npod-b 250m 128Mi\n"}
        )()
        with patch.object(
            collect_metrics.subprocess,
            "run",
            side_effect=(completed_deployment, completed_top),
        ) as run:
            metrics = collect_metrics.sample_kubernetes_metrics()
        self.assertEqual(metrics["replica_count"], 2)
        self.assertEqual(metrics["cpu_utilization_pct"], 50)
        self.assertEqual(metrics["memory_usage_bytes"], 192 * 1024**2)
        self.assertIn("get", run.call_args_list[0].args[0])


class Phase9DistributionAndPodTests(unittest.TestCase):
    def test_workload_includes_environment_pod_name_and_header(self):
        response = Response()
        with patch.dict("os.environ", {"HOSTNAME": "cloudpulse-pod-123"}):
            payload = main.workload(response)
        self.assertEqual(payload["pod_name"], "cloudpulse-pod-123")
        self.assertEqual(payload["message"], "CloudPulse workload endpoint is ready")
        self.assertIn("timestamp", payload)
        self.assertEqual(response.headers["X-Pod-Name"], "cloudpulse-pod-123")

    def test_missing_hostname_defaults_to_local(self):
        response = Response()
        with patch.dict("os.environ", {}, clear=True):
            payload = main.workload(response)
        self.assertEqual(payload["pod_name"], "local")
        self.assertEqual(response.headers["X-Pod-Name"], "local")

    def test_distribution_identity_ratio_and_loopback_restriction(self):
        pod = measure_distribution.parse_response_identity(
            [("X-Pod-Name", "pod-a")],
            b'{"pod_name":"pod-b"}',
        )
        self.assertEqual(pod, "pod-a")
        self.assertEqual(
            measure_distribution.load_imbalance_ratio({"pod-a": 4, "pod-b": 2}),
            4 / 3,
        )
        with self.assertRaises(measure_distribution.DistributionError):
            measure_distribution.validate_workload_url(
                "https://example.com/workload"
            )

    def test_probe_uses_connection_close_and_reuses_pooled_connection(self):
        created = []

        class FakeResponse:
            status = 200

            def read(self):
                return b'{"pod_name":"pod-a"}'

            def getheaders(self):
                return []

        class FakeConnection:
            def __init__(self, host, port, timeout):
                self.request_headers = []
                created.append(self)

            def request(self, method, path, headers):
                self.request_headers.append(headers)

            def getresponse(self):
                return FakeResponse()

            def close(self):
                pass

        with patch.object(
            measure_distribution.http.client,
            "HTTPConnection",
            FakeConnection,
        ):
            results = measure_distribution.probe_connection_mode(
                "http://127.0.0.1:8000/workload",
                request_count=3,
                workers=1,
                mode="both",
            )
        self.assertEqual(len(created), 4)
        self.assertEqual(results[0]["pod_counts"], {"pod-a": 3})
        self.assertEqual(results[1]["pod_counts"], {"pod-a": 3})
        self.assertTrue(
            all(
                connection.request_headers[0]["Connection"] == "close"
                for connection in created[:3]
            )
        )
        self.assertEqual(
            [header["Connection"] for header in created[3].request_headers],
            ["keep-alive"] * 3,
        )

    def test_distribution_report_writes_json_and_csv(self):
        result = {
            "mode": "fresh",
            "pod_counts": {"pod-a": 2},
            "total_responses": 2,
            "failed_requests": 0,
            "ideal_equal_share": 2,
            "load_imbalance_ratio": 1.0,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, csv_path = measure_distribution.write_report(
                [result], temp_dir, stamp="test"
            )
            self.assertEqual(json.loads(json_path.read_text())["results"][0]["total_responses"], 2)
            with csv_path.open(newline="", encoding="utf-8") as source:
                row = next(csv.DictReader(source))
        self.assertEqual(row["pod_name"], "pod-a")
        self.assertEqual(row["request_count"], "2")


class ServiceWorkloadCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.log_path = Path(self.temp.name) / "request_logs.csv"
        self.log_path.touch()
        self.addCleanup(self.temp.cleanup)
        self.patches = [
            patch.object(main, "CSV_FILE", self.log_path),
            patch.object(main, "request_timestamps", deque()),
            patch.object(main, "request_rate_lock", Lock()),
            patch.object(main, "total_requests", 0),
            patch.object(main, "endpoint_counts", Counter()),
            patch.object(main, "request_history", []),
            patch.object(
                main,
                "_get_kubernetes_health",
                return_value={
                    "replicas": None,
                    "cpu_percent": None,
                    "memory_percent": None,
                    "available": False,
                },
            ),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def request(self, target):
        parsed = urlsplit(target)
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": parsed.path,
            "raw_path": parsed.path.encode("ascii"),
            "query_string": parsed.query.encode("ascii"),
            "root_path": "",
            "headers": [(b"host", b"testserver")],
            "client": ("testclient", 12345),
            "server": ("testserver", 80),
            "state": {},
        }
        messages = []
        request_sent = False

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": b"",
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        asyncio.run(main.app(scope, receive, send))
        status = next(
            message["status"]
            for message in messages
            if message["type"] == "http.response.start"
        )
        headers = dict(
            next(
                message["headers"]
                for message in messages
                if message["type"] == "http.response.start"
            )
        )
        body = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        )
        return status, headers, json.loads(body)

    def test_work_endpoint_supports_bounded_configurable_work(self):
        status, headers, payload = self.request("/work?n=500")
        self.assertEqual(status, 200)
        self.assertEqual(payload["work_units"], 500)
        self.assertEqual(payload["result"], sum(i * i for i in range(500)))
        self.assertIn("timestamp", payload)
        self.assertEqual(
            headers[b"x-pod-name"].decode("utf-8"), payload["pod_name"]
        )

    def test_workload_endpoint_keeps_existing_fields_and_adds_work(self):
        status, headers, payload = self.request("/workload?n=500")
        self.assertEqual(status, 200)
        self.assertEqual(payload["work_units"], 500)
        self.assertEqual(payload["result"], sum(i * i for i in range(500)))
        self.assertEqual(payload["message"], "CloudPulse workload endpoint is ready")
        self.assertIn("timestamp", payload)
        self.assertEqual(
            headers[b"x-pod-name"].decode("utf-8"), payload["pod_name"]
        )

    def test_request_rate_excludes_health_and_api_dashboard_polls(self):
        status, _, first_rate = self.request("/metrics/request-rate")
        self.assertEqual(status, 200)
        self.assertEqual(
            {
                key: first_rate[key]
                for key in (
                    "requests_last_60_seconds",
                    "request_rate_rps",
                    "window_seconds",
                )
            },
            {
                "requests_last_60_seconds": 0,
                "request_rate_rps": 0,
                "window_seconds": 60,
            },
        )
        self.request("/work?n=1")
        self.request("/workload?n=1")
        for excluded_path in (
            "/health",
            "/api/dashboard",
            "/metrics/request-rate",
        ):
            status, _, payload = self.request(excluded_path)
            self.assertEqual(status, 200)
            if excluded_path == "/metrics/request-rate":
                self.assertEqual(payload["requests_last_60_seconds"], 2)


class LiveMetricsCollectorUrlTests(unittest.TestCase):
    def test_collect_metrics_accepts_an_explicit_url(self):
        payload = {
            "requests_last_60_seconds": 12,
            "request_rate_rps": 0.2,
            "window_seconds": 60,
            "timestamp": "2026-11-17T00:00:00+00:00",
        }

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        with patch.object(
            live_metrics_collector, "urlopen", return_value=FakeResponse()
        ) as open_url:
            result = live_metrics_collector.collect_metrics(
                "http://127.0.0.1:8088/metrics/request-rate"
            )
        self.assertEqual(result["requests_last_60_seconds"], 12)
        self.assertEqual(
            open_url.call_args.args[0].full_url,
            "http://127.0.0.1:8088/metrics/request-rate",
        )

    def test_collector_cli_uses_environment_url(self):
        metrics = {
            "metric_timestamp_utc": "2026-11-17T00:00:00+00:00",
            "requests_last_60_seconds": 3,
            "request_rate_rps": 0.05,
            "window_seconds": 60,
        }
        with tempfile.TemporaryDirectory() as output_dir:
            with (
                patch.object(live_metrics_collector, "OUTPUT_DIR", Path(output_dir)),
                patch.object(live_metrics_collector, "collect_metrics", return_value=metrics) as collect,
                patch.dict(
                    "os.environ",
                    {"CLOUDPULSE_METRICS_URL": "http://127.0.0.1:8000/metrics/request-rate"},
                ),
                patch.object(sys, "argv", ["live_metrics_collector.py", "--samples", "1"]),
            ):
                live_metrics_collector.main()
            output_file = next(Path(output_dir).glob("*.csv"))
            with output_file.open(newline="", encoding="utf-8") as source:
                row = next(csv.DictReader(source))
        self.assertEqual(
            collect.call_args.args[0],
            "http://127.0.0.1:8000/metrics/request-rate",
        )
        self.assertEqual(row["source_url"], collect.call_args.args[0])


class Phase12ComparisonTests(unittest.TestCase):
    def test_compare_run_summaries_and_chart_use_saved_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw = Path(temp_dir) / "raw"
            processed = Path(temp_dir) / "processed"
            graphs = Path(temp_dir) / "graphs"
            raw.mkdir()
            start = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
            ids = (
                "locust_spike_k8s_hpa_20261116174859",
                "locust_spike_k8s_predictive_20261116175826",
            )
            for index, run_id in enumerate(ids):
                with (raw / f"{run_id}_requests.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as output:
                    writer = csv.DictWriter(output, fieldnames=REQUEST_FIELDS)
                    writer.writeheader()
                    for offset, latency in ((1, 10 + index * 10), (8, 30 + index * 10), (21, 50 + index * 10)):
                        writer.writerow(
                            {
                                "timestamp_utc": (
                                    start + timedelta(seconds=offset)
                                ).isoformat(),
                                "method": "GET",
                                "name": "/workload",
                                "url": "http://127.0.0.1:8000/workload",
                                "status_code": 200,
                                "response_time_ms": latency,
                                "response_length": 100,
                                "exception": "",
                            }
                        )
            results = compare_results.compare_runs(ids, raw)
            self.assertEqual(results[0]["summary"]["request_count"], 3)
            self.assertEqual(results[0]["request_rate"], [0.2, 0, 0.1])
            summary_path, graph_path = compare_results.write_comparison(
                results, processed, graphs, stamp="test"
            )
            self.assertTrue(summary_path.is_file())
            self.assertIn("<svg", graph_path.read_text(encoding="utf-8"))
            with summary_path.open(newline="", encoding="utf-8") as source:
                summaries = list(csv.DictReader(source))
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[1]["run_id"], ids[1])


class EvaluationValidationAndPlotTests(unittest.TestCase):
    def test_validate_contract_preserves_and_reports_unknown_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "contract.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output, fieldnames=collect_metrics.CONTRACT_COLUMNS
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "timestamp": "2026-11-18T00:00:00+00:00",
                        "request_count": 5,
                        "request_rate_rps": 0.5,
                        "cpu_utilization_pct": "",
                        "memory_usage_bytes": "",
                        "replica_count": "",
                        "average_latency_ms": 12,
                        "p95_latency_ms": 18,
                        "error_count": 0,
                    }
                )
            report = validate_data.validate_csv(path)
            with redirect_stdout(StringIO()):
                self.assertEqual(validate_data.main([str(path)]), 0)
        self.assertTrue(report["valid"])
        self.assertEqual(report["format"], "contract_10s")
        self.assertEqual(report["row_count"], 1)
        self.assertEqual(report["missing_value_counts"]["replica_count"], 1)
        self.assertTrue(any("replica_count" in warning for warning in report["warnings"]))

    def test_validate_cli_returns_failure_for_missing_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.csv"
            with redirect_stdout(StringIO()):
                self.assertEqual(validate_data.main([str(missing_path)]), 1)

    def test_validate_per_second_checks_order_and_nonnegative_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "per_second.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output, fieldnames=validate_data.PER_SECOND_COLUMNS
                )
                writer.writeheader()
                for timestamp, rate in (
                    ("2026-11-18T00:00:10+00:00", 2),
                    ("2026-11-18T00:00:00+00:00", -1),
                ):
                    writer.writerow(
                        {
                            "run_id": "run",
                            "scenario": "steady",
                            "timestamp_utc": timestamp,
                            "elapsed_second": 0,
                            "requests_per_second": rate,
                            "request_count": 2,
                            "mean_response_latency_ms": "",
                            "latency_sample_count": 0,
                            "failures": "",
                            "error_rate": "",
                            "failure_sample_count": 0,
                            "configured_peak_user_count": 1,
                            "test_duration_seconds": 10,
                        }
                    )
            report = validate_data.validate_csv(path)
        self.assertFalse(report["valid"])
        self.assertTrue(any("strictly chronological" in error for error in report["errors"]))
        self.assertTrue(any("must not be negative" in error for error in report["errors"]))
        self.assertGreater(report["missing_value_counts"]["mean_response_latency_ms"], 0)

    def test_plots_contract_latency_rate_replica_and_distribution(self):
        contract = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "processed"
            / "locust_spike_k8s_hpa_20261116174859_10s_contract.csv"
        )
        distribution = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "processed"
            / "request_distribution_20261117T110345Z.json"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            contract_outputs = plot_results.plot_results(contract, temp_dir)
            distribution_outputs = plot_results.plot_results(distribution, temp_dir)
            contents = {
                path.name: path.read_text(encoding="utf-8")
                for path in (*contract_outputs, *distribution_outputs)
            }
        self.assertEqual(len(contract_outputs), 3)
        self.assertEqual(len(distribution_outputs), 1)
        self.assertTrue(any("Average latency" in svg and "P95 latency" in svg for svg in contents.values()))
        self.assertTrue(any("Request rate" in svg for svg in contents.values()))
        replicas_svg = next(value for key, value in contents.items() if "_replicas.svg" in key)
        self.assertIn("Replica count", replicas_svg)
        self.assertIn("Per-pod request distribution", contents[distribution_outputs[0].name])


class ScenarioWrapperTests(unittest.TestCase):
    def test_wrappers_select_scenario_and_forward_overrides(self):
        cases = (
            (run_steady, "steady"),
            (run_spike, "spike"),
            (run_ramp, "ramp"),
            (run_periodic, "periodic"),
        )
        for module, scenario in cases:
            with self.subTest(scenario=scenario):
                with patch.object(
                    module.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=17),
                ) as run:
                    result = module.main(
                        ["--users", "7", "--host", "http://127.0.0.1:8000"]
                    )
                command = run.call_args.args[0]
                self.assertEqual(result, 17)
                self.assertEqual(command[command.index("--scenario") + 1], scenario)
                self.assertIn("--users", command)
                self.assertIn("7", command)
                self.assertIn("--host", command)

    def test_spike_wrapper_defaults_and_respects_phase_duration_overrides(self):
        with patch.object(
            run_spike.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0),
        ) as run:
            run_spike.main(["--users", "8"])
        command = run.call_args.args[0]
        self.assertEqual(
            [
                command[command.index(option) + 1]
                for option in (
                    "--warmup-duration",
                    "--peak-duration",
                    "--recovery-duration",
                )
            ],
            ["30", "60", "30"],
        )

        with patch.object(
            run_spike.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0),
        ) as run:
            run_spike.main(
                [
                    "--duration",
                    "20",
                    "--warmup-duration",
                    "5",
                    "--peak-duration",
                    "10",
                    "--recovery-duration",
                    "5",
                ]
            )
        command = run.call_args.args[0]
        self.assertEqual(command.count("--warmup-duration"), 1)
        self.assertEqual(command[command.index("--duration") + 1], "20")


class PredictiveLiveLoopTests(unittest.TestCase):
    def start_metrics_server(self):
        class MetricsHandler(BaseHTTPRequestHandler):
            counter = 0

            def do_GET(self):
                if self.path != "/metrics/request-rate":
                    self.send_error(404)
                    return
                type(self).counter += 1
                payload = {
                    "requests_last_60_seconds": 30 * type(self).counter,
                    "request_rate_rps": 0.5 * type(self).counter,
                    "window_seconds": 60,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format_string, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), MetricsHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(thread.join, 2)
        return f"http://127.0.0.1:{server.server_port}/metrics/request-rate"

    def create_config(self, root):
        path = root / "config.yaml"
        path.write_text(
            "apiVersion: v1\n"
            "kind: ConfigMap\n"
            "metadata:\n  name: cloudpulse-config\n  namespace: cloudpulse\n"
            "data:\n"
            "  SAMPLING_INTERVAL_SECONDS: \"10\"\n"
            "  FORECAST_HORIZON_SECONDS: \"60\"\n"
            "  TARGET_RPS_PER_REPLICA: \"0.2\"\n"
            "  MIN_REPLICAS: \"1\"\n"
            "  MAX_REPLICAS: \"5\"\n"
            "  WORK_DEFAULT_UNITS: \"1000\"\n",
            encoding="utf-8",
        )
        return path

    def test_two_dry_run_cycles_poll_real_http_and_do_not_patch(self):
        metrics_url = self.start_metrics_server()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            apps_api = MockAppsV1Api()
            output = live_loop.run_loop(
                iterations=2,
                interval=0.01,
                metrics_url=metrics_url,
                predictor_name="moving_average",
                model_path=root / "missing.joblib",
                config_path=self.create_config(root),
                namespace="cloudpulse",
                deployment="cloudpulse-service",
                output_csv=root / "dry-run.csv",
                apps_api=apps_api,
                sleep_fn=lambda _: None,
            )
            with output.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["observed_rps"] for row in rows], ["0.5", "1.0"])
        self.assertEqual([row["predicted_rps"] for row in rows], ["0.5", "0.75"])
        self.assertEqual(rows[1]["predictor_used"], "moving_average_warmup")
        self.assertEqual(rows[0]["action"], "dry-run")
        self.assertEqual(rows[0]["current_replicas"], "1")
        self.assertEqual(rows[0]["target_replicas"], "3")
        self.assertEqual(apps_api.patches, [])

    def test_xgboost_predictor_uses_moving_average_during_lag_warmup(self):
        prediction, method = live_loop._predict(
            [2.0, 4.0],
            [
                datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc),
                datetime(2026, 11, 18, 12, 0, 10, tzinfo=timezone.utc),
            ],
            "xgboost",
            None,
            10,
            datetime(2026, 11, 18, 12, 0, 11, tzinfo=timezone.utc),
        )
        self.assertEqual(prediction, 3.0)
        self.assertEqual(method, "moving_average_xgboost_warmup")

    def test_live_mode_scales_only_when_no_matching_hpa_exists(self):
        metrics_url = self.start_metrics_server()

        class EmptyAutoscalingApi:
            def list_namespaced_horizontal_pod_autoscaler(self, namespace):
                return SimpleNamespace(items=[])

        class MatchingHpaApi:
            def list_namespaced_horizontal_pod_autoscaler(self, namespace):
                target = SimpleNamespace(name="cloudpulse-service")
                return SimpleNamespace(
                    items=[SimpleNamespace(spec=SimpleNamespace(scale_target_ref=target))]
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self.create_config(root)
            apps_api = MockAppsV1Api()
            live_loop.run_loop(
                iterations=1,
                interval=0.01,
                metrics_url=metrics_url,
                predictor_name="moving_average",
                model_path=root / "missing.joblib",
                config_path=config,
                namespace="cloudpulse",
                deployment="cloudpulse-service",
                live=True,
                output_csv=root / "live.csv",
                apps_api=apps_api,
                autoscaling_api=EmptyAutoscalingApi(),
            )
            self.assertEqual(len(apps_api.patches), 1)

            blocked_api = MockAppsV1Api()
            blocked_output = live_loop.run_loop(
                iterations=1,
                interval=0.01,
                metrics_url=metrics_url,
                predictor_name="moving_average",
                model_path=root / "missing.joblib",
                config_path=config,
                namespace="cloudpulse",
                deployment="cloudpulse-service",
                live=True,
                output_csv=root / "blocked.csv",
                apps_api=blocked_api,
                autoscaling_api=MatchingHpaApi(),
            )
            with blocked_output.open(newline="", encoding="utf-8") as source:
                row = next(csv.DictReader(source))
        self.assertEqual(blocked_api.patches, [])
        self.assertEqual(row["action"], "blocked_hpa")
        self.assertIn("Matching HPA is active", row["reason"])


class MockAppsV1Api:
    def __init__(self):
        self.replicas = 1
        self.patches = []

    def read_namespaced_deployment_scale(self, name, namespace):
        return SimpleNamespace(spec=SimpleNamespace(replicas=self.replicas))

    def patch_namespaced_deployment_scale(self, name, namespace, body):
        self.patches.append((name, namespace, body))
        self.replicas = body["spec"]["replicas"]


if __name__ == "__main__":
    unittest.main()
