"""Unit tests for replica calculations and scaling safety rules."""

import unittest
from datetime import datetime, timedelta, timezone
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from prediction.src.config import ScalingConfig, load_scaling_config
from prediction.src.controller import (
	PredictiveScalingController,
	create_kubernetes_apps_api,
)
from prediction.src.safety_rules import apply_safety_rules
from prediction.src.scaling_logic import calculate_desired_replicas


class ScalingLogicTests(unittest.TestCase):
	def setUp(self) -> None:
		self.config = ScalingConfig(
			capacity_rps_per_replica=100.0,
			min_replicas=1,
			max_replicas=8,
		)

	def test_replica_calculation_uses_explicit_capacity_assumption(self) -> None:
		self.assertEqual(calculate_desired_replicas(250, self.config), 3)
		self.assertEqual(calculate_desired_replicas(0, self.config), 1)
		self.assertEqual(calculate_desired_replicas(10_000, self.config), 8)

	def test_invalid_predictions_are_rejected(self) -> None:
		for value in (-1, float("nan"), float("inf"), True, "invalid"):
			with self.subTest(value=value):
				with self.assertRaises(ValueError):
					calculate_desired_replicas(value, self.config)

	def test_capacity_must_be_explicit_and_valid(self) -> None:
		for capacity in (0, -1, float("nan"), float("inf"), True):
			with self.subTest(capacity=capacity):
				with self.assertRaises(ValueError):
					ScalingConfig(capacity_rps_per_replica=capacity)

	def test_config_replica_bounds_are_validated(self) -> None:
		with self.assertRaisesRegex(ValueError, "max_replicas"):
			ScalingConfig(capacity_rps_per_replica=100, min_replicas=5, max_replicas=4)


class SafetyRuleTests(unittest.TestCase):
	def setUp(self) -> None:
		self.now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
		self.config = ScalingConfig(
			capacity_rps_per_replica=100,
			min_replicas=1,
			max_replicas=8,
			cooldown_seconds=300,
			max_scale_up_step=2,
			max_scale_down_step=1,
			max_forecast_age_seconds=120,
		)

	def test_missing_or_stale_forecasts_hold_current_replicas(self) -> None:
		missing = apply_safety_rules(
			5, 3, self.config, forecast_timestamp=None, now=self.now
		)
		stale = apply_safety_rules(
			5,
			3,
			self.config,
			forecast_timestamp=self.now - timedelta(seconds=121),
			now=self.now,
		)
		self.assertFalse(missing.allowed)
		self.assertFalse(stale.allowed)
		self.assertEqual(missing.target_replicas, 3)
		self.assertEqual(stale.target_replicas, 3)

	def test_future_forecast_holds_current_replicas(self) -> None:
		decision = apply_safety_rules(
			6,
			3,
			self.config,
			forecast_timestamp=self.now + timedelta(seconds=1),
			now=self.now,
		)
		self.assertFalse(decision.allowed)
		self.assertEqual(decision.target_replicas, 3)

	def test_cooldown_holds_current_replicas(self) -> None:
		decision = apply_safety_rules(
			6,
			3,
			self.config,
			forecast_timestamp=self.now,
			last_scale_time=self.now - timedelta(seconds=30),
			now=self.now,
		)
		self.assertFalse(decision.allowed)
		self.assertEqual(decision.target_replicas, 3)

	def test_scale_step_limits_and_replica_bounds(self) -> None:
		up = apply_safety_rules(
			8, 3, self.config, forecast_timestamp=self.now, now=self.now
		)
		down = apply_safety_rules(
			1, 3, self.config, forecast_timestamp=self.now, now=self.now
		)
		self.assertEqual(up.target_replicas, 5)
		self.assertEqual(down.target_replicas, 2)
		self.assertTrue(up.allowed)
		self.assertTrue(down.allowed)

	def test_exact_forecast_age_and_cooldown_boundaries_are_allowed(self) -> None:
		decision = apply_safety_rules(
			8,
			6,
			self.config,
			forecast_timestamp=self.now - timedelta(seconds=120),
			last_scale_time=self.now - timedelta(seconds=300),
			now=self.now,
		)
		self.assertTrue(decision.allowed)
		self.assertEqual(decision.target_replicas, 8)

	def test_timezone_aware_timestamps_with_different_offsets_work(self) -> None:
		local_now = self.now.astimezone(timezone(timedelta(hours=5, minutes=30)))
		decision = apply_safety_rules(
			4,
			2,
			self.config,
			forecast_timestamp=self.now - timedelta(seconds=10),
			now=local_now,
		)
		self.assertTrue(decision.allowed)

	def test_naive_or_future_scale_timestamps_are_rejected(self) -> None:
		for kwargs in (
			{"forecast_timestamp": datetime(2026, 1, 1, 11, 59, 50)},
			{
				"forecast_timestamp": self.now,
				"last_scale_time": datetime(2026, 1, 1, 11, 55),
			},
			{
				"forecast_timestamp": self.now,
				"last_scale_time": self.now + timedelta(seconds=1),
			},
		):
			with self.subTest(kwargs=kwargs):
				with self.assertRaises(ValueError):
					apply_safety_rules(4, 2, self.config, now=self.now, **kwargs)

	def test_invalid_current_or_desired_counts_are_rejected(self) -> None:
		for current, desired in ((0, 2), (9, 2), (3, 0), (3, 9), (True, 2)):
			with self.subTest(current=current, desired=desired):
				with self.assertRaises(ValueError):
					apply_safety_rules(
						desired,
						current,
						self.config,
						forecast_timestamp=self.now,
						now=self.now,
					)


class ControllerTests(unittest.TestCase):
	class FakeAppsApi:
		def __init__(self, current_replicas=2, fail_read=False, fail_patch=False):
			self.current_replicas = current_replicas
			self.fail_read = fail_read
			self.fail_patch = fail_patch
			self.patches = []
			self.reads = []

		def read_namespaced_deployment_scale(self, name, namespace):
			self.reads.append((name, namespace))
			if self.fail_read:
				raise RuntimeError("simulated API read failure")
			return SimpleNamespace(spec=SimpleNamespace(replicas=self.current_replicas))

		def patch_namespaced_deployment_scale(self, name, namespace, body):
			if self.fail_patch:
				raise RuntimeError("simulated API patch failure")
			self.patches.append((name, namespace, body))

	def setUp(self) -> None:
		self.now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
		self.forecast_time = self.now - timedelta(seconds=10)
		self.config = ScalingConfig(
			capacity_rps_per_replica=100,
			min_replicas=1,
			max_replicas=8,
			max_scale_up_step=2,
			max_scale_down_step=1,
			max_forecast_age_seconds=120,
		)

	def test_dry_run_reads_and_logs_proposal_without_patching(self) -> None:
		api = self.FakeAppsApi(current_replicas=2)
		controller = PredictiveScalingController(self.config, apps_api=api)
		result = controller.reconcile(
			"cloudpulse", "default", 500, self.forecast_time, now=self.now
		)
		self.assertEqual(result.action, "dry-run")
		self.assertEqual(result.decision.target_replicas, 4)
		self.assertEqual(api.reads, [("cloudpulse", "default")])
		self.assertEqual(api.patches, [])

	def test_missing_or_invalid_forecast_never_patches(self) -> None:
		api = self.FakeAppsApi(current_replicas=2)
		controller = PredictiveScalingController(self.config, apps_api=api)
		for forecast in (
			None,
			-1,
			float("nan"),
			float("inf"),
			float("-inf"),
			True,
			"not-a-number",
			object(),
		):
			with self.subTest(forecast=forecast):
				result = controller.reconcile(
					"cloudpulse", "default", forecast, self.forecast_time, now=self.now
				)
				self.assertEqual(result.action, "no-op")
				self.assertEqual(result.decision.target_replicas, 2)
		self.assertEqual(api.patches, [])

	def test_invalid_current_counts_are_rejected_before_prediction_fallback(self) -> None:
		controller = PredictiveScalingController(self.config)
		for current in (True, 0, -1, 9, 2.5, "2"):
			for forecast in (None, -1, float("nan")):
				with self.subTest(current=current, forecast=forecast):
					with self.assertRaises(ValueError):
						controller.decide(forecast, current, self.forecast_time)

	def test_malformed_api_replica_counts_return_error_without_patching(self) -> None:
		for current in (None, True, 0, -1, 9, 2.5, "2"):
			with self.subTest(current=current):
				api = self.FakeAppsApi(current_replicas=current)
				controller = PredictiveScalingController(self.config, apps_api=api)
				result = controller.reconcile(
					"cloudpulse", "default", None, self.forecast_time, now=self.now
				)
				self.assertEqual(result.action, "error")
				self.assertIsNone(result.decision)
				self.assertEqual(api.patches, [])

	def test_decision_exception_returns_error_without_patching(self) -> None:
		api = self.FakeAppsApi(current_replicas=2)
		controller = PredictiveScalingController(self.config, apps_api=api)
		with mock.patch.object(controller, "decide", side_effect=ValueError("bad time")):
			result = controller.reconcile(
				"cloudpulse", "default", 500, self.forecast_time, now=self.now
			)
		self.assertEqual(result.action, "error")
		self.assertIn("Could not evaluate scaling decision", result.message)
		self.assertIsNone(result.decision)
		self.assertEqual(api.patches, [])

	def test_live_patch_requires_explicit_dry_run_false(self) -> None:
		live_config = ScalingConfig(
			capacity_rps_per_replica=100,
			max_scale_up_step=2,
			dry_run=False,
		)
		api = self.FakeAppsApi(current_replicas=2)
		controller = PredictiveScalingController(live_config, apps_api=api)
		result = controller.reconcile(
			"cloudpulse", "default", 500, self.forecast_time, now=self.now
		)
		self.assertEqual(result.action, "patched")
		self.assertEqual(api.patches[0][2], {"spec": {"replicas": 4}})

	def test_kubernetes_api_errors_are_returned_without_crashing(self) -> None:
		api = self.FakeAppsApi(fail_read=True)
		controller = PredictiveScalingController(self.config, apps_api=api)
		result = controller.reconcile(
			"cloudpulse", "default", 500, self.forecast_time, now=self.now
		)
		self.assertEqual(result.action, "error")
		self.assertIn("simulated API read failure", result.message)

	def test_missing_api_client_is_reported_without_cluster_access(self) -> None:
		controller = PredictiveScalingController(self.config)
		result = controller.reconcile(
			"cloudpulse", "default", 500, self.forecast_time, now=self.now
		)
		self.assertEqual(result.action, "error")
		self.assertIn("No Kubernetes AppsV1Api client", result.message)

	def test_patch_api_failure_is_returned_without_crashing(self) -> None:
		live_config = ScalingConfig(
			capacity_rps_per_replica=100,
			max_scale_up_step=2,
			dry_run=False,
		)
		api = self.FakeAppsApi(current_replicas=2, fail_patch=True)
		controller = PredictiveScalingController(live_config, apps_api=api)
		result = controller.reconcile(
			"cloudpulse", "default", 500, self.forecast_time, now=self.now
		)
		self.assertEqual(result.action, "error")
		self.assertIn("simulated API patch failure", result.message)

	def test_config_loads_explicit_capacity_from_json(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "scaling.json"
			path.write_text(
				json.dumps({"capacity_rps_per_replica": 250, "dry_run": True}),
				encoding="utf-8",
			)
			config = load_scaling_config(path)
		self.assertEqual(config.capacity_rps_per_replica, 250)
		self.assertEqual(config.capacity_basis, "UNVERIFIED ASSUMPTION")
		self.assertTrue(config.dry_run)

	def test_config_requires_explicit_capacity(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "scaling.json"
			path.write_text("{}", encoding="utf-8")
			with self.assertRaisesRegex(ValueError, "must explicitly provide"):
				load_scaling_config(path)

	def test_kubernetes_client_uses_local_kubeconfig_by_default(self) -> None:
		with (
			mock.patch("kubernetes.config.load_kube_config") as load_kubeconfig,
			mock.patch("kubernetes.client.AppsV1Api") as api_factory,
		):
			api = create_kubernetes_apps_api()

		load_kubeconfig.assert_called_once_with()
		api_factory.assert_called_once_with()
		self.assertEqual(api, api_factory.return_value)

	def test_kubernetes_client_supports_in_cluster_configuration(self) -> None:
		with (
			mock.patch("kubernetes.config.load_incluster_config") as load_incluster,
			mock.patch("kubernetes.client.AppsV1Api") as api_factory,
		):
			api = create_kubernetes_apps_api(in_cluster=True)

		load_incluster.assert_called_once_with()
		api_factory.assert_called_once_with()
		self.assertEqual(api, api_factory.return_value)


if __name__ == "__main__":
	unittest.main()
