"""Offline tests for next-step inference and pure scaling integration."""

import unittest
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import tempfile
from unittest import mock

import joblib
import pandas as pd

from prediction.src.config import ScalingConfig
from prediction.src.controller import PredictiveScalingController
from prediction.src.inference import (
	EXPECTED_FEATURE_NAMES,
	forecast_and_decide,
	forecast_next_step,
	load_saved_predictor,
)


class PredictorStub:
	def __init__(self, output):
		self.output = output
		self.received = None

	def predict(self, features):
		self.received = features.copy()
		if isinstance(self.output, Exception):
			raise self.output
		return self.output


class SerializedEstimator:
	"""Small picklable predictor fixture with explicit training metadata."""

	def __init__(self, output=35.0, feature_names=None, feature_count=4):
		self.output = output
		self.feature_names_in_ = list(
			EXPECTED_FEATURE_NAMES if feature_names is None else feature_names
		)
		self.n_features_in_ = feature_count
		self.received = None

	def predict(self, features):
		self.received = features.copy()
		if isinstance(self.output, Exception):
			raise self.output
		return self.output


class SerializedObjectWithoutPredict:
	pass


class EstimatorWithoutFeatureNames:
	def __init__(self):
		self.n_features_in_ = 4

	def predict(self, features):
		return [1.0]


class FakeApiThatMustNotBeCalled:
	def __init__(self):
		self.calls = []

	def read_namespaced_deployment_scale(self, **kwargs):
		self.calls.append(("read", kwargs))
		raise AssertionError("offline integration must not read Kubernetes")

	def patch_namespaced_deployment_scale(self, **kwargs):
		self.calls.append(("patch", kwargs))
		raise AssertionError("offline integration must not patch Kubernetes")


class OfflineInferenceTests(unittest.TestCase):
	def setUp(self) -> None:
		history_end = pd.Timestamp.now(tz="UTC").floor("min")
		self.history = pd.DataFrame(
			{
				"timestamp": pd.date_range(
					end=history_end, periods=4, freq="min"
				),
				"request_rate_rps": [10.0, 20.0, 30.0, 40.0],
			}
		)
		target_at = history_end.to_pydatetime() + timedelta(minutes=1)
		self.generated_at = min(
			datetime.now(timezone.utc), target_at - timedelta(seconds=1)
		)

	def _save_temporary_estimator(self, estimator) -> Path:
		temporary_directory = tempfile.TemporaryDirectory()
		self.addCleanup(temporary_directory.cleanup)
		path = Path(temporary_directory.name) / "temporary_model.joblib"
		joblib.dump(estimator, path)
		return path

	def test_loads_compatible_temporary_estimator(self) -> None:
		path = self._save_temporary_estimator(SerializedEstimator())
		loaded = load_saved_predictor(path)
		self.assertIsInstance(loaded, SerializedEstimator)
		self.assertEqual(tuple(loaded.feature_names_in_), EXPECTED_FEATURE_NAMES)
		self.assertEqual(loaded.n_features_in_, 4)

	def test_missing_model_path_has_clear_error(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			missing_path = Path(directory) / "missing.joblib"
			with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
				load_saved_predictor(missing_path)

	def test_unreadable_model_file_has_clear_error(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			invalid_path = Path(directory) / "not_a_model.joblib"
			invalid_path.write_bytes(b"not a valid joblib file")
			with self.assertRaisesRegex(ValueError, "Could not load saved model"):
				load_saved_predictor(invalid_path)

	def test_rejects_loaded_object_without_callable_predict(self) -> None:
		path = self._save_temporary_estimator(SerializedObjectWithoutPredict())
		with self.assertRaisesRegex(TypeError, "callable predict"):
			load_saved_predictor(path)

	def test_rejects_incompatible_feature_names(self) -> None:
		wrong_names = list(EXPECTED_FEATURE_NAMES)
		wrong_names[0], wrong_names[1] = wrong_names[1], wrong_names[0]
		path = self._save_temporary_estimator(
			SerializedEstimator(feature_names=wrong_names)
		)
		with self.assertRaisesRegex(ValueError, "feature names do not match"):
			load_saved_predictor(path)

	def test_rejects_incorrect_feature_count(self) -> None:
		path = self._save_temporary_estimator(SerializedEstimator(feature_count=3))
		with self.assertRaisesRegex(ValueError, "expects 3 features"):
			load_saved_predictor(path)

	def test_requires_feature_names_when_only_count_is_available(self) -> None:
		path = self._save_temporary_estimator(EstimatorWithoutFeatureNames())
		with self.assertRaisesRegex(ValueError, "feature order cannot be verified"):
			load_saved_predictor(path)

	def test_loaded_estimator_runs_through_inference_adapter(self) -> None:
		path = self._save_temporary_estimator(SerializedEstimator(output=[35.0]))
		loaded = load_saved_predictor(path)
		forecast = forecast_next_step(
			self.history,
			loaded,
			sampling_interval="1min",
			generated_at=self.generated_at,
		)
		self.assertEqual(forecast.predicted_value, 35.0)
		self.assertEqual(loaded.received.columns.tolist(), list(EXPECTED_FEATURE_NAMES))

	def test_loaded_estimator_invalid_output_and_exception_are_rejected(self) -> None:
		for output in (-1.0, float("nan"), float("inf"), float("-inf"), "bad", []):
			with self.subTest(output=output):
				loaded = load_saved_predictor(
					self._save_temporary_estimator(SerializedEstimator(output=output))
				)
				with self.assertRaises(ValueError):
					forecast_next_step(
						self.history,
						loaded,
						sampling_interval="1min",
						generated_at=self.generated_at,
					)
		loaded = load_saved_predictor(
			self._save_temporary_estimator(
				SerializedEstimator(output=RuntimeError("prediction failed"))
			)
		)
		with self.assertRaisesRegex(RuntimeError, "prediction failed"):
			forecast_next_step(
				self.history,
				loaded,
				sampling_interval="1min",
				generated_at=self.generated_at,
			)

	def test_builds_latest_feature_row_in_training_column_order(self) -> None:
		predictor = PredictorStub([35.0])
		forecast = forecast_next_step(
			self.history,
			predictor,
			sampling_interval="1min",
			generated_at=self.generated_at,
			interpretation="synthetic test workload; prototype forecast",
		)
		self.assertEqual(
			predictor.received.columns.tolist(),
			["lag_1_rps", "lag_2_rps", "lag_3_rps", "rolling_mean_3_rps"],
		)
		self.assertEqual(
			predictor.received.iloc[0].tolist(), [40.0, 30.0, 20.0, 30.0]
		)
		self.assertEqual(forecast.predicted_value, 35.0)
		self.assertEqual(forecast.interpretation, "synthetic test workload; prototype forecast")

	def test_generation_time_and_next_target_time_are_separate(self) -> None:
		predictor = PredictorStub(35.0)
		forecast = forecast_next_step(
			self.history,
			predictor,
			sampling_interval="1min",
			generated_at=self.generated_at,
		)
		self.assertEqual(forecast.generated_at, self.generated_at)
		self.assertEqual(
			forecast.target_at,
			self.history["timestamp"].iloc[-1].to_pydatetime() + timedelta(minutes=1),
		)
		self.assertNotEqual(forecast.generated_at, forecast.target_at)
		self.assertIsNotNone(forecast.generated_at.utcoffset())
		self.assertIsNotNone(forecast.target_at.utcoffset())

	def test_generation_before_target_succeeds_and_target_uses_sampling_interval(self) -> None:
		predictor = PredictorStub(35.0)
		target_at = self.history["timestamp"].iloc[-1].to_pydatetime() + timedelta(
			minutes=1
		)
		generated_at = target_at - timedelta(seconds=1)

		forecast = forecast_next_step(
			self.history,
			predictor,
			sampling_interval="1min",
			generated_at=generated_at,
		)

		self.assertEqual(forecast.generated_at, generated_at)
		self.assertEqual(forecast.target_at, target_at)
		self.assertLess(forecast.generated_at, forecast.target_at)

	def test_generation_equal_to_target_is_rejected(self) -> None:
		target_at = self.history["timestamp"].iloc[-1].to_pydatetime() + timedelta(
			minutes=1
		)
		with self.assertRaisesRegex(ValueError, "generated_at must be earlier than target_at"):
			forecast_next_step(
				self.history,
				PredictorStub(35.0),
				sampling_interval="1min",
				generated_at=target_at,
			)

	def test_generation_after_target_is_rejected(self) -> None:
		target_at = self.history["timestamp"].iloc[-1].to_pydatetime() + timedelta(
			minutes=1
		)
		with self.assertRaisesRegex(ValueError, "generated_at must be earlier than target_at"):
			forecast_next_step(
				self.history,
				PredictorStub(35.0),
				sampling_interval="1min",
				generated_at=target_at + timedelta(seconds=1),
			)

	def test_naive_generation_time_is_rejected(self) -> None:
		with self.assertRaisesRegex(ValueError, "generated_at must be timezone-aware"):
			forecast_next_step(
				self.history,
				PredictorStub(35.0),
				sampling_interval="1min",
				generated_at=datetime(2026, 1, 1, 0, 4),
			)

	def test_rejects_missing_or_insufficient_history(self) -> None:
		predictor = PredictorStub(35.0)
		for history in (
			pd.DataFrame(columns=["timestamp", "request_rate_rps"]),
			self.history.iloc[:2].copy(),
		):
			with self.subTest(rows=len(history)):
				with self.assertRaises(ValueError):
					forecast_next_step(history, predictor, sampling_interval="1min")

	def test_rejects_naive_timestamps_and_wrong_sampling_interval(self) -> None:
		naive = self.history.copy()
		naive["timestamp"] = naive["timestamp"].dt.tz_localize(None)
		with self.assertRaisesRegex(ValueError, "timezone-aware"):
			forecast_next_step(naive, PredictorStub(1), sampling_interval="1min")
		with self.assertRaisesRegex(ValueError, "do not match sampling_interval"):
			forecast_next_step(self.history, PredictorStub(1), sampling_interval="2min")

	def test_rejects_invalid_or_non_single_predictions(self) -> None:
		invalid = (
			None,
			-1.0,
			True,
			float("nan"),
			float("inf"),
			float("-inf"),
			"not-a-number",
			[],
			[1.0, 2.0],
		)
		for output in invalid:
			with self.subTest(output=output):
				with self.assertRaises(ValueError):
					forecast_next_step(
						self.history,
						PredictorStub(output),
						sampling_interval="1min",
						generated_at=self.generated_at,
					)

	def test_valid_synthetic_forecast_flows_into_bounded_pure_decision(self) -> None:
		config = ScalingConfig(
			capacity_rps_per_replica=10,
			min_replicas=1,
			max_replicas=5,
			max_scale_up_step=2,
		)
		controller = PredictiveScalingController(config)
		with mock.patch.object(controller, "decide", wraps=controller.decide) as decide:
			result = forecast_and_decide(
				self.history,
				PredictorStub([35.0]),
				controller,
				current_replicas=1,
				sampling_interval="1min",
				generated_at=self.generated_at,
				interpretation="synthetic test workload; prototype forecast",
			)
		self.assertEqual(result.status, "proposal")
		self.assertTrue(result.actionable)
		self.assertEqual(result.decision.requested_replicas, 4)
		self.assertEqual(result.decision.target_replicas, 3)
		self.assertGreaterEqual(result.decision.target_replicas, config.min_replicas)
		self.assertLessEqual(result.decision.target_replicas, config.max_replicas)
		self.assertEqual(
			result.forecast.target_at,
			self.history["timestamp"].iloc[-1].to_pydatetime() + timedelta(minutes=1),
		)
		self.assertEqual(decide.call_args.args[2], self.generated_at)
		self.assertNotEqual(decide.call_args.args[2], result.forecast.target_at)

	def test_invalid_prediction_returns_error_without_proposal(self) -> None:
		controller = PredictiveScalingController(
			ScalingConfig(capacity_rps_per_replica=10)
		)
		result = forecast_and_decide(
			self.history,
			PredictorStub(float("nan")),
			controller,
			current_replicas=2,
			sampling_interval="1min",
			generated_at=self.generated_at,
		)
		self.assertEqual(result.status, "error")
		self.assertFalse(result.actionable)
		self.assertIsNone(result.forecast)
		self.assertIsNone(result.decision)
		self.assertIn("no scaling proposal created", result.message)

	def test_inference_error_returns_no_proposal(self) -> None:
		controller = PredictiveScalingController(
			ScalingConfig(capacity_rps_per_replica=10)
		)
		result = forecast_and_decide(
			self.history,
			PredictorStub(RuntimeError("model failed")),
			controller,
			current_replicas=2,
			sampling_interval="1min",
			generated_at=self.generated_at,
		)
		self.assertEqual(result.status, "error")
		self.assertFalse(result.actionable)
		self.assertIsNone(result.decision)

	def test_integration_uses_no_kubernetes_client_or_api_calls(self) -> None:
		api = FakeApiThatMustNotBeCalled()
		controller = PredictiveScalingController(
			ScalingConfig(capacity_rps_per_replica=10), apps_api=api
		)
		with mock.patch(
			"prediction.src.controller.create_kubernetes_apps_api",
			side_effect=AssertionError("client factory must not be used"),
		):
			result = forecast_and_decide(
				self.history,
				load_saved_predictor(
					self._save_temporary_estimator(SerializedEstimator(output=[35.0]))
				),
				controller,
				current_replicas=1,
				sampling_interval="1min",
				generated_at=self.generated_at,
			)
		self.assertEqual(result.status, "proposal")
		self.assertEqual(api.calls, [])

	def test_existing_saved_model_end_to_end_offline_decision(self) -> None:
		model_path = (
			Path(__file__).resolve().parents[1]
			/ "models"
			/ "xgboost_service_proxy.joblib"
		)
		predictor = load_saved_predictor(model_path)
		self.assertTrue(callable(predictor.predict))

		clock = datetime.now(timezone.utc)
		history = pd.DataFrame(
			{
				"timestamp": pd.date_range(
					end=pd.Timestamp(clock), periods=6, freq="1min"
				),
				"request_rate_rps": [100.0, 120.0, 140.0, 160.0, 180.0, 200.0],
			}
		)
		config = ScalingConfig(
			capacity_rps_per_replica=100.0,
			capacity_basis="TEST-ONLY PLACEHOLDER; not measured or production guidance",
			min_replicas=1,
			max_replicas=5,
			max_forecast_age_seconds=120,
			dry_run=True,
		)
		controller = PredictiveScalingController(config)

		with mock.patch(
			"prediction.src.controller.create_kubernetes_apps_api",
			side_effect=AssertionError("offline inference must not create a Kubernetes client"),
		) as client_factory:
			result = forecast_and_decide(
				history,
				predictor,
				controller,
				current_replicas=2,
				sampling_interval="1min",
				generated_at=clock,
				interpretation="synthetic workload; prototype forecast; not verified RPS",
			)

		self.assertIn(result.status, ("proposal", "hold"))
		self.assertIsNotNone(result.forecast)
		self.assertIsNotNone(result.decision)
		self.assertTrue(math.isfinite(result.forecast.predicted_value))
		self.assertGreaterEqual(result.forecast.predicted_value, 0.0)
		self.assertEqual(result.forecast.generated_at, clock)
		self.assertIsNotNone(result.forecast.generated_at.utcoffset())
		self.assertIsNotNone(result.forecast.target_at.utcoffset())
		self.assertGreater(result.forecast.target_at, result.forecast.generated_at)
		self.assertEqual(
			result.forecast.target_at,
			history["timestamp"].iloc[-1].to_pydatetime() + timedelta(minutes=1),
		)
		self.assertGreaterEqual(result.decision.target_replicas, config.min_replicas)
		self.assertLessEqual(result.decision.target_replicas, config.max_replicas)
		self.assertEqual(
			result.forecast.interpretation,
			"synthetic workload; prototype forecast; not verified RPS",
		)
		self.assertIsNone(controller.apps_api)
		client_factory.assert_not_called()


if __name__ == "__main__":
	unittest.main()
