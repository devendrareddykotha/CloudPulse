"""Tests for leakage-safe XGBoost training using synthetic workload fixtures."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd

from prediction.src.train_model import build_lagged_features, train_and_evaluate


def make_synthetic_workload(path: Path, count: int = 90) -> None:
	timestamps = pd.date_range("2026-01-01", periods=count, freq="min", tz="UTC")
	rates = [20.0 + ((index * 7) % 13) + index * 0.1 for index in range(count)]
	pd.DataFrame(
		{"timestamp": timestamps, "request_rate_rps": rates}
	).to_csv(path, index=False)


class LaggedFeatureTests(unittest.TestCase):
	def test_features_use_only_observations_before_the_target(self) -> None:
		workload = pd.DataFrame(
			{
				"timestamp": pd.date_range("2026-01-01", periods=5, freq="min", tz="UTC"),
				"request_rate_rps": [10, 20, 30, 40, 50],
			}
		)
		features, columns, warmup_rows = build_lagged_features(workload)

		self.assertEqual(warmup_rows, 3)
		self.assertEqual(features["source_row"].tolist(), [3, 4])
		self.assertEqual(features.loc[0, "lag_1_rps"], 30)
		self.assertEqual(features.loc[0, "lag_2_rps"], 20)
		self.assertEqual(features.loc[0, "lag_3_rps"], 10)
		self.assertEqual(features.loc[0, "rolling_mean_3_rps"], 20)
		self.assertEqual(len(columns), 4)

	def test_rejects_invalid_rate_input(self) -> None:
		workload = pd.DataFrame(
			{
				"timestamp": pd.date_range("2026-01-01", periods=5, freq="min", tz="UTC"),
				"request_rate_rps": [10, 20, -1, 40, 50],
			}
		)
		with self.assertRaisesRegex(ValueError, "must not be negative"):
			build_lagged_features(workload)


class XGBoostTrainingTests(unittest.TestCase):
	def test_chronological_training_saves_model_metrics_predictions_and_graph(self) -> None:
		with tempfile.TemporaryDirectory() as temporary_directory:
			root = Path(temporary_directory)
			input_path = root / "synthetic_proxy.csv"
			model_path = root / "model.joblib"
			predictions_path = root / "predictions.csv"
			metrics_path = root / "metrics.json"
			graph_path = root / "evaluation.png"
			make_synthetic_workload(input_path)

			metrics = train_and_evaluate(
				input_csv=input_path,
				model_output=model_path,
				predictions_output=predictions_path,
				metrics_output=metrics_path,
				graph_output=graph_path,
				dataset_label="synthetic unit-test fixture",
				expected_sampling_interval="1min",
			)
			saved_predictions = pd.read_csv(predictions_path)
			saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
			image = mpimg.imread(graph_path)
			outputs_exist = all(
				path.is_file()
				for path in (model_path, predictions_path, metrics_path, graph_path)
			)

		self.assertTrue(outputs_exist)
		self.assertEqual(saved_predictions["split"].iloc[0], "validation")
		self.assertEqual(saved_predictions["split"].iloc[-1], "test")
		self.assertEqual(len(saved_predictions), 35)
		self.assertEqual(saved_metrics["metadata"]["dataset_label"], "synthetic unit-test fixture")
		self.assertEqual(saved_metrics["metadata"]["split_rows"], {"train": 52, "validation": 17, "test": 18})
		self.assertEqual(metrics, saved_metrics)
		self.assertGreater(image.size, 0)
		for split in ("validation", "test"):
			split_rows = saved_predictions.loc[
				saved_predictions["split"] == split
			]
			actual = split_rows["actual_request_rate_rps"]
			for method, column in (
				("xgboost", "xgboost_prediction_rps"),
				("moving_average", "moving_average_prediction_rps"),
			):
				errors = actual - split_rows[column]
				recomputed_mae = errors.abs().mean()
				recomputed_rmse = math.sqrt((errors**2).mean())
				self.assertAlmostEqual(
					saved_metrics[split][method]["mae_rps"],
					recomputed_mae,
					places=10,
				)
				self.assertAlmostEqual(
					saved_metrics[split][method]["rmse_rps"],
					recomputed_rmse,
					places=10,
				)

	def test_refuses_to_overwrite_existing_outputs(self) -> None:
		with tempfile.TemporaryDirectory() as temporary_directory:
			root = Path(temporary_directory)
			input_path = root / "input.csv"
			model_path = root / "model.joblib"
			predictions_path = root / "predictions.csv"
			metrics_path = root / "metrics.json"
			graph_path = root / "graph.png"
			make_synthetic_workload(input_path)
			metrics_path.write_text("keep", encoding="utf-8")

			with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
				train_and_evaluate(
					input_path,
					model_path,
					predictions_path,
					metrics_path,
					graph_path,
				)
			self.assertEqual(metrics_path.read_text(encoding="utf-8"), "keep")

	def test_rejects_too_few_rows_before_training(self) -> None:
		with tempfile.TemporaryDirectory() as temporary_directory:
			root = Path(temporary_directory)
			input_path = root / "short.csv"
			make_synthetic_workload(input_path, count=10)
			with self.assertRaisesRegex(ValueError, "Insufficient observations"):
				train_and_evaluate(
					input_path,
					root / "model.joblib",
					root / "predictions.csv",
					root / "metrics.json",
					root / "graph.png",
				)


if __name__ == "__main__":
	unittest.main()