"""Tests for the moving-average baseline using labelled synthetic values."""

import math
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd

from prediction.src.moving_average import (
	evaluate_predictions,
	evaluate_walk_forward,
	moving_average,
	walk_forward_predictions,
)
from prediction.src.predict import run_baseline


SYNTHETIC_CSV = (
	"timestamp,request_rate_rps\n"
	"2026-10-12T10:00:00Z,20\n"
	"2026-10-12T10:00:10Z,25\n"
	"2026-10-12T10:00:20Z,23\n"
	"2026-10-12T10:00:30Z,30\n"
	"2026-10-12T10:00:40Z,35\n"
)


class MovingAverageTests(unittest.TestCase):
	def test_known_moving_average(self) -> None:
		synthetic_rates = [10, 20, 30]
		self.assertEqual(moving_average(synthetic_rates, window_size=2), 25.0)

	def test_different_valid_window_sizes(self) -> None:
		synthetic_rates = [2, 4, 6, 8]
		self.assertEqual(moving_average(synthetic_rates, window_size=1), 8.0)
		self.assertEqual(moving_average(synthetic_rates, window_size=3), 6.0)

	def test_insufficient_historical_observations(self) -> None:
		with self.assertRaisesRegex(ValueError, "Not enough historical observations"):
			moving_average([10, 20], window_size=3)

	def test_empty_request_rates_are_rejected(self) -> None:
		with self.assertRaisesRegex(ValueError, "available=0"):
			moving_average([], window_size=2)
		with self.assertRaisesRegex(ValueError, "available=0"):
			evaluate_walk_forward([], window_size=2)

	def test_invalid_window_sizes(self) -> None:
		for invalid_window in (0, -1, 1.5, True, "2"):
			with self.subTest(window=invalid_window):
				with self.assertRaisesRegex(ValueError, "positive integer"):
					moving_average([10, 20], window_size=invalid_window)

	def test_missing_nonnumeric_and_nonfinite_rates_are_rejected(self) -> None:
		invalid_values = (
			[10, None],
			[10, "not-a-number"],
			[10, float("inf")],
			[10, float("nan")],
			[10, True],
		)
		for rates in invalid_values:
			with self.subTest(rates=rates):
				with self.assertRaisesRegex(ValueError, "request_rates\\[1\\]"):
					moving_average(rates, window_size=2)

	def test_negative_request_rates_are_rejected(self) -> None:
		with self.assertRaisesRegex(ValueError, "must not be negative"):
			moving_average([10, -1], window_size=2)

	def test_walk_forward_uses_only_observations_before_each_target(self) -> None:
		earlier_synthetic_rates = [10, 20, 30, 40]
		predictions = walk_forward_predictions(earlier_synthetic_rates, 2)
		extended_predictions = walk_forward_predictions(
			earlier_synthetic_rates + [10000], 2
		)

		self.assertEqual(predictions, [15.0, 25.0])
		self.assertEqual(extended_predictions[: len(predictions)], predictions)

	def test_walk_forward_output_length_and_values(self) -> None:
		synthetic_rates = [10, 20, 30, 40, 50]
		predictions = walk_forward_predictions(synthetic_rates, window_size=2)

		self.assertEqual(len(predictions), len(synthetic_rates) - 2)
		self.assertEqual(predictions, [15.0, 25.0, 35.0])

	def test_walk_forward_requires_a_target_after_initial_window(self) -> None:
		with self.assertRaisesRegex(ValueError, "at least one observation"):
			walk_forward_predictions([10, 20], window_size=2)

	def test_evaluation_metrics_use_requests_per_second(self) -> None:
		report = evaluate_predictions(
			actual_rates=[10, 20], predicted_rates=[12, 16]
		)

		self.assertEqual(report["mae_rps"], 3.0)
		self.assertAlmostEqual(report["rmse_rps"], math.sqrt(10))
		self.assertEqual(report["n_predictions"], 2)

	def test_walk_forward_evaluation_scores_only_later_targets(self) -> None:
		synthetic_rates = [10, 20, 30, 40, 50]
		report = evaluate_walk_forward(synthetic_rates, window_size=2)

		self.assertEqual(report["mae_rps"], 15.0)
		self.assertEqual(report["rmse_rps"], 15.0)
		self.assertEqual(report["n_predictions"], 3)

	def test_walk_forward_evaluation_is_reproducible(self) -> None:
		synthetic_rates = [10, 20, 30, 40, 50]
		first_report = evaluate_walk_forward(synthetic_rates, window_size=2)
		second_report = evaluate_walk_forward(synthetic_rates, window_size=2)

		self.assertEqual(first_report, second_report)

	def test_walk_forward_evaluation_reports_too_few_targets(self) -> None:
		with self.assertRaisesRegex(ValueError, "at least two target observations"):
			evaluate_walk_forward([10, 20, 30], window_size=2)

	def test_evaluation_rejects_mismatched_lengths(self) -> None:
		with self.assertRaisesRegex(ValueError, "same number of values"):
			evaluate_predictions([10, 20], [12])


class BaselineOutputTests(unittest.TestCase):
	def run_with_temporary_outputs(self, directory: str):
		root = Path(directory)
		input_csv = root / "synthetic_input.csv"
		predictions_csv = root / "out" / "predictions.csv"
		metrics_json = root / "out" / "metrics.json"
		graph_png = root / "graphs" / "actual_vs_predicted.png"
		input_csv.write_text(SYNTHETIC_CSV, encoding="utf-8")

		metrics = run_baseline(
			input_csv=input_csv,
			window_size=2,
			predictions_output=predictions_csv,
			metrics_output=metrics_json,
			graph_output=graph_png,
			data_label="synthetic unit-test fixture",
		)
		return predictions_csv, metrics_json, graph_png, metrics

	def test_forecast_timestamps_actuals_and_predictions_align(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			predictions_path, _, _, _ = self.run_with_temporary_outputs(directory)
			predictions = pd.read_csv(predictions_path)

		timestamps = pd.to_datetime(predictions["timestamp"], utc=True)
		expected_timestamps = pd.to_datetime(
			[
				"2026-10-12T10:00:20Z",
				"2026-10-12T10:00:30Z",
				"2026-10-12T10:00:40Z",
			],
			utc=True,
		)
		self.assertTrue(timestamps.equals(pd.Series(expected_timestamps)))
		self.assertEqual(predictions["actual_request_rate_rps"].tolist(), [23, 30, 35])
		self.assertEqual(
			predictions["predicted_request_rate_rps"].tolist(), [22.5, 24.0, 26.5]
		)

	def test_csv_json_save_and_metrics_match_saved_predictions(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			predictions_path, metrics_path, _, returned_metrics = (
				self.run_with_temporary_outputs(directory)
			)
			saved_predictions = pd.read_csv(predictions_path)
			saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

		errors = (
			saved_predictions["actual_request_rate_rps"]
			- saved_predictions["predicted_request_rate_rps"]
		)
		expected_mae = errors.abs().mean()
		expected_rmse = math.sqrt((errors**2).mean())
		self.assertEqual(len(saved_predictions), 3)
		self.assertEqual(saved_metrics["number_of_evaluated_predictions"], 3)
		self.assertEqual(saved_metrics["forecast_method"], "moving_average_walk_forward")
		self.assertEqual(saved_metrics["window_size"], 2)
		self.assertEqual(saved_metrics["run_metadata"]["data_label"], "synthetic unit-test fixture")
		self.assertAlmostEqual(saved_metrics["mae_rps"], expected_mae)
		self.assertAlmostEqual(saved_metrics["rmse_rps"], expected_rmse)
		self.assertEqual(returned_metrics, saved_metrics)

	def test_graph_is_written_as_a_readable_png(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			_, _, graph_path, _ = self.run_with_temporary_outputs(directory)
			image = mpimg.imread(graph_path)

		self.assertGreater(image.shape[0], 0)
		self.assertGreater(image.shape[1], 0)

	def test_invalid_workload_is_rejected_without_writing_outputs(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			input_csv = root / "invalid.csv"
			input_csv.write_text(
				"timestamp,request_rate_rps\n"
				"2026-10-12T10:00:00Z,20\n"
				"2026-10-12T10:00:10Z,not-a-rate\n"
				"2026-10-12T10:00:20Z,23\n"
				"2026-10-12T10:00:30Z,30\n",
				encoding="utf-8",
			)
			with self.assertRaisesRegex(ValueError, "nonnumeric"):
				run_baseline(
					input_csv,
					2,
					root / "predictions.csv",
					root / "metrics.json",
					root / "graph.png",
				)
			self.assertFalse((root / "predictions.csv").exists())
			self.assertFalse((root / "metrics.json").exists())
			self.assertFalse((root / "graph.png").exists())

	def test_insufficient_data_is_rejected_without_writing_outputs(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			input_csv = root / "too_short.csv"
			input_csv.write_text(
				"timestamp,request_rate_rps\n"
				"2026-10-12T10:00:00Z,20\n"
				"2026-10-12T10:00:10Z,25\n"
				"2026-10-12T10:00:20Z,23\n",
				encoding="utf-8",
			)
			with self.assertRaisesRegex(ValueError, "at least window_size \\+ 2 rows"):
				run_baseline(
					input_csv,
					2,
					root / "predictions.csv",
					root / "metrics.json",
					root / "graph.png",
				)


if __name__ == "__main__":
	unittest.main()
