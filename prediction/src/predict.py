"""Command-line runner for the CloudPulse moving-average baseline."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
import pandas as pd

from prediction.src.moving_average import (
	evaluate_predictions,
	walk_forward_predictions,
)
from prediction.src.preprocessing import load_workload_csv


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS_PATH = (
	PROJECT_ROOT / "results" / "prediction" / "moving_average_predictions.csv"
)
DEFAULT_METRICS_PATH = (
	PROJECT_ROOT / "results" / "prediction" / "moving_average_metrics.json"
)
DEFAULT_GRAPH_PATH = (
	PROJECT_ROOT / "results" / "graphs" / "moving_average_actual_vs_predicted.png"
)
FORECAST_METHOD = "moving_average_walk_forward"


def _save_actual_vs_predicted_graph(
	predictions: pd.DataFrame, graph_path: Path
) -> None:
	"""Save a readable actual-versus-predicted request-rate plot."""
	graph_path.parent.mkdir(parents=True, exist_ok=True)
	figure, axis = plt.subplots(figsize=(10, 5))
	axis.plot(
		predictions["timestamp"],
		predictions["actual_request_rate_rps"],
		marker="o",
		label="Actual request rate",
	)
	axis.plot(
		predictions["timestamp"],
		predictions["predicted_request_rate_rps"],
		marker="o",
		label="Moving-average prediction",
	)
	axis.set_title("CloudPulse Moving-Average Forecast vs Actual")
	axis.set_xlabel("Forecast target timestamp")
	axis.set_ylabel("Request rate (requests per second)")
	axis.legend()
	axis.grid(True, alpha=0.3)
	figure.autofmt_xdate()
	figure.tight_layout()
	figure.savefig(graph_path, dpi=150)
	plt.close(figure)


def run_baseline(
	input_csv: Union[str, Path],
	window_size: int,
	predictions_output: Optional[Union[str, Path]] = None,
	metrics_output: Optional[Union[str, Path]] = None,
	graph_output: Optional[Union[str, Path]] = None,
	data_label: str = "unspecified",
) -> Dict[str, Any]:
	"""Run walk-forward forecasting and save aligned predictions, metrics, and plot.

	The prediction rows begin at ``window_size`` because each target must have
	that many earlier observations. Invalid rows and duplicate timestamps are
	rejected rather than dropped or silently misaligned.
	"""
	input_path = Path(input_csv).expanduser()
	prediction_path = Path(predictions_output or DEFAULT_PREDICTIONS_PATH)
	metrics_path = Path(metrics_output or DEFAULT_METRICS_PATH)
	graph_path = Path(graph_output or DEFAULT_GRAPH_PATH)

	workload = load_workload_csv(input_path)
	quality_report = workload.attrs["validation_report"]

	missing_timestamp_count = quality_report["missing_timestamps"]
	invalid_timestamp_count = quality_report["invalid_timestamps"]
	if missing_timestamp_count or invalid_timestamp_count:
		raise ValueError(
			"Forecasting requires a valid timestamp on every row; found "
			f"{missing_timestamp_count} missing and {invalid_timestamp_count} invalid."
		)

	if quality_report["duplicate_timestamps"]:
		raise ValueError(
			"Forecasting requires unique timestamps; duplicate timestamps were "
			f"found in {quality_report['duplicate_timestamps']} row(s)."
		)

	missing_rate_count = quality_report["missing_request_rates"]
	nonnumeric_rate_count = quality_report["non_numeric_request_rates"]
	negative_rate_count = quality_report["negative_request_rates"]
	if missing_rate_count or nonnumeric_rate_count or negative_rate_count:
		raise ValueError(
			"Forecasting requires valid nonnegative request rates; found "
			f"{missing_rate_count} missing, {nonnumeric_rate_count} nonnumeric, "
			f"and {negative_rate_count} negative value(s)."
		)

	rates = workload["request_rate_rps"].tolist()
	predictions = walk_forward_predictions(rates, window_size)
	if len(predictions) < 2:
		raise ValueError(
			"Not enough data for a useful evaluation: provide at least "
			f"window_size + 2 rows (window_size={window_size}, "
			f"available_rows={len(workload)})."
		)

	target_rows = workload.iloc[window_size:].reset_index(drop=True)
	if len(target_rows) != len(predictions):
		raise RuntimeError(
			"Internal alignment error: prediction and target row counts differ."
		)

	prediction_rows = pd.DataFrame(
		{
			"timestamp": target_rows["timestamp"],
			"actual_request_rate_rps": target_rows["request_rate_rps"].astype(float),
			"predicted_request_rate_rps": predictions,
			"forecast_method": FORECAST_METHOD,
			"window_size": window_size,
		}
	)
	metric_values = evaluate_predictions(
		prediction_rows["actual_request_rate_rps"].tolist(), predictions
	)

	for output_path in (prediction_path, metrics_path, graph_path):
		output_path.parent.mkdir(parents=True, exist_ok=True)
	prediction_rows.to_csv(prediction_path, index=False)

	metrics: Dict[str, Any] = {
		"mae_rps": metric_values["mae_rps"],
		"rmse_rps": metric_values["rmse_rps"],
		"number_of_evaluated_predictions": metric_values["n_predictions"],
		"forecast_method": FORECAST_METHOD,
		"window_size": window_size,
		"input_data_path": str(input_path.resolve()),
		"run_metadata": {
			"generated_at_utc": datetime.now(timezone.utc).isoformat(),
			"data_label": data_label,
			"evaluation_strategy": "walk-forward; each target uses earlier observations only",
			"timezone_handling": "preserved from input; no timezone is assumed",
		},
		"output_files": {
			"predictions_csv": str(prediction_path.resolve()),
			"metrics_json": str(metrics_path.resolve()),
			"actual_vs_predicted_png": str(graph_path.resolve()),
		},
	}
	metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
	_save_actual_vs_predicted_graph(prediction_rows, graph_path)
	return metrics


def build_argument_parser() -> argparse.ArgumentParser:
	"""Create the command-line argument parser."""
	parser = argparse.ArgumentParser(
		description="Run CloudPulse's walk-forward moving-average baseline."
	)
	parser.add_argument(
		"--input-csv",
		required=True,
		type=Path,
		help="Workload CSV containing timestamp and request_rate_rps columns.",
	)
	parser.add_argument(
		"--window-size",
		default=2,
		type=int,
		help="Number of earlier observations per moving average (default: 2).",
	)
	parser.add_argument(
		"--predictions-output",
		type=Path,
		default=DEFAULT_PREDICTIONS_PATH,
		help=f"Predictions CSV (default: {DEFAULT_PREDICTIONS_PATH}).",
	)
	parser.add_argument(
		"--metrics-output",
		type=Path,
		default=DEFAULT_METRICS_PATH,
		help=f"Metrics JSON (default: {DEFAULT_METRICS_PATH}).",
	)
	parser.add_argument(
		"--graph-output",
		type=Path,
		default=DEFAULT_GRAPH_PATH,
		help=f"Actual-versus-predicted PNG (default: {DEFAULT_GRAPH_PATH}).",
	)
	parser.add_argument(
		"--data-label",
		default="unspecified",
		help="Optional provenance label recorded in the metrics JSON.",
	)
	return parser


def main(argv: Optional[List[str]] = None) -> int:
	"""Run the baseline command and print output locations and metric values."""
	parser = build_argument_parser()
	arguments = parser.parse_args(argv)

	try:
		metrics = run_baseline(
			input_csv=arguments.input_csv,
			window_size=arguments.window_size,
			predictions_output=arguments.predictions_output,
			metrics_output=arguments.metrics_output,
			graph_output=arguments.graph_output,
			data_label=arguments.data_label,
		)
	except (FileNotFoundError, OSError, ValueError) as error:
		parser.error(str(error))

	print(f"Forecast method: {metrics['forecast_method']}")
	print(f"Evaluated predictions: {metrics['number_of_evaluated_predictions']}")
	print(f"MAE: {metrics['mae_rps']:.4f} requests per second")
	print(f"RMSE: {metrics['rmse_rps']:.4f} requests per second")
	print(f"Predictions CSV: {arguments.predictions_output}")
	print(f"Metrics JSON: {arguments.metrics_output}")
	print(f"Graph PNG: {arguments.graph_output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
