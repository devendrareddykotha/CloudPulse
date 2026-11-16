"""Train and evaluate XGBoost on an explicitly labelled workload series."""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import matplotlib
import pandas as pd
from xgboost import XGBRegressor

from prediction.src.moving_average import evaluate_predictions, walk_forward_predictions
from prediction.src.preprocessing import load_workload_csv


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "prediction" / "models" / "xgboost_service_proxy.joblib"
DEFAULT_PREDICTIONS_PATH = (
	PROJECT_ROOT
	/ "results"
	/ "prediction"
	/ "xgboost_service_proxy_predictions.csv"
)
DEFAULT_METRICS_PATH = (
	PROJECT_ROOT / "results" / "prediction" / "xgboost_service_proxy_metrics.json"
)
DEFAULT_GRAPH_PATH = (
	PROJECT_ROOT / "results" / "graphs" / "xgboost_service_proxy_evaluation.png"
)
DEFAULT_LAGS = (1, 2, 3)
DEFAULT_ROLLING_WINDOWS = (3,)
FORECAST_METHOD = "xgboost_lagged_walk_forward_features"
BASELINE_METHOD = "moving_average_walk_forward_window_3"


def build_lagged_features(
	workload: pd.DataFrame,
	lags: Sequence[int] = DEFAULT_LAGS,
	rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
) -> Tuple[pd.DataFrame, List[str], int]:
	"""Build target rows with lagged and prior-only rolling features.

	The initial rows without enough past observations are excluded as model
	feature warm-up, not imputed. For each target at row ``t``, all features use
	values no later than ``t - 1``.
	"""
	if "timestamp" not in workload.columns or "request_rate_rps" not in workload.columns:
		raise ValueError("Input must contain timestamp and request_rate_rps columns")
	if not lags and not rolling_windows:
		raise ValueError("At least one lag or rolling window is required")
	for name, windows in (("lags", lags), ("rolling_windows", rolling_windows)):
		if len(set(windows)) != len(windows):
			raise ValueError(f"{name} must not contain duplicates")
		for window in windows:
			if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
				raise ValueError(f"Every {name} value must be a positive integer")

	frame = workload[["timestamp", "request_rate_rps"]].copy().reset_index(drop=True)
	if frame["timestamp"].isna().any():
		raise ValueError("Workload contains missing or invalid timestamps")
	rates = pd.to_numeric(frame["request_rate_rps"], errors="coerce")
	if rates.isna().any() or not rates.map(math.isfinite).all():
		raise ValueError("Request rates must be present, numeric, and finite")
	if (rates < 0).any():
		raise ValueError("Request rates must not be negative")

	features = pd.DataFrame(
		{
			"source_row": range(len(frame)),
			"timestamp": frame["timestamp"],
			"actual_request_rate_rps": rates,
		}
	)
	feature_columns = _add_lagged_feature_columns(
		features, rates, lags=lags, rolling_windows=rolling_windows
	)

	warmup_rows = max([*lags, *rolling_windows], default=0)
	features = features.dropna(subset=feature_columns).reset_index(drop=True)
	return features, feature_columns, warmup_rows


def _add_lagged_feature_columns(
	features: pd.DataFrame,
	rates: pd.Series,
	*,
	lags: Sequence[int],
	rolling_windows: Sequence[int],
) -> List[str]:
	"""Add the shared, prior-only feature columns in model input order."""
	feature_columns = []
	for lag in lags:
		column = f"lag_{lag}_rps"
		features[column] = rates.shift(lag)
		feature_columns.append(column)
	prior_rates = rates.shift(1)
	for window in rolling_windows:
		column = f"rolling_mean_{window}_rps"
		features[column] = prior_rates.rolling(window=window, min_periods=window).mean()
		feature_columns.append(column)
	return feature_columns


def build_next_step_feature_row(
	workload: pd.DataFrame, target_timestamp: object
) -> Tuple[pd.DataFrame, List[str]]:
	"""Build the next target's feature row using the training feature logic.

	The future target value is left missing and is never used as a feature or
	passed to the model. At least ``warmup_rows`` historical samples are needed.
	"""
	_, feature_columns, warmup_rows = build_lagged_features(workload)
	if len(workload) < warmup_rows:
		raise ValueError(
			f"Insufficient history for next-step features: need {warmup_rows} rows, "
			f"received {len(workload)}"
		)

	frame = workload[["timestamp", "request_rate_rps"]].copy().reset_index(drop=True)
	rates = pd.to_numeric(frame["request_rate_rps"], errors="coerce")
	future_rates = pd.concat(
		[rates, pd.Series([float("nan")])], ignore_index=True
	)
	features = pd.DataFrame(
		{
			"source_row": range(len(frame) + 1),
			"timestamp": pd.concat(
				[frame["timestamp"], pd.Series([target_timestamp])], ignore_index=True
			),
			"actual_request_rate_rps": future_rates,
		}
	)
	generated_columns = _add_lagged_feature_columns(
		features,
		future_rates,
		lags=DEFAULT_LAGS,
		rolling_windows=DEFAULT_ROLLING_WINDOWS,
	)
	if generated_columns != feature_columns:
		raise RuntimeError("Next-step feature order does not match the training model")
	return features.iloc[[-1]][feature_columns].reset_index(drop=True), feature_columns


def _metric_pair(actual: pd.Series, predicted: Sequence[float]) -> Dict[str, Union[float, int]]:
	return evaluate_predictions(actual.tolist(), predicted)


def _check_output_paths(
	input_path: Path, output_paths: Sequence[Path], overwrite: bool
) -> None:
	resolved_outputs = [path.resolve() for path in output_paths]
	if len(set(resolved_outputs)) != len(resolved_outputs):
		raise ValueError("Model, predictions, metrics, and graph output paths must be distinct")
	if input_path.resolve() in resolved_outputs:
		raise ValueError("An output path must not overwrite the input workload CSV")
	if not overwrite:
		existing = [str(path) for path in resolved_outputs if path.exists()]
		if existing:
			raise FileExistsError(
				"Refusing to overwrite existing output(s): " + ", ".join(existing)
			)


def _save_test_graph(predictions: pd.DataFrame, graph_path: Path) -> None:
	"""Save test-set actuals and forecasts for both methods."""
	figure, axis = plt.subplots(figsize=(11, 5))
	axis.plot(
		predictions["timestamp"],
		predictions["actual_request_rate_rps"],
		marker="o",
		label="Actual selected-service proxy",
	)
	axis.plot(
		predictions["timestamp"],
		predictions["xgboost_prediction_rps"],
		marker="o",
		label="XGBoost",
	)
	axis.plot(
		predictions["timestamp"],
		predictions["moving_average_prediction_rps"],
		marker="o",
		label="Moving average (window 3)",
	)
	axis.set_title("Experimental selected-service MCR proxy: test predictions")
	axis.set_xlabel("Timestamp offset represented from Unix epoch (UTC)")
	axis.set_ylabel("MCR (calls per second; selected-service proxy)")
	axis.legend()
	axis.grid(True, alpha=0.3)
	figure.autofmt_xdate()
	figure.tight_layout()
	figure.savefig(graph_path, dpi=150)
	plt.close(figure)


def train_and_evaluate(
	input_csv: Union[str, Path],
	model_output: Union[str, Path] = DEFAULT_MODEL_PATH,
	predictions_output: Union[str, Path] = DEFAULT_PREDICTIONS_PATH,
	metrics_output: Union[str, Path] = DEFAULT_METRICS_PATH,
	graph_output: Union[str, Path] = DEFAULT_GRAPH_PATH,
	dataset_label: str = "unspecified",
	expected_sampling_interval: Optional[Union[str, pd.Timedelta]] = None,
	train_fraction: float = 0.6,
	validation_fraction: float = 0.2,
	baseline_window_size: int = 3,
	overwrite: bool = False,
) -> Dict[str, Any]:
	"""Fit XGBoost on chronological history and compare it with moving average.

	The generated scores apply only to the input series. For the Alibaba proxy,
	they must not be interpreted as cluster-wide workload performance.
	"""
	input_path = Path(input_csv).expanduser().resolve()
	model_path = Path(model_output).expanduser().resolve()
	prediction_path = Path(predictions_output).expanduser().resolve()
	metrics_path = Path(metrics_output).expanduser().resolve()
	graph_path = Path(graph_output).expanduser().resolve()
	_check_output_paths(
		input_path, [model_path, prediction_path, metrics_path, graph_path], overwrite
	)
	if not (0 < train_fraction < 1):
		raise ValueError("train_fraction must be between zero and one")
	if not (0 < validation_fraction < 1 - train_fraction):
		raise ValueError("validation_fraction must leave a non-empty test split")
	if isinstance(baseline_window_size, bool) or baseline_window_size <= 0:
		raise ValueError("baseline_window_size must be a positive integer")

	workload = load_workload_csv(
		input_path, expected_sampling_interval=expected_sampling_interval
	)
	quality = workload.attrs["validation_report"]
	if (
		quality["missing_timestamps"]
		or quality["invalid_timestamps"]
		or quality["duplicate_timestamps"]
	):
		raise ValueError("Training requires valid, unique timestamps on every row")
	if (
		quality["missing_request_rates"]
		or quality["non_numeric_request_rates"]
		or quality["negative_request_rates"]
	):
		raise ValueError("Training requires valid nonnegative request rates on every row")
	if expected_sampling_interval is not None and quality["unexpected_sampling_gaps"]:
		raise ValueError(
			f"Training input has {quality['unexpected_sampling_gaps']} unexpected sampling gaps"
		)

	features, feature_columns, warmup_rows = build_lagged_features(workload)
	feature_count = len(features)
	train_end = int(feature_count * train_fraction)
	validation_end = int(feature_count * (train_fraction + validation_fraction))
	if train_end < 10 or validation_end - train_end < 2 or feature_count - validation_end < 2:
		raise ValueError(
		"Insufficient observations for chronological train/validation/test splits; "
		f"usable_rows={feature_count}, train={train_end}, "
		f"validation={validation_end - train_end}, test={feature_count - validation_end}"
		)

	train_rows = features.iloc[:train_end]
	validation_rows = features.iloc[train_end:validation_end]
	test_rows = features.iloc[validation_end:]
	model_parameters = {
		"n_estimators": 100,
		"max_depth": 2,
		"learning_rate": 0.05,
		"subsample": 0.9,
		"colsample_bytree": 0.9,
		"objective": "reg:squarederror",
		"random_state": 42,
		"n_jobs": 1,
	}
	model = XGBRegressor(**model_parameters)
	model.fit(
		train_rows[feature_columns],
		train_rows["actual_request_rate_rps"],
		eval_set=[
			(
				validation_rows[feature_columns],
				validation_rows["actual_request_rate_rps"],
			)
		],
		verbose=False,
	)
	validation_predictions = model.predict(validation_rows[feature_columns])
	test_predictions = model.predict(test_rows[feature_columns])

	rates = workload["request_rate_rps"].tolist()
	baseline_predictions = walk_forward_predictions(rates, baseline_window_size)
	baseline_by_source_row = {
		index + baseline_window_size: value
		for index, value in enumerate(baseline_predictions)
	}

	validation_report = {
		"xgboost": _metric_pair(
			validation_rows["actual_request_rate_rps"], validation_predictions
		),
		"moving_average": _metric_pair(
			validation_rows["actual_request_rate_rps"],
			[baseline_by_source_row[int(index)] for index in validation_rows["source_row"]],
		),
	}
	test_report = {
		"xgboost": _metric_pair(test_rows["actual_request_rate_rps"], test_predictions),
		"moving_average": _metric_pair(
			test_rows["actual_request_rate_rps"],
			[baseline_by_source_row[int(index)] for index in test_rows["source_row"]],
		),
	}

	prediction_frames = []
	for split_name, rows, xgb_values in (
		("validation", validation_rows, validation_predictions),
		("test", test_rows, test_predictions),
	):
		output_rows = rows[["source_row", "timestamp", "actual_request_rate_rps"]].copy()
		output_rows["split"] = split_name
		output_rows["xgboost_prediction_rps"] = xgb_values
		output_rows["moving_average_prediction_rps"] = [
			baseline_by_source_row[int(index)] for index in rows["source_row"]
		]
		prediction_frames.append(output_rows)
	prediction_output = pd.concat(prediction_frames, ignore_index=True)
	test_output = prediction_output.loc[prediction_output["split"] == "test"]

	metadata: Dict[str, Any] = {
		"generated_at_utc": datetime.now(timezone.utc).isoformat(),
		"dataset_label": dataset_label,
		"data_interpretation": (
			"Experimental single-service consumerMQ_MCR proxy; not validated "
			"cluster-wide incoming demand."
		),
		"timestamp_interpretation": (
			"timestamp_ms offsets represented from Unix epoch UTC for compatibility; "
			"the actual wall-clock start date is unknown."
		),
		"evaluation_strategy": "chronological 60/20/20 split; no random shuffling",
		"feature_policy": "lags and rolling means use prior observations only",
		"feature_warmup_rows_excluded": warmup_rows,
		"feature_columns": feature_columns,
		"sampling_interval_argument": (
			str(expected_sampling_interval)
			if expected_sampling_interval is not None
			else None
		),
		"split_rows": {
			"train": len(train_rows),
			"validation": len(validation_rows),
			"test": len(test_rows),
		},
		"split_timestamps": {
			"train_start": str(train_rows["timestamp"].iloc[0]),
			"train_end": str(train_rows["timestamp"].iloc[-1]),
			"validation_start": str(validation_rows["timestamp"].iloc[0]),
			"validation_end": str(validation_rows["timestamp"].iloc[-1]),
			"test_start": str(test_rows["timestamp"].iloc[0]),
			"test_end": str(test_rows["timestamp"].iloc[-1]),
		},
		"input_data_path": str(input_path),
		"xgboost_parameters": model_parameters,
		"limitations": [
			"Only 90 one-minute observations are available for this experiment.",
			"The selected per-service proxy is not validated as cluster-wide demand.",
			"Metrics are exploratory and do not establish production accuracy.",
		],
	}
	metrics: Dict[str, Any] = {
		"forecast_method": FORECAST_METHOD,
		"baseline_method": BASELINE_METHOD,
		"window_size": baseline_window_size,
		"validation": validation_report,
		"test": test_report,
		"metadata": metadata,
	}

	for output_path in (model_path, prediction_path, metrics_path, graph_path):
		output_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, model_path)
	prediction_output.to_csv(prediction_path, index=False, float_format="%.17g")
	metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
	_save_test_graph(test_output, graph_path)
	return metrics


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Train an exploratory XGBoost model on a chronological workload series."
	)
	parser.add_argument("--input-csv", type=Path, required=True)
	parser.add_argument("--dataset-label", default="unspecified")
	parser.add_argument("--expected-sampling-interval")
	parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
	parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_PATH)
	parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
	parser.add_argument("--graph-output", type=Path, default=DEFAULT_GRAPH_PATH)
	parser.add_argument("--overwrite", action="store_true")
	return parser


def main(argv: Optional[List[str]] = None) -> int:
	parser = build_argument_parser()
	arguments = parser.parse_args(argv)
	try:
		metrics = train_and_evaluate(
			input_csv=arguments.input_csv,
			dataset_label=arguments.dataset_label,
			expected_sampling_interval=arguments.expected_sampling_interval,
			model_output=arguments.model_output,
			predictions_output=arguments.predictions_output,
			metrics_output=arguments.metrics_output,
			graph_output=arguments.graph_output,
			overwrite=arguments.overwrite,
		)
	except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
		parser.error(str(error))

	for split_name in ("validation", "test"):
		for method_name, result in metrics[split_name].items():
			print(
				f"{split_name.capitalize()} {method_name}: "
				f"MAE={result['mae_rps']:.4f} RPS, "
				f"RMSE={result['rmse_rps']:.4f} RPS, "
				f"n={result['n_predictions']}"
			)
	print(f"Model: {arguments.model_output}")
	print(f"Predictions: {arguments.predictions_output}")
	print(f"Metrics: {arguments.metrics_output}")
	print(f"Graph: {arguments.graph_output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
