"""Compare moving-average windows with the existing walk-forward evaluator."""

import argparse
from numbers import Integral
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd

from prediction.src.moving_average import evaluate_walk_forward
from prediction.src.preprocessing import load_workload_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = (
	PROJECT_ROOT / "results" / "prediction" / "moving_average_window_comparison.csv"
)
DEFAULT_WINDOW_SIZES = (2, 3, 4)
FORECAST_METHOD = "moving_average_walk_forward"


def _validated_window_sizes(window_sizes: Iterable[int]) -> List[int]:
	"""Check that requested window sizes are unique positive integers."""
	if isinstance(window_sizes, (str, bytes)):
		raise ValueError("window_sizes must be a non-empty sequence of positive integers")
	try:
		requested_sizes = list(window_sizes)
	except TypeError as error:
		raise ValueError(
			"window_sizes must be a non-empty sequence of positive integers"
		) from error

	if not requested_sizes:
		raise ValueError("At least one window size must be provided")
	for window_size in requested_sizes:
		if (
			isinstance(window_size, bool)
			or not isinstance(window_size, Integral)
			or window_size <= 0
		):
			raise ValueError("Every window size must be a positive integer")
	if len(set(requested_sizes)) != len(requested_sizes):
		raise ValueError("Window sizes must not contain duplicates")
	return [int(window_size) for window_size in requested_sizes]


def compare_window_sizes(
	input_csv: Union[str, Path],
	window_sizes: Iterable[int] = DEFAULT_WINDOW_SIZES,
	output_csv: Optional[Union[str, Path]] = None,
	dataset_label: Optional[str] = None,
) -> pd.DataFrame:
	"""Evaluate requested windows and save evaluated or explicitly skipped rows.

	Only insufficient evaluation history is recorded as a skip. Invalid workload
	data and other evaluation failures raise errors instead of disappearing from
	the comparison. A supported window is scored by ``evaluate_walk_forward``.
	"""
	requested_sizes = _validated_window_sizes(window_sizes)
	input_path = Path(input_csv).expanduser()
	output_path = Path(output_csv) if output_csv is not None else DEFAULT_OUTPUT_PATH
	if input_path.resolve() == output_path.resolve():
		raise ValueError("The comparison output path must not overwrite the input CSV")
	workload = load_workload_csv(input_path)
	quality_report = workload.attrs["validation_report"]

	if quality_report["missing_timestamps"] or quality_report["invalid_timestamps"]:
		raise ValueError("Window comparison requires a valid timestamp on every row")
	if quality_report["duplicate_timestamps"]:
		raise ValueError("Window comparison requires unique timestamps")
	if (
		quality_report["missing_request_rates"]
		or quality_report["non_numeric_request_rates"]
		or quality_report["negative_request_rates"]
	):
		raise ValueError("Window comparison requires valid nonnegative request rates")

	rates = workload["request_rate_rps"].tolist()
	records = []
	for window_size in requested_sizes:
		base_record = {
			"window_size": window_size,
			"forecast_method": FORECAST_METHOD,
			"mae_rps": None,
			"rmse_rps": None,
			"number_of_evaluated_predictions": 0,
			"input_data_path": str(input_path.resolve()),
			"dataset_label": dataset_label or input_path.stem,
			"evaluation_status": "evaluated",
			"skip_reason": "",
		}
		minimum_rows = window_size + 2
		if len(rates) < minimum_rows:
			base_record["evaluation_status"] = "skipped"
			base_record["skip_reason"] = (
				f"Needs at least {minimum_rows} rows to produce two evaluated targets; "
				f"dataset has {len(rates)}."
			)
			records.append(base_record)
			continue

		metrics = evaluate_walk_forward(rates, window_size)
		base_record["mae_rps"] = metrics["mae_rps"]
		base_record["rmse_rps"] = metrics["rmse_rps"]
		base_record["number_of_evaluated_predictions"] = metrics["n_predictions"]
		records.append(base_record)

	comparison = pd.DataFrame.from_records(records)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	comparison.to_csv(output_path, index=False)
	return comparison


def build_argument_parser() -> argparse.ArgumentParser:
	"""Create the comparison command-line parser."""
	parser = argparse.ArgumentParser(
		description="Compare moving-average windows with walk-forward evaluation."
	)
	parser.add_argument(
		"--input-csv",
		required=True,
		type=Path,
		help="Workload CSV with timestamp and request_rate_rps columns.",
	)
	parser.add_argument(
		"--window-sizes",
		nargs="+",
		type=int,
		default=list(DEFAULT_WINDOW_SIZES),
		help="Positive window sizes to compare (default: 2 3 4).",
	)
	parser.add_argument(
		"--output-csv",
		type=Path,
		default=DEFAULT_OUTPUT_PATH,
		help=f"Comparison CSV output (default: {DEFAULT_OUTPUT_PATH}).",
	)
	parser.add_argument(
		"--dataset-label",
		help="Optional dataset label stored in every result row.",
	)
	return parser


def main(argv: Optional[List[str]] = None) -> int:
	"""Run the comparison and print each window's outcome."""
	parser = build_argument_parser()
	arguments = parser.parse_args(argv)
	try:
		comparison = compare_window_sizes(
			input_csv=arguments.input_csv,
			window_sizes=arguments.window_sizes,
			output_csv=arguments.output_csv,
			dataset_label=arguments.dataset_label,
		)
	except (FileNotFoundError, OSError, ValueError) as error:
		parser.error(str(error))

	for record in comparison.to_dict(orient="records"):
		if record["evaluation_status"] == "evaluated":
			print(
				f"Window {record['window_size']}: evaluated "
				f"({record['number_of_evaluated_predictions']} predictions), "
				f"MAE={record['mae_rps']:.4f} RPS, "
				f"RMSE={record['rmse_rps']:.4f} RPS"
			)
		else:
			print(
				f"Window {record['window_size']}: skipped - "
				f"{record['skip_reason']}"
			)
	print(f"Comparison CSV: {arguments.output_csv}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())