"""Moving-average forecasting baseline for request-rate workloads."""

import math
from numbers import Integral
from typing import Dict, Iterable, List, Union


MetricReport = Dict[str, Union[float, int]]


def _validate_window_size(window_size: int) -> int:
	"""Return a valid positive integer window size."""
	if isinstance(window_size, bool) or not isinstance(window_size, Integral):
		raise ValueError("window_size must be a positive integer")
	if window_size <= 0:
		raise ValueError("window_size must be a positive integer")
	return int(window_size)


def _validate_request_rates(request_rates: Iterable[object]) -> List[float]:
	"""Convert rates to finite floats, rejecting missing or invalid values."""
	if isinstance(request_rates, (str, bytes)):
		raise ValueError("request_rates must be a sequence of numeric values")

	try:
		values = list(request_rates)
	except TypeError as error:
		raise ValueError("request_rates must be an iterable of numeric values") from error

	validated_values = []
	for index, value in enumerate(values):
		if value is None or isinstance(value, bool):
			raise ValueError(f"request_rates[{index}] must be a finite numeric value")
		try:
			numeric_value = float(value)
		except (TypeError, ValueError, OverflowError) as error:
			raise ValueError(
				f"request_rates[{index}] must be a finite numeric value"
			) from error
		if not math.isfinite(numeric_value):
			raise ValueError(f"request_rates[{index}] must be finite")
		if numeric_value < 0:
			raise ValueError(f"request_rates[{index}] must not be negative")
		validated_values.append(numeric_value)

	return validated_values


def moving_average(request_rates: Iterable[object], window_size: int) -> float:
	"""Calculate the mean of the most recent ``window_size`` request rates.

	The input should contain values from the ``request_rate_rps`` target column.
	Values must be finite numbers (or strings convertible to finite numbers).
	No missing values are filled or observations invented.
	"""
	window_size = _validate_window_size(window_size)
	values = _validate_request_rates(request_rates)
	if len(values) < window_size:
		raise ValueError(
			f"Not enough historical observations: window_size={window_size}, "
			f"available={len(values)}"
		)

	return sum(values[-window_size:]) / window_size


def walk_forward_predictions(
	request_rates: Iterable[object], window_size: int
) -> List[float]:
	"""Predict each next rate using only the preceding window of observed rates.

	Returns ``len(request_rates) - window_size`` predictions. The first
	prediction uses the first window; each later prediction can use actual
	observations available before its target, never that target or later values.
	At least one target observation beyond the initial window is required.
	"""
	window_size = _validate_window_size(window_size)
	values = _validate_request_rates(request_rates)
	if len(values) <= window_size:
		raise ValueError(
			"Walk-forward prediction needs at least one observation beyond "
			f"the initial window (window_size={window_size}, available={len(values)})"
		)

	predictions = []
	for target_index in range(window_size, len(values)):
		historical_window = values[target_index - window_size : target_index]
		predictions.append(sum(historical_window) / window_size)
	return predictions


def evaluate_predictions(
	actual_rates: Iterable[object], predicted_rates: Iterable[object]
) -> MetricReport:
	"""Calculate MAE and RMSE in requests per second for aligned values."""
	actual_values = _validate_request_rates(actual_rates)
	predicted_values = _validate_request_rates(predicted_rates)
	if not actual_values:
		raise ValueError("At least one actual/predicted pair is required")
	if len(actual_values) != len(predicted_values):
		raise ValueError(
			"actual_rates and predicted_rates must have the same number of values"
		)

	errors = [actual - predicted for actual, predicted in zip(actual_values, predicted_values)]
	mean_absolute_error = sum(abs(error) for error in errors) / len(errors)
	root_mean_squared_error = math.sqrt(
		sum(error * error for error in errors) / len(errors)
	)
	return {
		"mae_rps": mean_absolute_error,
		"rmse_rps": root_mean_squared_error,
		"n_predictions": len(errors),
	}


def evaluate_walk_forward(
	request_rates: Iterable[object], window_size: int
) -> MetricReport:
	"""Evaluate walk-forward predictions against later observed request rates.

	Requires at least two evaluated targets so a one-point smoke result is not
	presented as a meaningful evaluation. The returned count shows the size of
	the held-out prediction set; it should be considered when interpreting scores.
	"""
	window_size = _validate_window_size(window_size)
	values = _validate_request_rates(request_rates)
	if len(values) < window_size + 2:
		raise ValueError(
			"Walk-forward evaluation needs at least two target observations "
			f"after the initial window (window_size={window_size}, "
			f"available={len(values)})"
		)

	predictions = walk_forward_predictions(values, window_size)
	actual_rates = values[window_size:]
	return evaluate_predictions(actual_rates, predictions)
