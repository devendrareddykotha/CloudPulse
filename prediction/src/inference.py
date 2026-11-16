"""Offline one-step inference and pure controller-decision integration."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from numbers import Real
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from prediction.src.controller import PredictiveScalingController
from prediction.src.safety_rules import SafetyDecision
from prediction.src.train_model import (
	DEFAULT_LAGS,
	DEFAULT_ROLLING_WINDOWS,
	build_next_step_feature_row,
)


EXPECTED_FEATURE_NAMES = tuple(
	[f"lag_{lag}_rps" for lag in DEFAULT_LAGS]
	+ [f"rolling_mean_{window}_rps" for window in DEFAULT_ROLLING_WINDOWS]
)


def load_saved_predictor(model_path: Union[str, Path]) -> Any:
	"""Load a caller-selected joblib estimator with a verified feature contract.

	The model file must be trusted because joblib uses Python deserialization.
	Artifacts without stored feature names are rejected: a feature count alone
	cannot prove that the estimator learned the expected column order.
	"""
	path = Path(model_path).expanduser()
	if not path.is_file():
		raise FileNotFoundError(f"Saved model file does not exist: {path}")

	try:
		import joblib

		estimator = joblib.load(path)
	except Exception as error:
		raise ValueError(f"Could not load saved model from {path}: {error}") from error

	if not callable(getattr(estimator, "predict", None)):
		raise TypeError(f"Loaded object from {path} does not expose a callable predict()")

	try:
		feature_names = getattr(estimator, "feature_names_in_", None)
		feature_count = getattr(estimator, "n_features_in_", None)
		booster_factory = getattr(estimator, "get_booster", None)
		booster = booster_factory() if callable(booster_factory) else None
		if feature_names is None and booster is not None:
			feature_names = getattr(booster, "feature_names", None)
		if feature_count is None and booster is not None:
			num_features = getattr(booster, "num_features", None)
			feature_count = num_features() if callable(num_features) else None
	except Exception as error:
		raise ValueError(f"Could not inspect saved model feature metadata: {error}") from error

	if feature_names is None:
		raise ValueError(
			"Saved model does not expose feature names, so its feature order cannot "
			"be verified. Supply a model with stored feature-name metadata or add "
			"an explicit, independently verified feature-order attestation."
		)
	try:
		stored_names = tuple(str(name) for name in feature_names)
	except TypeError as error:
		raise ValueError("Saved model feature names are not a readable sequence") from error
	if stored_names != EXPECTED_FEATURE_NAMES:
		raise ValueError(
			"Saved model feature names do not match the inference order: "
			f"expected {EXPECTED_FEATURE_NAMES}, found {stored_names}"
		)

	if feature_count is not None:
		if isinstance(feature_count, bool):
			raise ValueError("Saved model feature count metadata is invalid")
		try:
			count = int(feature_count)
		except (TypeError, ValueError, OverflowError) as error:
			raise ValueError("Saved model feature count metadata is invalid") from error
		if count != feature_count:
			raise ValueError("Saved model feature count metadata is invalid")
		if count != len(EXPECTED_FEATURE_NAMES):
			raise ValueError(
				f"Saved model expects {count} features; "
				f"inference supplies {len(EXPECTED_FEATURE_NAMES)}"
			)
	elif len(stored_names) != len(EXPECTED_FEATURE_NAMES):
		raise ValueError(
			f"Saved model exposes {len(stored_names)} feature names; "
			f"inference supplies {len(EXPECTED_FEATURE_NAMES)}"
		)

	return estimator


@dataclass(frozen=True)
class ForecastResult:
	"""One validated numeric forecast with separate generation and target times."""

	predicted_value: float
	generated_at: datetime
	target_at: datetime
	interpretation: str


@dataclass(frozen=True)
class OfflineDecisionResult:
	"""Offline forecast and scaling decision; never performs Kubernetes I/O."""

	status: str
	forecast: Optional[ForecastResult]
	decision: Optional[SafetyDecision]
	actionable: bool
	message: str


def _aware_utc(value: Any, name: str) -> datetime:
	try:
		timestamp = pd.Timestamp(value)
	except (TypeError, ValueError, OverflowError) as error:
		raise ValueError(f"{name} must be a valid timezone-aware timestamp") from error
	if pd.isna(timestamp) or timestamp.tzinfo is None or timestamp.utcoffset() is None:
		raise ValueError(f"{name} must be timezone-aware; naive timestamps are not assumed to be UTC")
	return timestamp.tz_convert("UTC").to_pydatetime()


def _single_numeric_prediction(output: Any) -> float:
	"""Accept one real numeric result and reject malformed or multiple outputs."""
	if output is None or isinstance(output, (bool, str, bytes)):
		raise ValueError("Predictor must return exactly one numeric prediction")

	if hasattr(output, "size") and hasattr(output, "item"):
		if output.size != 1:
			raise ValueError("Predictor must return exactly one prediction")
		output = output.item()
	elif isinstance(output, (list, tuple)):
		if len(output) != 1:
			raise ValueError("Predictor must return exactly one prediction")
		output = output[0]

	if isinstance(output, bool) or not isinstance(output, Real):
		raise ValueError("Prediction must be a real numeric value")
	prediction = float(output)
	if not math.isfinite(prediction) or prediction < 0:
		raise ValueError("Prediction must be finite and nonnegative")
	return prediction


def forecast_next_step(
	workload: pd.DataFrame,
	predictor: Any,
	*,
	sampling_interval: Union[str, pd.Timedelta],
	generated_at: Optional[datetime] = None,
	interpretation: str = "Prototype forecast; source units and provenance unverified",
) -> ForecastResult:
	"""Predict the next step from history using the training feature order.

	The sampling interval is explicit. Historical timestamps must be aware,
	strictly increasing, and regularly spaced at that interval. ``generated_at``
	means when this forecast was produced; ``target_at`` is the next sample time.
	"""
	try:
		interval = pd.Timedelta(sampling_interval)
	except (TypeError, ValueError, OverflowError) as error:
		raise ValueError("sampling_interval must be a valid positive duration") from error
	if pd.isna(interval) or interval <= pd.Timedelta(0):
		raise ValueError("sampling_interval must be a valid positive duration")
	if not isinstance(interpretation, str) or not interpretation.strip():
		raise ValueError("interpretation must be a nonempty label")

	if "timestamp" not in workload.columns:
		raise ValueError("Workload must contain a timestamp column")
	history_times = [_aware_utc(value, "historical timestamp") for value in workload["timestamp"]]
	if len(history_times) < 2:
		raise ValueError("At least two timestamped history rows are required")
	for previous, current in zip(history_times, history_times[1:]):
		if current <= previous:
			raise ValueError("Historical timestamps must be unique and chronological")
		if current - previous != interval.to_pytimedelta():
			raise ValueError("Historical timestamps do not match sampling_interval")

	generation_time = _aware_utc(
		datetime.now(timezone.utc) if generated_at is None else generated_at,
		"generated_at",
	)
	target_time = _aware_utc(
		history_times[-1] + interval.to_pytimedelta(), "target_at"
	)
	if generation_time >= target_time:
		raise ValueError("generated_at must be earlier than target_at")
	feature_row, _ = build_next_step_feature_row(workload, target_time)
	output = predictor.predict(feature_row)
	prediction = _single_numeric_prediction(output)
	return ForecastResult(
		predicted_value=prediction,
		generated_at=generation_time,
		target_at=target_time,
		interpretation=interpretation,
	)


def forecast_and_decide(
	workload: pd.DataFrame,
	predictor: Any,
	controller: PredictiveScalingController,
	current_replicas: int,
	*,
	sampling_interval: Union[str, pd.Timedelta],
	generated_at: Optional[datetime] = None,
	interpretation: str = "Prototype forecast; source units and provenance unverified",
) -> OfflineDecisionResult:
	"""Run inference and the pure scaling decision without API-client calls."""
	try:
		forecast = forecast_next_step(
			workload,
			predictor,
			sampling_interval=sampling_interval,
			generated_at=generated_at,
			interpretation=interpretation,
		)
	except Exception as error:
		return OfflineDecisionResult(
			status="error",
			forecast=None,
			decision=None,
			actionable=False,
			message=f"Forecast unavailable; no scaling proposal created: {error}",
		)

	try:
		decision = controller.decide(
			forecast.predicted_value,
			current_replicas,
			forecast.generated_at,
		)
	except Exception as error:
		return OfflineDecisionResult(
			status="error",
			forecast=forecast,
			decision=None,
			actionable=False,
			message=f"Scaling decision failed; no proposal created: {error}",
		)

	actionable = decision.allowed and decision.target_replicas != current_replicas
	return OfflineDecisionResult(
		status="proposal" if actionable else "hold",
		forecast=forecast,
		decision=decision,
		actionable=actionable,
		message=decision.reason,
	)
