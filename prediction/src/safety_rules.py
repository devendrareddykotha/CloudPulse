"""Pure safety checks for proposed replica-count changes."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from prediction.src.config import ScalingConfig


@dataclass(frozen=True)
class SafetyDecision:
	"""Result of evaluating a proposed replica target."""

	requested_replicas: int
	target_replicas: int
	allowed: bool
	reason: str


def _validate_replicas(value: int, name: str) -> None:
	if isinstance(value, bool) or not isinstance(value, int):
		raise ValueError(f"{name} must be an integer")


def _utc_datetime(value: datetime, name: str) -> datetime:
	if not isinstance(value, datetime) or value.tzinfo is None:
		raise ValueError(f"{name} must be a timezone-aware datetime")
	return value.astimezone(timezone.utc)


def apply_safety_rules(
	desired_replicas: int,
	current_replicas: int,
	config: ScalingConfig,
	*,
	forecast_timestamp: Optional[datetime],
	last_scale_time: Optional[datetime] = None,
	now: Optional[datetime] = None,
) -> SafetyDecision:
	"""Reject unsafe/stale proposals and bound allowed changes per decision."""
	_validate_replicas(desired_replicas, "desired_replicas")
	_validate_replicas(current_replicas, "current_replicas")
	if not config.min_replicas <= current_replicas <= config.max_replicas:
		raise ValueError("current_replicas is outside configured replica limits")
	if not config.min_replicas <= desired_replicas <= config.max_replicas:
		raise ValueError("desired_replicas is outside configured replica limits")

	if forecast_timestamp is None:
		return SafetyDecision(
			desired_replicas, current_replicas, False, "Missing forecast timestamp; hold current replicas"
		)
	current_time = _utc_datetime(now or datetime.now(timezone.utc), "now")
	forecast_time = _utc_datetime(forecast_timestamp, "forecast_timestamp")
	if forecast_time > current_time:
		return SafetyDecision(
			desired_replicas, current_replicas, False, "Forecast timestamp is in the future; hold current replicas"
		)
	forecast_age = current_time - forecast_time
	if forecast_age > timedelta(seconds=config.max_forecast_age_seconds):
		return SafetyDecision(
			desired_replicas,
			current_replicas,
			False,
			f"Forecast is stale ({forecast_age.total_seconds():.0f}s old); hold current replicas",
		)

	target = desired_replicas
	if last_scale_time is not None:
		scale_time = _utc_datetime(last_scale_time, "last_scale_time")
		if scale_time > current_time:
			raise ValueError("last_scale_time must not be in the future")
		elapsed = current_time - scale_time
		if desired_replicas != current_replicas and elapsed < timedelta(
			seconds=config.cooldown_seconds
		):
			return SafetyDecision(
				desired_replicas,
				current_replicas,
				False,
				f"Cooldown active ({elapsed.total_seconds():.0f}s elapsed); hold current replicas",
			)

	lower_bound = max(
		config.min_replicas, current_replicas - config.max_scale_down_step
	)
	upper_bound = min(
		config.max_replicas, current_replicas + config.max_scale_up_step
	)
	target = min(upper_bound, max(lower_bound, desired_replicas))
	if desired_replicas == target:
		reason = "Scaling proposal passes safety checks"
	else:
		reason = (
			f"Proposal bounded from {desired_replicas} to {target} replicas "
			f"(up step <= {config.max_scale_up_step}, "
			f"down step <= {config.max_scale_down_step})"
		)
	return SafetyDecision(desired_replicas, target, True, reason)
