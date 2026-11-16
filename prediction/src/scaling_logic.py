"""Convert a predicted request rate into a bounded replica target."""

import math

from prediction.src.config import ScalingConfig


def calculate_desired_replicas(
	predicted_request_rate_rps: float, config: ScalingConfig
) -> int:
	"""Calculate ceil(predicted RPS / assumed capacity), clamped to limits."""
	if isinstance(predicted_request_rate_rps, bool):
		raise ValueError("predicted_request_rate_rps must be finite and nonnegative")
	try:
		predicted_rate = float(predicted_request_rate_rps)
	except (TypeError, ValueError, OverflowError) as error:
		raise ValueError(
			"predicted_request_rate_rps must be finite and nonnegative"
		) from error
	if not math.isfinite(predicted_rate) or predicted_rate < 0:
		raise ValueError("predicted_request_rate_rps must be finite and nonnegative")

	requested = math.ceil(predicted_rate / config.capacity_rps_per_replica)
	return min(config.max_replicas, max(config.min_replicas, requested))
