"""Configuration for Member 1 predictive scaling decisions."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class ScalingConfig:
	"""Safety limits and an explicitly supplied capacity assumption.

	``capacity_rps_per_replica`` must be measured or deliberately selected by the
	project team. It has no default because an invented capacity would make
	replica recommendations appear more certain than they are.
	"""

	capacity_rps_per_replica: float
	capacity_basis: str = "UNVERIFIED ASSUMPTION"
	min_replicas: int = 1
	max_replicas: int = 10
	cooldown_seconds: int = 300
	max_scale_up_step: int = 2
	max_scale_down_step: int = 1
	max_forecast_age_seconds: int = 120
	dry_run: bool = True

	def __post_init__(self) -> None:
		capacity = self.capacity_rps_per_replica
		if isinstance(capacity, bool) or not isinstance(capacity, (int, float)):
			raise ValueError("capacity_rps_per_replica must be a positive finite number")
		if not math.isfinite(float(capacity)) or capacity <= 0:
			raise ValueError("capacity_rps_per_replica must be a positive finite number")
		if not isinstance(self.capacity_basis, str) or not self.capacity_basis.strip():
			raise ValueError("capacity_basis must explain whether capacity is measured or assumed")
		for name in (
			"min_replicas",
			"max_replicas",
			"cooldown_seconds",
			"max_scale_up_step",
			"max_scale_down_step",
			"max_forecast_age_seconds",
		):
			value = getattr(self, name)
			if isinstance(value, bool) or not isinstance(value, int):
				raise ValueError(f"{name} must be an integer")
		if self.min_replicas < 1:
			raise ValueError("min_replicas must be at least one")
		if self.max_replicas < self.min_replicas:
			raise ValueError("max_replicas must be at least min_replicas")
		if self.cooldown_seconds < 0 or self.max_forecast_age_seconds < 0:
			raise ValueError("cooldown and forecast age limits must not be negative")
		if self.max_scale_up_step < 1 or self.max_scale_down_step < 1:
			raise ValueError("scale step limits must be at least one replica")
		if not isinstance(self.dry_run, bool):
			raise ValueError("dry_run must be a boolean")


def load_scaling_config(config_path: Union[str, Path]) -> ScalingConfig:
	"""Load explicit scaling assumptions from a JSON file."""
	path = Path(config_path).expanduser()
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except json.JSONDecodeError as error:
		raise ValueError(f"Invalid scaling configuration JSON: {error}") from error
	if not isinstance(payload, dict):
		raise ValueError("Scaling configuration must be a JSON object")
	allowed = set(ScalingConfig.__dataclass_fields__)
	unknown = sorted(set(payload) - allowed)
	if unknown:
		raise ValueError(f"Unknown scaling configuration fields: {unknown}")
	if "capacity_rps_per_replica" not in payload:
		raise ValueError(
			"Configuration must explicitly provide capacity_rps_per_replica; "
			"measure it before using scaling recommendations"
		)
	return ScalingConfig(**payload)
