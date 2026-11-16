"""Predictive scaling controller with dry-run as the default behavior."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from prediction.src.config import ScalingConfig
from prediction.src.safety_rules import SafetyDecision, apply_safety_rules
from prediction.src.scaling_logic import calculate_desired_replicas


LOGGER = logging.getLogger(__name__)


def create_kubernetes_apps_api(in_cluster: bool = False) -> Any:
	"""Create an AppsV1Api client using local kubeconfig or in-cluster auth.

	This function only creates a client configuration. It does not read or patch
	a Deployment; callers choose when to invoke ``reconcile``.
	"""
	try:
		from kubernetes import client as kube_client
		from kubernetes import config as kube_config
	except ImportError as error:
		raise RuntimeError(
			"Kubernetes Python client is unavailable; install the declared "
			"kubernetes dependency in the project environment"
		) from error

	try:
		if in_cluster:
			kube_config.load_incluster_config()
		else:
			kube_config.load_kube_config()
	except Exception as error:
		mode = "in-cluster credentials" if in_cluster else "local kubeconfig"
		raise RuntimeError(f"Could not load {mode}: {error}") from error
	return kube_client.AppsV1Api()


@dataclass(frozen=True)
class ControllerResult:
	"""Outcome from proposing or applying one scaling decision."""

	action: str
	current_replicas: Optional[int]
	decision: Optional[SafetyDecision]
	message: str


class PredictiveScalingController:
	"""Propose safe replica changes and optionally patch a Kubernetes scale.

	Pass an initialized ``kubernetes.client.AppsV1Api`` to ``reconcile`` to
	connect to a cluster. With the default ``dry_run=True``, it reads current
	replicas and logs a proposal but never patches the Deployment.
	"""

	def __init__(self, config: ScalingConfig, apps_api: Optional[Any] = None) -> None:
		self.config = config
		self.apps_api = apps_api
		self.logger = LOGGER

	def decide(
		self,
		predicted_request_rate_rps: Optional[float],
		current_replicas: int,
		forecast_timestamp: Optional[datetime],
		last_scale_time: Optional[datetime] = None,
		now: Optional[datetime] = None,
	) -> SafetyDecision:
		"""Create a bounded decision; invalid/missing forecasts always hold."""
		if isinstance(current_replicas, bool) or not isinstance(current_replicas, int):
			raise ValueError("current_replicas must be an integer")
		if not self.config.min_replicas <= current_replicas <= self.config.max_replicas:
			raise ValueError("current_replicas is outside configured replica limits")
		if predicted_request_rate_rps is None:
			return SafetyDecision(
				current_replicas,
				current_replicas,
				False,
				"Missing prediction; hold current replicas",
			)
		try:
			desired = calculate_desired_replicas(
				predicted_request_rate_rps, self.config
			)
		except (TypeError, ValueError, OverflowError) as error:
			return SafetyDecision(
				current_replicas,
				current_replicas,
				False,
				f"Invalid prediction ({error}); hold current replicas",
			)
		return apply_safety_rules(
			desired,
			current_replicas,
			self.config,
			forecast_timestamp=forecast_timestamp,
			last_scale_time=last_scale_time,
			now=now,
		)

	def reconcile(
		self,
		deployment_name: str,
		namespace: str,
		predicted_request_rate_rps: Optional[float],
		forecast_timestamp: Optional[datetime],
		last_scale_time: Optional[datetime] = None,
		now: Optional[datetime] = None,
	) -> ControllerResult:
		"""Read the Deployment scale, log the decision, and patch only if enabled."""
		if not deployment_name or not namespace:
			return ControllerResult("error", None, None, "deployment_name and namespace are required")
		if self.apps_api is None:
			return ControllerResult(
				"error",
				None,
				None,
				"No Kubernetes AppsV1Api client was provided; cluster access was not attempted",
			)

		try:
			scale = self.apps_api.read_namespaced_deployment_scale(
				name=deployment_name, namespace=namespace
			)
			current_replicas = scale.spec.replicas
			if current_replicas is None:
				raise ValueError("Deployment scale returned no replica count")
		except Exception as error:
			message = f"Could not read Deployment scale: {error}"
			self.logger.exception(message)
			return ControllerResult("error", None, None, message)

		try:
			decision = self.decide(
				predicted_request_rate_rps,
				current_replicas,
				forecast_timestamp,
				last_scale_time=last_scale_time,
				now=now,
			)
		except Exception as error:
			message = f"Could not evaluate scaling decision: {error}"
			self.logger.exception(message)
			return ControllerResult("error", None, None, message)
		self.logger.info(
			"Predictive scale decision deployment=%s namespace=%s current=%s "
			"requested=%s target=%s allowed=%s capacity_rps_per_replica=%s "
			"capacity_basis=%s dry_run=%s reason=%s",
			deployment_name,
			namespace,
			current_replicas,
			decision.requested_replicas,
			decision.target_replicas,
			decision.allowed,
			self.config.capacity_rps_per_replica,
			self.config.capacity_basis,
			self.config.dry_run,
			decision.reason,
		)

		if not decision.allowed or decision.target_replicas == current_replicas:
			return ControllerResult("no-op", current_replicas, decision, decision.reason)
		if self.config.dry_run:
			return ControllerResult(
				"dry-run",
				current_replicas,
				decision,
				f"Would change replicas from {current_replicas} to {decision.target_replicas}: {decision.reason}",
			)

		try:
			self.apps_api.patch_namespaced_deployment_scale(
				name=deployment_name,
				namespace=namespace,
				body={"spec": {"replicas": decision.target_replicas}},
			)
		except Exception as error:
			message = f"Could not patch Deployment scale: {error}"
			self.logger.exception(message)
			return ControllerResult("error", current_replicas, decision, message)

		return ControllerResult(
			"patched",
			current_replicas,
			decision,
			f"Patched replicas to {decision.target_replicas}: {decision.reason}",
		)
