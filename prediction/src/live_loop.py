"""Closed-loop predictive scaler; dry-run is the safe default."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import time
from typing import Callable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from prediction.src.config import ScalingConfig
from prediction.src.controller import (
    PredictiveScalingController,
)
from prediction.src.moving_average import moving_average


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "kubernetes" / "config.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "prediction" / "models" / "xgboost_service_proxy.joblib"
DEFAULT_METRICS_URL = "http://127.0.0.1:8000/metrics/request-rate"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "processed"
CSV_COLUMNS = (
    "timestamp_utc",
    "observed_rps",
    "predicted_rps",
    "predictor_used",
    "action",
    "current_replicas",
    "target_replicas",
    "reason",
)
MODEL_HISTORY_SIZE = 3


class LiveLoopError(RuntimeError):
    """Raised when a live-loop prerequisite or sample is invalid."""


def _positive_number(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def load_loop_config(config_path: str | Path) -> Tuple[ScalingConfig, int, int]:
    """Read shared Kubernetes ConfigMap parameters into controller settings."""
    path = Path(config_path)
    try:
        contents = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise LiveLoopError(f"Could not read controller config {path}: {error}") from error
    document = None
    try:
        import yaml

        document = yaml.safe_load(contents)
    except ImportError:
        data_lines = []
        in_data = False
        for line in contents.splitlines():
            if line.strip() == "data:":
                in_data = True
                continue
            if in_data:
                match = re.fullmatch(
                    r"\s{2}([A-Z][A-Z0-9_]*):\s*(?:\"([^\"]*)\"|'([^']*)'|([^#\s]+))\s*",
                    line,
                )
                if match:
                    value = next(
                        item for item in match.groups()[1:] if item is not None
                    )
                    data_lines.append((match.group(1), value))
                elif line.strip() and not line.startswith("  #"):
                    raise LiveLoopError(f"Unsupported ConfigMap data line: {line!r}")
        if not data_lines:
            raise LiveLoopError("ConfigMap YAML must contain a simple data mapping")
        document = {"data": dict(data_lines)}
    except Exception as error:
        if isinstance(error, LiveLoopError):
            raise
        raise LiveLoopError(f"Could not parse controller config {path}: {error}") from error
    if not isinstance(document, dict):
        raise LiveLoopError("Controller config must be a YAML object")
    data = document.get("data", document)
    if not isinstance(data, dict):
        raise LiveLoopError("Controller config data must be a YAML object")
    try:
        sampling_interval = int(data["SAMPLING_INTERVAL_SECONDS"])
        forecast_horizon = int(data["FORECAST_HORIZON_SECONDS"])
        capacity = float(data["TARGET_RPS_PER_REPLICA"])
        min_replicas = int(data["MIN_REPLICAS"])
        max_replicas = int(data["MAX_REPLICAS"])
        int(data["WORK_DEFAULT_UNITS"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise LiveLoopError(f"Invalid or missing shared ConfigMap parameter: {error}") from error
    if sampling_interval < 1 or forecast_horizon < 1:
        raise LiveLoopError("Sampling interval and forecast horizon must be positive")
    try:
        scaling_config = ScalingConfig(
            capacity_rps_per_replica=capacity,
            capacity_basis="Configured TARGET_RPS_PER_REPLICA; not independently measured",
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            dry_run=True,
        )
    except ValueError as error:
        raise LiveLoopError(f"Invalid scaling limits in ConfigMap: {error}") from error
    return scaling_config, sampling_interval, forecast_horizon


def fetch_request_rate(metrics_url: str, timeout: float = 10.0) -> Tuple[float, datetime]:
    request = Request(metrics_url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise LiveLoopError(f"Metrics endpoint returned HTTP {response.status}")
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise LiveLoopError(f"Could not read request-rate metrics: {error}") from error
    if not isinstance(payload, dict):
        raise LiveLoopError("Metrics response must be a JSON object")
    observed = payload.get("request_rate_rps")
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise LiveLoopError("Metrics response has no numeric request_rate_rps")
    observed = float(observed)
    if not math.isfinite(observed) or observed < 0:
        raise LiveLoopError("Metrics response request_rate_rps must be finite and nonnegative")
    try:
        timestamp = datetime.fromisoformat(
            str(payload["timestamp"]).replace("Z", "+00:00")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LiveLoopError("Metrics response has an invalid timestamp") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise LiveLoopError("Metrics response timestamp must include a timezone")
    return observed, timestamp.astimezone(timezone.utc)


def create_kubernetes_clients(kube_api_url: Optional[str] = None):
    """Load kubeconfig and create Apps/Autoscaling clients, optionally overriding the API URL."""
    try:
        from kubernetes import client, config
    except ImportError as error:
        raise LiveLoopError("Install the declared kubernetes Python dependency") from error
    try:
        config.load_kube_config()
        if kube_api_url:
            configuration = client.Configuration.get_default_copy()
            configuration.host = kube_api_url
            api_client = client.ApiClient(configuration)
            return (
                client.AppsV1Api(api_client),
                client.AutoscalingV2Api(api_client),
            )
        return client.AppsV1Api(), client.AutoscalingV2Api()
    except Exception as error:
        raise LiveLoopError(f"Could not initialize Kubernetes API clients: {error}") from error


def _hpa_targets_deployment(autoscaling_api, namespace: str, deployment: str) -> bool:
    try:
        response = autoscaling_api.list_namespaced_horizontal_pod_autoscaler(
            namespace=namespace
        )
    except Exception as error:
        raise LiveLoopError(f"Could not check for an HPA targeting {deployment}: {error}") from error
    for hpa in getattr(response, "items", ()):
        target = getattr(getattr(hpa, "spec", None), "scale_target_ref", None)
        if getattr(target, "name", None) == deployment:
            return True
    return False


def _predict(
    rates: Sequence[float],
    scheduled_times: Sequence[datetime],
    predictor_name: str,
    model,
    interval_seconds: int,
    generated_at: datetime,
) -> Tuple[float, str]:
    if predictor_name == "moving_average":
        window = min(3, len(rates))
        value = moving_average(rates, window)
        used = "moving_average" if len(rates) >= 3 else "moving_average_warmup"
        return value, used
    if len(rates) < MODEL_HISTORY_SIZE:
        value, _ = _predict(
            rates,
            scheduled_times,
            "moving_average",
            None,
            interval_seconds,
            generated_at,
        )
        return value, "moving_average_xgboost_warmup"
    if model is None:
        raise LiveLoopError("XGBoost predictor was not loaded")

    import pandas as pd
    from prediction.src.inference import forecast_next_step

    workload = pd.DataFrame(
        {
            "timestamp": scheduled_times,
            "request_rate_rps": list(rates),
        }
    )
    forecast = forecast_next_step(
        workload,
        model,
        sampling_interval=f"{interval_seconds}s",
        generated_at=generated_at,
        interpretation="Live CloudPulse request-rate metric",
    )
    return forecast.predicted_value, "xgboost"


def _unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    raise FileExistsError(f"Refusing to overwrite existing live-loop output: {path}")


def run_loop(
    *,
    iterations: int,
    interval: Optional[float],
    metrics_url: str,
    predictor_name: str,
    model_path: str | Path,
    config_path: str | Path,
    namespace: str,
    deployment: str,
    kube_api_url: Optional[str] = None,
    live: bool = False,
    output_csv: Optional[str | Path] = None,
    metrics_reader: Optional[Callable[[str], Tuple[float, datetime]]] = None,
    apps_api=None,
    autoscaling_api=None,
    sleep_fn: Optional[Callable[[float], None]] = None,
) -> Path:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if predictor_name not in {"moving_average", "xgboost"}:
        raise ValueError("predictor must be moving_average or xgboost")
    scaling_config, configured_interval, _ = load_loop_config(config_path)
    interval = configured_interval if interval is None else interval
    if interval <= 0:
        raise ValueError("interval must be positive")
    if output_csv is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = DEFAULT_OUTPUT_DIR / f"predictive_live_loop_{stamp}.csv"
    else:
        output_path = Path(output_csv)
    output_path = _unique_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if apps_api is None or (live and autoscaling_api is None):
        created_apps_api, created_autoscaling_api = create_kubernetes_clients(
            kube_api_url
        )
        apps_api = apps_api or created_apps_api
        autoscaling_api = autoscaling_api or created_autoscaling_api
    if not namespace or not deployment:
        raise ValueError("namespace and deployment must be nonempty")

    scaling_config = ScalingConfig(
        capacity_rps_per_replica=scaling_config.capacity_rps_per_replica,
        capacity_basis=scaling_config.capacity_basis,
        min_replicas=scaling_config.min_replicas,
        max_replicas=scaling_config.max_replicas,
        cooldown_seconds=scaling_config.cooldown_seconds,
        max_scale_up_step=scaling_config.max_scale_up_step,
        max_scale_down_step=scaling_config.max_scale_down_step,
        max_forecast_age_seconds=scaling_config.max_forecast_age_seconds,
        dry_run=not live,
    )
    controller = PredictiveScalingController(scaling_config, apps_api=apps_api)
    warmup_controller = PredictiveScalingController(
        ScalingConfig(
            capacity_rps_per_replica=scaling_config.capacity_rps_per_replica,
            capacity_basis=scaling_config.capacity_basis,
            min_replicas=scaling_config.min_replicas,
            max_replicas=scaling_config.max_replicas,
            cooldown_seconds=scaling_config.cooldown_seconds,
            max_scale_up_step=scaling_config.max_scale_up_step,
            max_scale_down_step=scaling_config.max_scale_down_step,
            max_forecast_age_seconds=scaling_config.max_forecast_age_seconds,
            dry_run=True,
        ),
        apps_api=apps_api,
    )
    model = None

    read_metrics = metrics_reader or fetch_request_rate
    wait = sleep_fn or time.sleep
    history_rates: List[float] = []
    sample_times: List[datetime] = []
    start_time = datetime.now(timezone.utc)
    metric_reader_origin = time.monotonic()
    with output_path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        output.flush()
        for index in range(iterations):
            scheduled_time = start_time + timedelta(seconds=index * interval)
            observed_rps, metric_timestamp = read_metrics(metrics_url)
            history_rates.append(observed_rps)
            sample_times.append(scheduled_time)
            generated_at = datetime.now(timezone.utc)
            if (
                predictor_name == "xgboost"
                and len(history_rates) >= MODEL_HISTORY_SIZE
                and model is None
            ):
                from prediction.src.inference import load_saved_predictor

                model = load_saved_predictor(model_path)
            predicted_rps, predictor_used = _predict(
                history_rates,
                sample_times,
                predictor_name,
                model,
                interval,
                generated_at,
            )
            hpa_present = (
                _hpa_targets_deployment(
                    autoscaling_api, namespace, deployment
                )
                if live
                else False
            )
            selected_controller = warmup_controller if hpa_present else controller
            result = selected_controller.reconcile(
                deployment_name=deployment,
                namespace=namespace,
                predicted_request_rate_rps=predicted_rps,
                forecast_timestamp=generated_at,
            )
            action = "blocked_hpa" if hpa_present else result.action
            reason = (
                f"Matching HPA is active; predictive scaling did not patch replicas. "
                f"{result.message}"
                if hpa_present
                else result.message
            )
            current_replicas = result.current_replicas
            target_replicas = (
                result.decision.target_replicas
                if result.decision is not None
                else current_replicas
            )
            writer.writerow(
                {
                    "timestamp_utc": metric_timestamp.isoformat(),
                    "observed_rps": observed_rps,
                    "predicted_rps": predicted_rps,
                    "predictor_used": predictor_used,
                    "action": action,
                    "current_replicas": (
                        "" if current_replicas is None else current_replicas
                    ),
                    "target_replicas": (
                        "" if target_replicas is None else target_replicas
                    ),
                    "reason": reason,
                }
            )
            output.flush()
            if result.action == "error":
                raise LiveLoopError(result.message)
            if hpa_present:
                print(reason)
            if index + 1 < iterations:
                elapsed = time.monotonic() - metric_reader_origin
                target_elapsed = (index + 1) * interval
                wait(max(0.0, target_elapsed - elapsed))
    return output_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Poll CloudPulse telemetry and safely reconcile predictive replica targets."
    )
    parser.add_argument("--iterations", type=_positive_integer, default=1)
    parser.add_argument("--interval", type=_positive_number)
    parser.add_argument("--metrics-url", default=DEFAULT_METRICS_URL)
    parser.add_argument(
        "--predictor", choices=("moving_average", "xgboost"), default="moving_average"
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--namespace", default="cloudpulse")
    parser.add_argument("--deployment", default="cloudpulse-service")
    parser.add_argument("--kube-api-url")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--dry-run", dest="live", action="store_false")
    execution.add_argument("--live", dest="live", action="store_true")
    parser.set_defaults(live=False)
    parser.add_argument("--output-csv", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        output = run_loop(
            iterations=arguments.iterations,
            interval=arguments.interval,
            metrics_url=arguments.metrics_url,
            predictor_name=arguments.predictor,
            model_path=arguments.model_path,
            config_path=arguments.config,
            namespace=arguments.namespace,
            deployment=arguments.deployment,
            kube_api_url=arguments.kube_api_url,
            live=arguments.live,
            output_csv=arguments.output_csv,
        )
    except (FileExistsError, FileNotFoundError, LiveLoopError, OSError, ValueError) as error:
        parser.exit(1, f"Predictive live loop failed: {error}\n")
    print(f"Live-loop observations saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
