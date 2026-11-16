"""Historical replay evaluation comparing Persistence, Moving Average, and XGBoost.

Evaluates forecast accuracy and simulated replica decisions on the chronological
test split (source rows 72-89) of the Alibaba MCR selected-service proxy workload.
Simulates PredictiveScalingController decisions with historical timestamps without
invoking Kubernetes API endpoints.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prediction.src.config import ScalingConfig, load_scaling_config
from prediction.src.controller import PredictiveScalingController
from prediction.src.inference import forecast_next_step, load_saved_predictor
from prediction.src.moving_average import moving_average
from prediction.src.scaling_logic import calculate_desired_replicas
from prediction.src.train_model import (
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    build_lagged_features,
    load_workload_csv,
)

ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_PATH = (
    ROOT
    / "prediction"
    / "data"
    / "processed"
    / "alibaba_consumer_mcr_service_proxy.csv"
)
MODEL_PATH = (
    ROOT / "prediction" / "models" / "xgboost_service_proxy.joblib"
)
CONFIG_PATH = (
    ROOT / "prediction" / "config" / "scaling.example.json"
)
OUTPUT_ROOT = ROOT / "results" / "prediction"


def run_method(
    method: str,
    workload: pd.DataFrame,
    test_rows: pd.DataFrame,
    predictor: Any,
    config: ScalingConfig,
) -> List[Dict[str, Any]]:
    """Replay one forecasting method with independent simulated state."""
    controller = PredictiveScalingController(config)
    current_replicas = config.min_replicas
    last_scale_time = None
    rates = workload["request_rate_rps"].astype(float).tolist()
    records = []

    for row in test_rows.itertuples(index=False):
        source_row = int(row.source_row)
        target_time = pd.Timestamp(row.timestamp).to_pydatetime()
        actual_rps = float(row.actual_request_rate_rps)

        # In chronological sequence, forecast for target row t is generated at t - 1
        generated_at = pd.Timestamp(
            workload.iloc[source_row - 1]["timestamp"]
        ).to_pydatetime()
        history = workload.iloc[:source_row].copy()

        if method == "persistence":
            prediction = float(rates[source_row - 1])
        elif method == "moving_average":
            prediction = float(
                moving_average(
                    rates[:source_row],
                    window_size=3,
                )
            )
        elif method == "xgboost":
            forecast = forecast_next_step(
                history,
                predictor,
                sampling_interval="60s",
                generated_at=generated_at,
                interpretation="Historical selected-service proxy replay",
            )
            prediction = float(forecast.predicted_value)
            if forecast.target_at != target_time:
                raise RuntimeError(
                    f"Forecast timestamp mismatch at source row {source_row}"
                )
        else:
            raise ValueError(f"Unknown method: {method}")

        error_rps = prediction - actual_rps
        abs_error_rps = abs(error_rps)

        # Scaling decision using historical clock
        decision = controller.decide(
            prediction,
            current_replicas,
            forecast_timestamp=generated_at,
            last_scale_time=last_scale_time,
            now=generated_at,
        )

        prev_replicas = current_replicas
        scale_applied = decision.allowed and decision.target_replicas != current_replicas

        if scale_applied:
            current_replicas = decision.target_replicas
            last_scale_time = generated_at

        if current_replicas > prev_replicas:
            action = "scale_up"
        elif current_replicas < prev_replicas:
            action = "scale_down"
        else:
            action = "hold"

        # Capacity coverage under simulated replicas
        allocated_capacity = current_replicas * config.capacity_rps_per_replica
        under_provisioned = allocated_capacity < actual_rps
        over_provisioned = (
            (current_replicas - 1) * config.capacity_rps_per_replica >= actual_rps
            if current_replicas > config.min_replicas
            else False
        )

        records.append(
            {
                "method": method,
                "source_row": source_row,
                "timestamp": target_time.isoformat(),
                "generated_at": generated_at.isoformat(),
                "actual_request_rate_rps": actual_rps,
                "predicted_request_rate_rps": prediction,
                "error_rps": error_rps,
                "abs_error_rps": abs_error_rps,
                "prev_replicas": prev_replicas,
                "desired_replicas": decision.requested_replicas,
                "target_replicas": decision.target_replicas,
                "allowed": decision.allowed,
                "reason": decision.reason,
                "simulated_replicas": current_replicas,
                "scaling_action": action,
                "allocated_capacity_rps": allocated_capacity,
                "under_provisioned": under_provisioned,
                "over_provisioned": over_provisioned,
            }
        )

    return records


def calculate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate forecasting accuracy and scaling stability metrics."""
    df = pd.DataFrame(records)
    actual = df["actual_request_rate_rps"]
    predicted = df["predicted_request_rate_rps"]
    errors = df["error_rps"]
    abs_errors = df["abs_error_rps"]

    mae = float(abs_errors.mean())
    mse = float((errors ** 2).mean())
    rmse = float(math.sqrt(mse))
    max_abs_error = float(abs_errors.max())
    mean_signed_bias = float(errors.mean())

    replicas = df["simulated_replicas"]
    scale_ups = int((df["scaling_action"] == "scale_up").sum())
    scale_downs = int((df["scaling_action"] == "scale_down").sum())
    holds = int((df["scaling_action"] == "hold").sum())
    cooldown_holds = int(df["reason"].str.startswith("Cooldown active").sum())
    bounded_steps = int(df["reason"].str.startswith("Proposal bounded").sum())
    under_steps = int(df["under_provisioned"].sum())
    over_steps = int(df["over_provisioned"].sum())

    return {
        "observation_count": len(df),
        "forecast_accuracy": {
            "mae_rps": mae,
            "rmse_rps": rmse,
            "max_abs_error_rps": max_abs_error,
            "mean_signed_bias_rps": mean_signed_bias,
        },
        "replica_behavior": {
            "min_replicas": int(replicas.min()),
            "max_replicas": int(replicas.max()),
            "mean_replicas": float(replicas.mean()),
            "scale_up_events": scale_ups,
            "scale_down_events": scale_downs,
            "hold_events": holds,
            "cooldown_rejections": cooldown_holds,
            "step_bounded_events": bounded_steps,
        },
        "service_level_impact": {
            "under_provisioned_steps": under_steps,
            "over_provisioned_steps": over_steps,
        },
    }


def save_replay_graph(df_all: pd.DataFrame, output_path: Path) -> None:
    """Generate comparison visualization of forecasts and simulated replica adjustments."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Filter by method
    p_df = df_all[df_all["method"] == "persistence"]
    ma_df = df_all[df_all["method"] == "moving_average"]
    xgb_df = df_all[df_all["method"] == "xgboost"]

    x = range(len(p_df))
    ticks = [ts[11:16] for ts in p_df["timestamp"]]

    # Subplot 1: Workload & Forecasts
    ax1.plot(x, p_df["actual_request_rate_rps"], "k-o", label="Actual Workload (RPS)", linewidth=2.2, zorder=5)
    ax1.plot(x, p_df["predicted_request_rate_rps"], "g--s", label="Persistence (w=1)", alpha=0.85)
    ax1.plot(x, ma_df["predicted_request_rate_rps"], "b--^", label="Moving Average (w=3)", alpha=0.85)
    ax1.plot(x, xgb_df["predicted_request_rate_rps"], "r--d", label="XGBoost Model", alpha=0.85)
    ax1.set_ylabel("Request Rate (RPS)", fontsize=11, fontweight="bold")
    ax1.set_title("CloudPulse Historical Replay: Forecast Comparison on Test Split (Rows 72-89)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.9)

    # Subplot 2: Simulated Replicas
    ax2.step(x, p_df["simulated_replicas"], "g-", where="mid", label="Persistence Replicas", linewidth=1.8)
    ax2.step(x, ma_df["simulated_replicas"], "b-", where="mid", label="Moving Avg Replicas", linewidth=1.8)
    ax2.step(x, xgb_df["simulated_replicas"], "r-", where="mid", label="XGBoost Replicas", linewidth=1.8)
    ax2.set_ylabel("Simulated Replicas (c=100 RPS)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Timeline (UTC HH:MM)", fontsize=11, fontweight="bold")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(ticks, rotation=45, ha="right")
    ax2.set_ylim(0, 11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Run historical replay across all three methods and output artifacts."""
    parser = argparse.ArgumentParser(description="Historical predictive scaling replay evaluation")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    workload = load_workload_csv(WORKLOAD_PATH, expected_sampling_interval="60s")
    predictor = load_saved_predictor(MODEL_PATH)
    config = load_scaling_config(CONFIG_PATH)

    features, feature_columns, warmup_rows = build_lagged_features(
        workload, lags=DEFAULT_LAGS, rolling_windows=DEFAULT_ROLLING_WINDOWS
    )

    feature_count = len(features)
    train_end = int(feature_count * 0.6)
    validation_end = int(feature_count * 0.8)
    test_rows = features.iloc[validation_end:].copy().reset_index(drop=True)

    if len(test_rows) != 18:
        raise ValueError(f"Expected 18 test rows, found {len(test_rows)}")

    methods = ["persistence", "moving_average", "xgboost"]
    all_records: List[Dict[str, Any]] = []
    summary_results: Dict[str, Any] = {}

    for method in methods:
        method_records = run_method(method, workload, test_rows, predictor, config)
        all_records.extend(method_records)
        summary_results[method] = calculate_metrics(method_records)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_ROOT / f"historical_replay_{timestamp_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    records_df = pd.DataFrame(all_records)
    csv_path = output_dir / "replay_records.csv"
    records_df.to_csv(csv_path, index=False)

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workload_file": str(WORKLOAD_PATH),
        "model_file": str(MODEL_PATH),
        "config_file": str(CONFIG_PATH),
        "scaling_config": {
            "capacity_rps_per_replica": config.capacity_rps_per_replica,
            "capacity_basis": config.capacity_basis,
            "min_replicas": config.min_replicas,
            "max_replicas": config.max_replicas,
            "cooldown_seconds": config.cooldown_seconds,
            "max_scale_up_step": config.max_scale_up_step,
            "max_scale_down_step": config.max_scale_down_step,
            "max_forecast_age_seconds": config.max_forecast_age_seconds,
            "dry_run": config.dry_run,
        },
        "test_set": {
            "row_count": len(test_rows),
            "source_row_start": int(test_rows["source_row"].min()),
            "source_row_end": int(test_rows["source_row"].max()),
            "start_timestamp": str(test_rows["timestamp"].iloc[0]),
            "end_timestamp": str(test_rows["timestamp"].iloc[-1]),
        },
        "methods": summary_results,
    }

    json_path = output_dir / "replay_summary.json"
    json_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    graph_path = output_dir / "replay_comparison.png"
    save_replay_graph(records_df, graph_path)

    # Formatted terminal display
    sep = "=" * 80
    subsep = "-" * 80
    print(sep)
    print("CloudPulse Historical Replay: Test Set Evaluation (Rows 72-89, N=18)")
    print(sep)
    print(f"Capacity Basis : {config.capacity_basis}")
    print(f"Capacity (RPS) : {config.capacity_rps_per_replica} | Cooldown: {config.cooldown_seconds}s | Steps: +{config.max_scale_up_step}/-{config.max_scale_down_step}")
    print(subsep)
    hdr1 = f"{'Method':<16} {'MAE (RPS)':<10} {'RMSE (RPS)':<11} {'MaxErr':<10} {'Bias (RPS)':<11} {'Replicas (Min/Avg/Max)':<24}"
    print(hdr1)
    print(subsep)
    for method in methods:
        m = summary_results[method]
        fa = m["forecast_accuracy"]
        rb = m["replica_behavior"]
        rep_str = f"{rb['min_replicas']} / {rb['mean_replicas']:.2f} / {rb['max_replicas']}"
        print(f"{method:<16} {fa['mae_rps']:<10.2f} {fa['rmse_rps']:<11.2f} {fa['max_abs_error_rps']:<10.2f} {fa['mean_signed_bias_rps']:<11.2f} {rep_str:<24}")
    print(subsep)
    hdr2 = f"{'Method':<16} {'Scale Ups':<11} {'Scale Downs':<13} {'Holds':<8} {'Cooldown Block':<16} {'Under-prov':<10}"
    print(hdr2)
    print(subsep)
    for method in methods:
        m = summary_results[method]
        rb = m["replica_behavior"]
        sli = m["service_level_impact"]
        print(f"{method:<16} {rb['scale_up_events']:<11} {rb['scale_down_events']:<13} {rb['hold_events']:<8} {rb['cooldown_rejections']:<16} {sli['under_provisioned_steps']:<10}")
    print(sep)
    print(f"CSV Records    : {csv_path}")
    print(f"JSON Summary   : {json_path}")
    print(f"Graph Artifact : {graph_path}")
    print(sep)


if __name__ == "__main__":
    main()

