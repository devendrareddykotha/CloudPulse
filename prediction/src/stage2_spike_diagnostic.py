"""Stage 2: Spike-sensitive diagnostic experiment.

Prepares a diagnostic copy of the workload series where the two training spike
observations (rows 30 and 34) are replaced with the median of their immediate neighbors.
Validation and test targets are completely untouched.
Generates diagnostic_training_series.csv and original_comparison.json.
"""

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from prediction.src.moving_average import evaluate_predictions

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "prediction/data/processed/alibaba_consumer_mcr_service_proxy.csv"
ORIGINAL = ROOT / "results/stage2_reproduction/xgboost_predictions.csv"
OUT = ROOT / "results/stage2_spike_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

SPIKE_ROWS = (30, 34)

data = pd.read_csv(DATA)
original = pd.read_csv(ORIGINAL)

assert len(data) == 90
assert len(original) == 35
assert set(original["split"]) == {"validation", "test"}
assert original["source_row"].is_unique

for row in SPIKE_ROWS:
    assert row < 54
    print(f"Spike row {row}: {data.loc[row, 'request_rate_rps']:.4f}")

# Preserve the original series.
treated = data.copy(deep=True)
replacements: Dict[int, Dict[str, float]] = {}

for row in SPIKE_ROWS:
    prev_val = float(data.loc[row - 1, "request_rate_rps"])
    next_val = float(data.loc[row + 1, "request_rate_rps"])
    neighbors = [prev_val, next_val]
    replacement = float(np.median(neighbors))
    treated.loc[row, "request_rate_rps"] = replacement
    replacements[row] = {
        "original_value": float(data.loc[row, "request_rate_rps"]),
        "neighbor_previous": prev_val,
        "neighbor_next": next_val,
        "replacement_median": replacement,
    }
    print(f"Row {row} replacement: {replacement:.4f} (neighbors: {prev_val:.4f}, {next_val:.4f})")

# Save diagnostic training series
diagnostic_csv = OUT / "diagnostic_training_series.csv"
treated.to_csv(diagnostic_csv, index=False)
print(f"Saved diagnostic series to: {diagnostic_csv}")

# Compute and structure original comparison baseline report
test_mask = original["split"] == "test"
val_mask = original["split"] == "validation"

test_actual = original.loc[test_mask, "actual_request_rate_rps"].tolist()
test_xgb = original.loc[test_mask, "xgboost_prediction_rps"].tolist()
test_ma3 = original.loc[test_mask, "moving_average_prediction_rps"].tolist()

# Persistence on test rows (previous observed workload: data[row - 1])
test_source_rows = original.loc[test_mask, "source_row"].tolist()
test_persistence = [float(data.loc[int(r) - 1, "request_rate_rps"]) for r in test_source_rows]

val_actual = original.loc[val_mask, "actual_request_rate_rps"].tolist()
val_xgb = original.loc[val_mask, "xgboost_prediction_rps"].tolist()
val_ma3 = original.loc[val_mask, "moving_average_prediction_rps"].tolist()
val_source_rows = original.loc[val_mask, "source_row"].tolist()
val_persistence = [float(data.loc[int(r) - 1, "request_rate_rps"]) for r in val_source_rows]


def compute_metrics(actual: list, predicted: list) -> Dict[str, float]:
    act = np.array(actual)
    pred = np.array(predicted)
    res = evaluate_predictions(actual, predicted)
    res["max_error_rps"] = float(np.max(np.abs(act - pred)))
    res["bias_rps"] = float(np.mean(pred - act))
    return res


comparison_report: Dict[str, Any] = {
    "experiment": "Stage 2 Spike-Treatment Diagnostic Baseline",
    "spike_replacements": replacements,
    "validation": {
        "xgboost_original": compute_metrics(val_actual, val_xgb),
        "moving_average_window_3": compute_metrics(val_actual, val_ma3),
        "persistence_window_1": compute_metrics(val_actual, val_persistence),
    },
    "test": {
        "xgboost_original": compute_metrics(test_actual, test_xgb),
        "moving_average_window_3": compute_metrics(test_actual, test_ma3),
        "persistence_window_1": compute_metrics(test_actual, test_persistence),
    },
    "test_rows": test_source_rows,
}

comparison_json = OUT / "original_comparison.json"
comparison_json.write_text(json.dumps(comparison_report, indent=2) + "\n", encoding="utf-8")
print(f"Saved original comparison report to: {comparison_json}")

