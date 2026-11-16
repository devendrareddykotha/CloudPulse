
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load synthetic dataset
data_path = Path("prediction/data/sample_experiment/day12_synthetic_workload.csv")
df = pd.read_csv(data_path)

df["request_rate_rps"] = pd.to_numeric(
    df["request_rate_rps"], errors="raise"
)

actual = df["request_rate_rps"]

# One-step-ahead Persistence forecast
persistence = actual.shift(1)

# One-step-ahead Moving Average forecast
moving_average = actual.shift(1).rolling(window=3).mean()

# Evaluate only rows with valid predictions
results = pd.DataFrame({
    "actual": actual,
    "persistence_prediction": persistence,
    "moving_average_prediction": moving_average
})

results = results.dropna().reset_index(drop=True)

print("\n=== CloudPulse Day 12 Synthetic Experiment ===")
print("Dataset rows:", len(df))
print("Evaluated predictions:", len(results))

for model in ["persistence_prediction", "moving_average_prediction"]:
    mae = mean_absolute_error(
        results["actual"], results[model]
    )

    rmse = np.sqrt(mean_squared_error(
        results["actual"], results[model]
    ))

    print(f"\nModel: {model}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

# Save results separately
output_path = Path(
    "results/prediction/day12_sample_baseline_results.csv"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)
