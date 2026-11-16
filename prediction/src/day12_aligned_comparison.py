import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

path = "results/prediction/day12_sample_xgboost_predictions.csv"

df = pd.read_csv(path)

# Previous actual workload is the Persistence forecast.
df["persistence_prediction_rps"] = (
    df["actual_request_rate_rps"].shift(1)
)

# First row in each split has no previous actual from that split.
# Exclude it so comparisons use aligned predictions.
df["previous_split"] = df["split"].shift(1)

df = df[
    (df["split"] == df["previous_split"])
].copy()

models = {
    "Persistence": "persistence_prediction_rps",
    "Moving Average": "moving_average_prediction_rps",
    "XGBoost": "xgboost_prediction_rps",
}

for split in ["validation", "test"]:
    subset = df[df["split"] == split]

    print(f"\n=== {split.upper()} ===")

    for model, column in models.items():
        mae = mean_absolute_error(
            subset["actual_request_rate_rps"],
            subset[column]
        )

        rmse = mean_squared_error(
            subset["actual_request_rate_rps"],
            subset[column]
        ) ** 0.5

        print(
            f"{model}: MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}, "
            f"n={len(subset)}"
        )