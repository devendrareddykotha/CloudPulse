# CloudPulse - Workload Prediction & Proactive Scaling

Module for workload prediction, time-series forecasting, and predictive auto-scaling logic.

## Directory Structure

- `data/`: Raw and preprocessed metrics data.
- `models/`: Saved prediction models and artifacts.
- `src/`: Source code including preprocessing, moving average baselines, ML model training, prediction, scaling logic, safety checks, and controller.
- `tests/`: Unit and integration test suite.

## Workload CSV Validation

`src/preprocessing.py` expects these columns:

- `timestamp`: The time associated with a workload measurement.
- `request_rate_rps`: The request rate in requests per second.

`load_workload_csv()` raises a clear error when the file is missing, empty, or
does not contain both required columns. For other data-quality issues, it prints
the number of affected rows and their CSV row numbers, and stores the same
counts and row details in the returned DataFrame's
`df.attrs["validation_report"]`.

Rows with invalid timestamps or request rates are retained. Timestamps that
cannot be parsed are represented as `NaT`; missing or non-numeric request rates
are represented as `NaN`. Valid timestamps are sorted chronologically and
invalid timestamps appear last. The raw CSV file is not modified, and no rows or
measurements are silently removed, repaired, or invented.

The timezone and sampling interval have not been confirmed with Member 2. By
default, timezone information from each timestamp is preserved and sampling-gap
checks are disabled. Both can be provided as optional loader arguments when
confirmed.

`data/raw/synthetic_smoke_test.csv` is a synthetic development fixture only; it
is not real workload data and does not demonstrate prediction performance.

## Day 4: Moving-Average Baseline Outputs

From the project root in PowerShell, with `.venv` active, run:

```powershell
python -m prediction.src.predict --input-csv prediction/data/raw/synthetic_smoke_test.csv --window-size 2 --data-label "synthetic smoke-test (not real workload)"
```

The command calls the existing workload loader and moving-average functions. It
uses walk-forward predictions: each target's prediction is based only on the
preceding window of observed request rates. Invalid timestamps/rates and
duplicate timestamps are reported by preprocessing and then rejected by the
baseline command rather than dropped or aligned incorrectly. At least two
walk-forward targets are required for the reported evaluation.

By default, the command writes:

- `results/prediction/moving_average_predictions.csv`: target timestamp, actual
  RPS, predicted RPS, method, and window size for each evaluated target.
- `results/prediction/moving_average_metrics.json`: MAE and RMSE in RPS, number
  of evaluated predictions, method, window, input path, and run metadata.
- `results/graphs/moving_average_actual_vs_predicted.png`: actual and predicted
  rates plotted at their target timestamps.

MAE is the average absolute difference between actual and predicted rates;
RMSE is the square root of the average squared difference, so larger errors
have more influence. The graph's horizontal axis is the forecast target time;
its vertical axis is request rate in requests per second. Optional
`--predictions-output`, `--metrics-output`, and `--graph-output` arguments can
change the output paths.

The five-row smoke-test CSV is synthetic. Its generated metrics and graph only
verify that the pipeline runs and outputs align; they are not evidence of
real-world workload performance or a comparison against another model.
