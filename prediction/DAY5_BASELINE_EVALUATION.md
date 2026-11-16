# CloudPulse Day 5: Baseline Window Evaluation

## Dataset and method

The comparison used `prediction/data/raw/synthetic_smoke_test.csv`, a five-row synthetic development fixture. Each supported window was evaluated with the existing `evaluate_walk_forward()` function and the same moving-average method. Every target prediction used only earlier observed request rates. The comparison output is `results/prediction/moving_average_window_comparison.csv`.

## Results

| Window size | Status    | Evaluated predictions |         MAE (RPS) |        RMSE (RPS) |
| ----------- | --------- | --------------------: | ----------------: | ----------------: |
| 2           | Evaluated |                     3 |               5.0 | 6.013872850889571 |
| 3           | Evaluated |                     2 | 8.166666666666666 | 8.209073570683167 |
| 4           | Skipped   |                     0 |    Not calculated |    Not calculated |

Window size 4 needs at least six observations to leave two targets for evaluation (`window size + 2`). The dataset has five rows. It is explicitly marked `skipped` with this reason in the CSV; the evaluation threshold was not weakened.

## Limitations

These metrics are calculated from a synthetic smoke-test dataset, not real workload measurements. Five observations and only two or three scored targets are not enough for model selection or claims of real-world accuracy. No window is selected as a winner from this comparison. Meaningful performance conclusions require a sufficiently large real workload dataset with confirmed timestamp, timezone, and sampling assumptions.

## Reproduce

From the project root in PowerShell, with `.venv` active:

```powershell
python -m prediction.src.compare_windows --input-csv prediction/data/raw/synthetic_smoke_test.csv --window-sizes 2 3 4 --dataset-label "synthetic smoke test (not real workload)"
```
