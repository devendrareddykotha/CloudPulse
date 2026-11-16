# CloudPulse Predictive Scaling — Historical Replay Evaluation Report

**Document ID:** CP-REP-20261031-01  
**Evaluation Scope:** Historical Replay of Workload Rows 72–89 ($N=18$ one-minute steps)  
**Target Workload:** `alibaba_consumer_mcr_service_proxy.csv` (Selected-Service Microservice Call Rate)  
**Models Evaluated:** Persistence ($w=1$), Moving Average ($w=3$), XGBoost Production Model (`xgboost_service_proxy.joblib`)  
**Execution Environment:** Python 3.10.8 Virtual Environment, Windows 11 Pro  
**Scaling Engine:** `PredictiveScalingController` (Offline historical simulation mode, zero cluster I/O)  
**Status:** Verification Complete & Reproducible  

---

## 1. Executive Summary

This report delivers the complete evaluation of the **Historical Replay** component for the CloudPulse Predictive Scaling system. The replay rigorously compares three request-rate forecasting methodologies across the exact chronological test split (source rows 72–89, spanning `1970-01-01 01:12:00+00:00` to `01:29:00+00:00` UTC) under identical historical conditions.

Each method was evaluated with independent simulated replica state and cooldown tracking using the production `PredictiveScalingController` logic with historical timestamps.

### Key Evaluation Findings:

1. **Forecasting Accuracy Superiority of Simple Baselines:**
   - **Persistence ($w=1$):** Achieved the lowest forecast error with an **MAE of 107.37 RPS** and **RMSE of 167.92 RPS**, exhibiting near-zero mean signed bias (**-4.67 RPS**).
   - **Moving Average ($w=3$):** Produced an **MAE of 205.42 RPS** and **RMSE of 299.88 RPS** with a modest positive bias (**+38.05 RPS**).
   - **XGBoost:** Produced an **MAE of 1873.38 RPS**, an **RMSE of 3499.12 RPS**, and an extreme positive bias of **+1752.85 RPS**, peaking at a catastrophic maximum prediction error of **10,705.60 RPS** at row 87. This exactly reproduces the bit-for-bit error observed in Stage 2.

2. **Controller Safety Rules Successfully Dampen Wild Over-Predictions:**
   - At source row 87, XGBoost predicted **10,949.67 RPS**, which would naively demand over 100 replicas (or the configured maximum of 10 replicas).
   - The `PredictiveScalingController`'s step bounding ($+2$ scale-up limit per event) and 300-second cooldown rule strictly bounded the target, holding simulated replicas at **5**, and later peaking at **6 replicas**.
   - This empirically confirms that the architectural safety bounds effectively insulate the deployment from catastrophic ML model failures.

3. **Reactivity vs. Stability Trade-offs:**
   - Persistence triggered 3 scale-up events, 1 scale-down, and 14 holds, with 7 decisions blocked by cooldown.
   - Moving Average triggered 3 scale-up events, 0 scale-downs, and 15 holds, with 8 decisions blocked by cooldown.
   - XGBoost triggered 3 scale-ups, 1 scale-down, and 14 holds, with 11 decisions blocked by cooldown due to chronic over-requesting.

---

## 2. Methodology & Chronological Test Split

The evaluation strictly adheres to the data split established during initial model training:
- **Workload:** 90 one-minute aggregate observations of Alibaba MCR trace proxy data.
- **Warm-up Rows:** 3 rows excluded for feature lags ($lags \in \{1, 2, 3\}$, rolling window $w=3$).
- **Train Split (60%):** Usable rows 0–51 (Source rows 3–54, $N=52$).
- **Validation Split (20%):** Usable rows 52–68 (Source rows 55–71, $N=17$).
- **Test Split (20%):** Usable rows 69–86 (Source rows 72–89, $N=18$).

### Simulation State Isolation

To ensure scientific validity, each forecasting method maintained an isolated instance of the simulated autoscaler:
- **Clock Synchronization:** At target step $t$, the forecast was generated strictly using observations available up to $t-1$ (`generated_at = timestamp[t-1]`). The historical clock was passed as `now = generated_at` to validate forecast age ($\le 120$s) and cooldown ($\ge 300$s).
- **Independent State:** Replica changes and cooldown timers from one method had zero impact on any other method.

---

## 3. Comparative Evaluation Results

### Table 1: Forecast Accuracy and Bias Comparison ($N=18$)

| Forecasting Method | Window / Features | MAE (RPS) | RMSE (RPS) | Max Absolute Error (RPS) | Mean Signed Bias (RPS) | Error vs. Persistence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | $w=1$ (lag 1) | **107.37** | **167.92** | **387.43** | **-4.67** | Baseline |
| **Moving Average** | $w=3$ (rolling mean) | **205.42** | **299.88** | **662.53** | **+38.05** | $+91.3\%$ error |
| **XGBoost** | Lags 1-3, Rolling 3 | **1873.38** | **3499.12** | **10,705.60** | **+1752.85** | $+1644.8\%$ error |

### Table 2: Simulated Autoscaler Decision Metrics

*Evaluated with illustrative capacity $C = 100.0$ RPS/replica, limits $[1, 10]$, step limits $+2 / -1$, cooldown $300$s.*

| Metric | Persistence ($w=1$) | Moving Average ($w=3$) | XGBoost (Production) |
| :--- | :---: | :---: | :---: |
| **Minimum Simulated Replicas** | 2 | 3 | 2 |
| **Maximum Simulated Replicas** | 5 | 6 | 6 |
| **Mean Simulated Replicas** | 3.44 | 3.78 | 3.50 |
| **Scale-Up Events** | 3 | 3 | 3 |
| **Scale-Down Events** | 1 | 0 | 1 |
| **Hold Decisions** | 14 | 15 | 14 |
| **Cooldown Rejections** | 7 | 8 | **11** |
| **Step-Bounded Decisions** | 2 | 1 | 3 |
| **Under-Provisioned Steps** | 7 | 5 | 9 |
| **Over-Provisioned Steps** | 4 | 6 | 6 |

---

## 4. Visual Analysis: Workload, Forecasts, and Replica Decisions

Chart artifact saved to `replay_comparison.png`.

1. **Step-Bounding in Action (01:21 – 01:23 UTC):**
   - As actual traffic surged from 270.7 RPS to 1043.5 RPS, all three methods requested rapid scale-ups.
   - For Persistence at row 82, requested replicas jumped to 7, but the controller bounded the step to $+2$, granting 5 replicas (`simulated_replicas = 5`).
   - For XGBoost, requested replicas jumped to 10 (the maximum), but the controller bounded the change to 4, then to 6 on the next allowed window.
2. **Cooldown Suppression of Downscaling:**
   - Following the peak at 01:22 UTC, traffic plummeted back down to ~240 RPS by 01:26 UTC.
   - The 300-second cooldown prevented immediate replica thrashing, ensuring stability while traffic stabilized.
3. **Over-provisioning vs. Service Protection:**
   - Moving average maintained slightly higher average replica count (3.78) than persistence (3.44), resulting in fewer under-provisioned steps (5 vs 7) during the onset of the surge, at the cost of slight over-provisioning during recovery.

---

## 5. Safety Compliance & Constraints

The execution adhered strictly to all CloudPulse project safety rules:
- **Zero Cluster I/O:** The historical replay ran completely offline using pure functional controller calls (`controller.decide`). No Kubernetes API client was initialized or contacted.
- **Unmodified Protected Settings:**
  - `capacity_rps_per_replica`: Exactly `100.0`
  - `capacity_basis`: Preserved as `"ILLUSTRATIVE ONLY: unmeasured placeholder; replace with a measured capacity before use"`
  - `dry_run`: `true`
- **Model Deserialization Integrity:** The trusted production model was loaded via `load_saved_predictor`, confirming the exact 4-feature contract (`['lag_1_rps', 'lag_2_rps', 'lag_3_rps', 'rolling_mean_3_rps']`).

---

## 6. Generated Deliverables

The evaluation produced the following permanent artifacts:

1. **Python Replay Script:**
   - `prediction/src/historical_replay.py`
2. **Replay Run Results:**
   - CSV Records: `results/prediction/historical_replay_20261031_194256/replay_records.csv`
   - JSON Summary: `results/prediction/historical_replay_20261031_194256/replay_summary.json`
   - Graphical Artifact: `results/prediction/historical_replay_20261031_194256/replay_comparison.png`
3. **PDF Deliverables:**
   - `results/prediction/CloudPulse_Historical_Replay_Report.pdf`

