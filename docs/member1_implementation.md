# Member 1 Implementation: Predictive Workload Forecasting & Scaling Logic

## Responsibilities

- Data collection, cleaning, and feature engineering for cloud workload time-series.
- Baseline models (Moving Average, Exponential Smoothing) and machine learning forecasting models.
- Proactive auto-scaling policy and safety rule validations (min/max limits, cooldowns, oscillation dampers).
- Scaling controller integration and test coverage.

## Modules

1. `preprocessing.py`: Cleans raw time-series data and handles missing values/outliers.
2. `moving_average.py`: Simple and weighted moving average implementations for workload baseline.
3. `train_model.py`: Model training pipeline and weights export to `models/`.
4. `predict.py`: Real-time inference to forecast incoming demand / RPS / CPU load.
5. `scaling_logic.py`: Evaluates forecasts to calculate required replica count.
6. `safety_rules.py`: Enforces minimum/maximum replicas, cooldown windows, and flap prevention.
7. `controller.py`: End-to-end controller loop integrating forecasting, scaling decision, and safety execution.

## Day 5: Moving-Average Window Comparison

Moving-average windows were compared using walk-forward evaluation. Each forecast used only observations from before its target row.

| Window size | Status    |         MAE (RPS) |        RMSE (RPS) | Predictions |
| ----------- | --------- | ----------------: | ----------------: | ----------: |
| 2           | Evaluated |               5.0 | 6.013872850889571 |           3 |
| 3           | Evaluated | 8.166666666666666 | 8.209073570683167 |           2 |
| 4           | Skipped   |    Not calculated |    Not calculated |           0 |

Window 4 was skipped because five dataset rows are insufficient to leave two evaluated predictions. The comparison results are saved to `results/prediction/moving_average_window_comparison.csv`.

These results use the five-row synthetic smoke-test dataset, not real workload measurements. They are for pipeline verification only and are not evidence of real-world accuracy or a basis for selecting a production model.

Verification: 36 tests passed, and `python -m pip check` reported no broken requirements.

Run the comparison and tests from the project root in PowerShell, with `.venv` active:

```powershell
python -m prediction.src.compare_windows --input-csv prediction/data/raw/synthetic_smoke_test.csv --window-sizes 2 3 4 --dataset-label "synthetic smoke test (not real workload)"
python -m unittest discover -s prediction\tests -p "test_*.py"
python -m pip check
```

Day 5 covers baseline comparison. The exploratory XGBoost experiment is recorded under Day 7 below; live Kubernetes integration remains unverified.

## Day 6: Alibaba MCR investigation and service proxy

The original archive remains at `C:\Users\Sivakumar\Downloads\UpolishedData.zip` and was not modified. Its local MD5 (`cc2f1056f784e1c667ebf31c5e9a667f`) matches the Zenodo V2 `UpolishedData.zip` listing. This confirms the archive identity, not its internal field mapping. The archive's three MCR members contain 5,209,908 rows total and 90 consecutive `timestamp_ms` offsets, spaced 60,000 ms apart. Every scanned MCR row uses `consumerMQ_MCR`; duplicate checks found no exact duplicates or repeated `(timestamp_ms, id1, id2, metric)` keys within the files.

The official Alibaba README documents `MS_MCR_RT_Table` as `timestamp`, `msname`, `msinstanceid`, `metrics`, and `value`, and says MCR values are calls per second. The local MCR CSVs have six fields. Field 1 is unique and increasing within each shard, but its meaning is not documented and it is not globally unique across shards. The remaining fields align in order and observed types with timestamp, service name, instance ID, metric, and value; this mapping remains a best-supported interpretation, not a fully verified file specification. The README sections also differ in stated intervals; the inspected local MCR offsets are 60 seconds apart.

`prediction/data/processed/alibaba_consumer_mcr_service_proxy.csv` was present before the current processing attempt. Its 90 rows were independently recomputed from the archive and matched exactly. It represents the sum of `consumerMQ_MCR` values across instances for one selected `id1` (`2be3b98bfce0e36bfa57f93d594bb1e183b434540c5bce38233a5dc2e9b80a1b`) at each timestamp. This is an experimental **single-service proxy**, not validated cluster-wide incoming demand. `timestamp_ms` is represented as a UTC offset from the Unix epoch because the real wall-clock start date is unknown. The unexplained leading raw field is not included in the processed file.

The reproducible processor is `prediction/src/prepare_workload.py`. It checks the source schema, selected metric, duplicate keys, numeric values, expected coverage, and 60-second spacing. It refuses to overwrite an existing output unless explicitly authorized. It does not assert that the service-level sum is total application traffic.

To reproduce the selected-service proxy without replacing the existing processed CSV, run from the project root in PowerShell with `.venv` active:

```powershell
python -m prediction.src.prepare_workload --archive "$HOME\Downloads\UpolishedData.zip" --service-id 2be3b98bfce0e36bfa57f93d594bb1e183b434540c5bce38233a5dc2e9b80a1b --output prediction\data\processed\alibaba_consumer_mcr_service_proxy_reproduction.csv
```

## Day 7: Chronological XGBoost experiment

`prediction/src/train_model.py` builds lag and prior-only rolling-mean features, drops only the initial feature warm-up rows, and uses chronological 60/20/20 train/validation/test splits. No observations are randomly shuffled. On the 90-point experimental service proxy, 3 warm-up rows were excluded; split counts were 52 train, 17 validation, and 18 test.

| Split      | Method                   | MAE (proxy numeric units) | RMSE (proxy numeric units) | Predictions |
| ---------- | ------------------------ | -----------------: | -----------------: | ----------: |
| Validation | XGBoost                  | 1418.9666388863614 | 2404.4122108894235 |          17 |
| Validation | Moving average, window 3 | 385.72480070445357 |  695.3238705522451 |          17 |
| Test       | XGBoost                  | 1873.3839468862186 | 3499.1242414798644 |          18 |
| Test       | Moving average, window 3 | 205.42119723283292 |  299.8801949816415 |          18 |

For this run, XGBoost has larger errors than the moving-average baseline. The small, skewed service proxy does not support a real-world performance claim or a conclusion about cluster-wide demand.

Artifacts are saved separately from Day 4 outputs:

- `prediction/models/xgboost_service_proxy.joblib`
- `results/prediction/xgboost_service_proxy_predictions.csv`
- `results/prediction/xgboost_service_proxy_metrics.json`
- `results/graphs/xgboost_service_proxy_evaluation.png`

### Day 7: Error analysis and validation-selected persistence baseline

The saved Day 6 prediction CSV was checked row by row. XGBoost and the
three-observation moving average use identical target timestamps and actual
values: 17 validation targets (source rows 55–71) and 18 test targets (source
rows 72–89). The split is chronological. The test segment contains a large
proxy increase around source rows 81–83; predictions at rows 82–85 and 87 show
especially large XGBoost overestimates. The worst test error is at source row
87: actual 244.0705, XGBoost 10,949.6660, signed error (prediction minus
actual) +10,705.5955. Other largest errors were:

| Source row | Actual | XGBoost prediction | Signed error (prediction − actual) | Absolute error |
|---:|---:|---:|---:|---:|
| 87 | 244.0705 | 10,949.6660 | +10,705.5955 | 10,705.5955 |
| 82 | 1,043.4854 | 8,877.0488 | +7,833.5635 | 7,833.5635 |
| 85 | 493.0103 | 4,488.7485 | +3,995.7382 | 3,995.7382 |
| 84 | 796.0416 | 4,488.7485 | +3,692.7069 | 3,692.7069 |
| 83 | 899.8917 | 4,488.7485 | +3,588.8568 | 3,588.8568 |

Errors in this table are in proxy numeric units; their RPS interpretation is
not validated.

A small set of causal moving-average windows (1, 2, 3, 5, and 10) was compared.
The window was selected by validation MAE: window 1 had the lowest validation
MAE. The experiment CSV also records test scores for all five windows, but
those scores were not used for selection. Window 1 (persistence: use the
immediately prior observed value) was selected before examining its test score. Its validation
MAE/RMSE were 275.0973/478.2787 across 17 targets. On the chronological test
segment, the locked window-1 baseline scored MAE 107.3703 and RMSE 167.9155
across 18 targets. For comparison, the existing window-3 baseline scored
205.4212/299.8802, while XGBoost scored 1873.3839/3499.1242 on those same test
targets. This is one small holdout segment, not evidence of reliable future or
production performance. Each one-step walk-forward prediction uses prior
observed values available at that target time.

The evidence is consistent with sensitivity to abrupt spikes and very few
examples of such regimes: the series includes sharp bursts, and XGBoost
produces repeated extreme prediction values even when the corresponding actual
values are much lower. The experiment does not isolate whether the cause is
hyperparameters, sparse training regimes, tree behavior, or another factor, so
those remain hypotheses rather than established causes. No XGBoost tuning was
performed on the test segment.

Day 7 analysis artifacts:

- `results/prediction/xgboost_service_proxy_day7_window_experiments.csv`
- `results/prediction/xgboost_service_proxy_day7_error_analysis.json`

No source code or tests were changed for this analysis. The later Day 8 audit
matched the local input archive and reproduced the processed series, but the
source mapping still has unresolved fields and aggregation semantics. The
results do not establish validated RPS or cluster-wide demand forecasting. Do
not use the current XGBoost model for scaling decisions based on these results.

To repeat training without overwriting the verified artifacts, use distinct output paths:

```powershell
python -m prediction.src.train_model --input-csv prediction\data\processed\alibaba_consumer_mcr_service_proxy.csv --dataset-label "Alibaba consumerMQ_MCR selected-service proxy; exploratory, not cluster demand" --expected-sampling-interval 1min --model-output prediction\models\xgboost_service_proxy_reproduction.joblib --predictions-output results\prediction\xgboost_service_proxy_reproduction_predictions.csv --metrics-output results\prediction\xgboost_service_proxy_reproduction_metrics.json --graph-output results\graphs\xgboost_service_proxy_reproduction.png
```

## Day 8: Source schema and provenance audit

The local `C:\Users\Sivakumar\Downloads\UpolishedData.zip` was hashed
read-only. Its MD5 is
`cc2f1056f784e1c667ebf31c5e9a667f`, matching the `UpolishedData.zip` file
listed on [Zenodo record 10475644, version v2](https://zenodo.org/records/10475644).
This verifies that the local archive bytes match that published artifact; it
does not by itself document every transformation used to create the processed
CSV. The archive passed ZIP integrity checking and contains three MCR shards
with 5,209,908 rows total.

The v2 record describes its 180,000-row trace as recorded at 30-second
intervals, but it does not provide a field-level schema for either listed
file. The local `dataset-mcr-mt-180000.csv` has 30 distinct values in its
second positional column spaced 60 seconds apart, and the MCR archive shards
cover 90 consecutive offsets spaced 60 seconds apart. These are observed local
spacings, not authoritative confirmation of the intended sampling cadence;
the Zenodo description's 30-second statement remains unreconciled.

The Alibaba [v2021 trace README](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-microservices-v2021/README.md)
documents the five `MS_MCR_RT_Table` fields as `timestamp`, `msname`,
`msinstanceid`, `metrics`, and `value`. It defines `consumerMQ` as a service
consuming from a third-party message queue and includes `consumerMQ_MCR` among
the metrics. Its value description characterizes an MCR value as a number of
calls per second (with response time in milliseconds); this is v2021
documentation, not confirmation that Zenodo v2's repackaged files preserve
that schema or semantics. Zenodo's v2 record calls its artifacts a trace of
microservice time and call rates and describes a 30-second recording interval,
but supplies no field-level schema. The local files' observed 60-second
timestamp spacing does not resolve that documentation mismatch. Thus the
calls-per-second interpretation is documented for the Alibaba v2021 schema,
while the units and interval meaning of `consumerMQ_MCR` in these exact v2
files remain unverified. Neither source identifies this metric as incoming
user requests or cluster-wide demand.

The archived MCR rows have six positional fields, named by the processor as
`field1`, `timestamp_ms`, `msname`, `msinstanceid`, `metrics`, and `value`.
This is a code-assigned positional mapping, not a schema published for the
Zenodo v2 archive. The Alibaba v2021 five-field documentation gives plausible
counterparts for positions 2–6 (`timestamp`, `msname`, `msinstanceid`,
`metrics`, `value`), but no source located here establishes that compatibility.
`field1` is numeric, unique, and increasing within each shard in file order;
neither the Alibaba README nor the Zenodo record explains it. The project
mapping contains no field named `id2`; the two hashed identifier columns are
assigned to `msname` and `msinstanceid` by their observed values and the v2021
schema, which remains an unconfirmed inference for this archive.

The code behavior in `prediction/src/prepare_workload.py` is to read the three
named MCR members without headers using a six-name positional mapping, retain
rows matching the caller-supplied `msname`, and omit `field1` from the derived
records. It checks that the selected records contain only `consumerMQ_MCR`,
that timestamps and values are numeric and finite, timestamps are whole
milliseconds, values are non-negative, instance IDs are present, duplicate
`(timestamp_ms, msname, msinstanceid, metrics)` keys are absent, and the
configured timestamp count and spacing match. It then groups by timestamp,
sums `value`, counts distinct instances, represents the offsets as UTC times
from the Unix epoch, and writes the selected-service proxy. By default it
expects 90 timestamps spaced 60,000 ms apart and refuses to overwrite an
existing output. The output's `request_rate_rps` name and the sum operation
describe the current implementation; they do not independently verify the
source units or establish that summation is the authoritative service-level
rule. Alibaba v2021 says a microservice may have multiple containers and that
its metrics aggregate calls to/from DMs and UMs; it does not state that rates
should be summed across `msinstanceid` values. Cross-instance summation is the
processor's chosen proxy rule, not an aggregation rule verified by authoritative
documentation. Likewise, rejecting duplicate keys is the processor's local
defensive policy; the documentation located here provides no duplicate-record
resolution rule. The epoch-formatted timestamps are not verified calendar
dates.

The six-field positional mapping, the meaning of `field1`, its relationship
to Alibaba's five-field MCR schema, `consumerMQ_MCR` units and interval
semantics in this repackaged v2 archive, the intended cross-instance
aggregation, duplicate-record policy, and the schema and sampling cadence of
the seven-column file (including whether and how files with overlapping time
ranges may be combined) remain unverified.

The existing processed CSV was recomputed from the checksum-verified archive
using the current processor logic without writing files. It yielded the same
90 timestamps, instance counts, and workload values to floating-point
tolerance (maximum rate difference `5.7e-14`). The proxy itself does not embed
the source archive hash, so its provenance is established by this reproducible
comparison rather than internal metadata.

The local
`prediction/data/raw/alibaba_microservice_trace/dataset-mcr-mt-180000.csv`
also matches the Zenodo v2 file of that name by MD5
(`2099ad092226afe85db7247798d8300d`). It has seven columns and 179,997 rows
locally; the Zenodo record describes 180,000 rows but does not publish the
column definitions. Its second positional column has 30 distinct integer
offsets from 0 to 1,740,000 ms, spaced 60 seconds apart; treating that column
as a timestamp is an observed positional inference, not a source-defined
schema. It does not provide a longer continuous series than the 90-point MCR
proxy. The file has seven positional columns while the archive MCR shards
have six; because their layouts and relationship are undocumented and their
observed offsets overlap, these files must not be concatenated as if they
were compatible rows. The three archive MCR shards each contain 30 timestamp
offsets; together they cover 90 consecutive one-minute points.
The Alibaba README describes its broader v2021 MCR trace as a 12-hour dataset,
but that longer source data is not present locally. No replacement data was
downloaded and no raw file was changed.

The extra field meaning, exact six-field archive specification, the
seven-column derived file's schema, and whether summing instance rates is
formally intended as the canonical service aggregate remain unresolved. The
smallest next verification step is to request the v2 transformation/schema
documentation from the Zenodo authors, specifically the six-field shard and
the seven-field `dataset-mcr-mt-180000.csv` layouts and the cross-instance
aggregation rule. Until then, retain the existing proxy label and avoid
interpreting its model errors as validated RPS or cluster demand.

### Day 8 follow-up: clarification request status

The [Zenodo v2 record](https://zenodo.org/records/10475644) lists seven
creators and two data files, but no author email, schema supplement, or
transformation notes. The official [Alibaba clusterdata repository](https://github.com/alibaba/clusterdata)
publishes trace-team contact routes; these are verified contacts for Alibaba
trace maintainers, not confirmation that they are the Zenodo v2 curators. Per
the project owner's report, the schema clarification email has been sent; its
delivery has not been independently verified and no response has been
received or is claimed. The questions above remain open pending an
authoritative reply. Keep the processed series labeled as a selected-service
proxy and do not treat its values as verified RPS or cluster-wide demand.

### Day 8: Local verification closure (2026-10-28)

The local verification and documentation tasks are complete; the overall data
validation milestone remains **PARTIAL** because the source contract is still
unverified. Read-only checks using the project `.venv` established:

- `C:\Users\Sivakumar\Downloads\UpolishedData.zip` exists (591,058,711
  bytes), its MD5 is `cc2f1056f784e1c667ebf31c5e9a667f` (matching Zenodo v2's
  listed checksum), and `zipfile.ZipFile.testzip()` returned no corrupt member.
  The three MCR shards are present and contain 5,209,908 rows total.
- The processed CSV has 90 rows, columns `timestamp`, `request_rate_rps`,
  `timestamp_ms`, `service_id`, `metric`, and `instance_count`; no missing
  cells; unique, ordered `timestamp_ms` values with all 89 gaps exactly
  60,000 ms; and finite, non-negative values. Stored `timestamp` values match
  conversion of `timestamp_ms` from the Unix epoch. The selected service and
  metric are constant across the rows.
- The seven-column local file has 179,997 rows, and its MD5
  (`2099ad092226afe85db7247798d8300d`) matches Zenodo v2's listed checksum.
  Its second positional column has 30 distinct observed offsets, all of which
  overlap the processed proxy's offsets. The different column count and
  undocumented schema do not support combining the files.
- For an in-memory reproduction, the three archive members were read with the
  current six-name positional mapping, filtered by the processed file's
  selected service, and passed to `aggregate_service_records()` with the
  current 90-point and 60,000-ms expectations. The result matched all 90
  timestamps, service/metric labels, and instance counts. Maximum absolute
  workload difference versus the saved CSV was `5.684341886080802e-14`.
  No output file was written.

`preprocessing.load_workload_csv()` requires `timestamp` and
`request_rate_rps`, reports structural quality issues, and sorts valid times;
it cannot establish source units or schema. The preparation code's service
filter, metric filter, duplicate-key rejection, sum across instances, and
distinct-instance count are implementation behavior. The positional field
meanings, `field1`, v2 units and sampling semantics, cross-instance sum rule,
source duplicate policy, and seven-column schema/relationship remain
**UNVERIFIED**. The email remains sent per owner report; delivery is not
independently verified and no response is available in the project records.

For reproducible validation after authoritative schema information is
obtained: (1) save the source/version/schema guidance and checksum; (2) map
each source position and confirm units, cadence, aggregation, and duplicate
handling; (3) rerun ZIP integrity and raw structural checks; (4) reproduce in
memory using the current processor only if its mapping and aggregation match
the confirmed rules; otherwise update the processor in a separately reviewed
change; (5) compare row counts, timestamps, service/metric keys, instance
counts, and values against the saved CSV; and (6) if a newly generated file is
needed, write only to a checked, new output path and retain the existing proxy.
Do not combine the seven-column file unless its schema and overlap policy are
explicitly confirmed.

### Day 8: Option B assessment — v2021 schema candidate (2026-10-28)

The official Alibaba v2021 README defines the five MCR columns as
`timestamp`, `msname`, `msinstanceid`, `metrics`, and `value`; it lists
`consumerMQ_MCR`, describes `msinstanceid` as a container identifier, and
describes MCR values as calls per second. For the v2021 source, its MCR table
timestamp is tied to `MS_Metrics_Table`, whose documented recording interval
is 60 seconds. This is authoritative documentation for v2021, but it does not
prove that the checksum-matched Zenodo v2 archive was produced from that exact
schema. The Zenodo v2 page gives a general 30-second description and no
field-level mapping or transformation history. Those release-specific
differences remain unresolved.

A read-only positional scan of all three local MCR shards found 5,209,908
rows. Every row's fifth positional field was `consumerMQ_MCR`; positional
timestamp and value fields parsed as finite numeric values, with no negative
values. Each shard contains 30 distinct offsets, and the three shard ranges
are consecutive at 60,000-ms spacing. The six positions after assigning the
processor's provisional labels are consistent in order and observed type with
an extra field followed by the five documented v2021 fields. This establishes
structural compatibility, not release provenance or meaning.

The first positional field is an integer and strictly increasing within each
shard, but it is not a zero-based contiguous row count: its observed ranges
are 211,396–15,784,543, 210,770–15,825,348, and 633,516–15,861,833. That
pattern is consistent with a preserved source-row/index identifier, but does
not prove that interpretation. It remains **UNVERIFIED** and is not relabeled
or discarded as a known index.

For the already selected `msname`
`2be3b98bfce0e36bfa57f93d594bb1e183b434540c5bce38233a5dc2e9b80a1b` and
metric `consumerMQ_MCR`, the scan found 146,612 instance records and no
duplicate `(timestamp_ms, msname, msinstanceid, metrics)` keys. It found 1,629
instances with all 90 observed timestamps and one instance with only two.
Thus a no-sum, instance-level candidate is feasible if the positional mapping
is confirmed. One full-coverage instance observed was
`fff9a70087fd281f7f436d6952e8874321f2b1fb3217d4438c231152b81078d9`; its
90 numeric values are finite and non-negative at 60,000-ms steps. Selecting
one instance would describe that instance only, not service-wide or
cluster-wide demand.

No new processed file or forecast experiment was generated. The extra-field
meaning and the Zenodo-v2-to-Alibaba-v2021 schema crosswalk are not verified,
so writing a purported v2021 dataset or interpreting model scores as calls
per second would overstate the evidence. Forecast reevaluation is **BLOCKED**
until the selected file's five-field mapping, source interval semantics, and
metric units are supported for this exact archive. Cross-instance aggregation
is unnecessary for a one-instance series and remains unjustified for a
service-level series. Existing files and Days 1–8 artifacts were preserved.

When authoritative evidence becomes available, the safe reproducible path is:
(1) record the exact release, source filename, schema and checksum; (2) confirm
the extra field's meaning and the mapping of the five metric columns; (3)
select `msname`, `msinstanceid`, and `consumerMQ_MCR` explicitly; (4) keep
`timestamp` as the original elapsed millisecond integer and retain each
instance separately; (5) validate finite values, missing fields, duplicate
keys, timestamp ordering, gaps, and the documented interval; (6) compare the
result against the source using an in-memory transformation; (7) write to a
new, checked path only after those checks pass; then (8) evaluate persistence,
moving average, and XGBoost chronologically on that instance-level series,
using proxy-unit metric labels unless the exact source units are confirmed.
Do not combine the seven-column file or use forecasts for scaling without
separate schema and capacity evidence.

## Predictive scaling safety prototype

`prediction/src/config.py`, `prediction/src/scaling_logic.py`, and `prediction/src/safety_rules.py` now provide explicit capacity-based replica math, replica bounds, forecast freshness checks, cooldown handling, and bounded changes per decision. `prediction/config/scaling.example.json` is an example only: its capacity of 100 RPS per replica is an unmeasured placeholder, and `dry_run` is true.

`prediction/src/controller.py` accepts an injected Kubernetes `AppsV1Api` client, logs decision reasons, and only patches a deployment when configuration explicitly sets `dry_run` to false. Tests use a fake API; this project has not been tested against a reachable Kubernetes cluster, and live scaling is not claimed as verified.

`prediction/config/scaling.example.json` shows the required configuration format. Its capacity is only an illustrative placeholder and must be replaced with a measured assumption. To instantiate a local dry-run controller, load the config and client in Python; a reachable cluster and kubeconfig are still needed to read the target Deployment, but the example config will not patch it:

```python
from prediction.src.config import load_scaling_config
from prediction.src.controller import PredictiveScalingController, create_kubernetes_apps_api

config = load_scaling_config("prediction/config/scaling.example.json")
controller = PredictiveScalingController(config, create_kubernetes_apps_api())
```

## Current limitations and next work

- The processed series is a selected-service aggregate proxy, not validated cluster-wide demand. The extra leading archive field is still unexplained.
- The 90-point experiment is exploratory and too small to support robust model selection or production accuracy claims.
- Confirm the raw archive/version and timestamp/identifier mapping against authoritative version-specific metadata before treating the proxy as a validated workload dataset.
- Capacity per replica must be measured for the target application before using replica recommendations.
- Kubernetes API behavior still requires environment-specific integration testing; keep dry-run enabled until authorized and tested.
- Member 2's FastAPI, Docker, deployment, HPA, load generation, and monitoring responsibilities are not claimed as complete here.

## Latest verification

- 2026-10-27 Day 7 checkpoint retest: `& .\.venv\Scripts\python.exe -m unittest discover -s prediction\tests -p "test_*.py"` completed successfully: **67 tests passed, 0 failures, 0 errors**. The simulated Kubernetes read/patch error traces in the test output were expected mock failure cases. `& .\.venv\Scripts\python.exe -m pip check` reported **No broken requirements found**. No source code or dependencies were changed for this verification.
- All 18 Python files under `prediction/src/` and `prediction/tests/` compiled successfully.
- Full unittest suite: **67 tests passed**.
- `python -m pip check`: **No broken requirements found**.
- XGBoost model, prediction CSV, metrics JSON, and graph loaded successfully. MAE/RMSE recomputed from the saved CSV matched the metrics JSON on both validation and test splits.
- Day 4 prediction CSV, metrics JSON, and graph retained their previously recorded SHA-256 hashes.
- Kubernetes controller tests use mocked API clients only. No live cluster was contacted; a deployment scale change has not been verified.

## Offline integration and forecast-quality recheck (2026-10-28)

The current offline integration test, `prediction.tests.test_inference.InferenceIntegrationTests.test_existing_saved_model_end_to_end_offline_decision`, loads the existing `prediction/models/xgboost_service_proxy.joblib` artifact read-only, builds synthetic one-minute history, runs the inference adapter, and passes the forecast to the pure controller decision function. It asserts a finite non-negative forecast, aware and correctly ordered generation/target times, a decision within configured replica bounds, and that the Kubernetes client factory is not called. This verifies the saved-model-to-pure-decision path offline; it does not test a live cluster or deployment patch.

Commands run with the project virtual environment on 2026-10-28:

```powershell
& .\.venv\Scripts\python.exe -m unittest prediction.tests.test_inference
& .\.venv\Scripts\python.exe -m unittest prediction.tests.test_scaling_logic prediction.tests.test_controller_safety
& .\.venv\Scripts\python.exe -m unittest discover -s prediction\tests -p "test_*.py"
```

Results: inference tests **23 passed**; scaling/controller safety tests **107 passed**; full suite **178 passed**. The printed traceback-style log messages in safety tests are expected simulated failure cases from mocks/fake clients. No tests contacted a live Kubernetes cluster. `prediction/config/scaling.example.json` remains configured with `"dry_run": true`; its 100-RPS-per-replica capacity is still an illustrative, unmeasured placeholder.

The saved Day 6 prediction CSV has 17 validation targets (source rows 55–71) and 18 chronological test targets (source rows 72–89). XGBoost and the three-point moving average share the timestamp and actual target columns on all 35 rows. Recomputed MAE/RMSE from the saved values match the metrics JSON to displayed precision. On validation, XGBoost scored **1418.9666 / 2404.4122** and persistence scored **275.0973 / 478.2787**. On test, XGBoost scored **1873.3839 / 3499.1242**, moving average window 3 scored **205.4212 / 299.8802**, and persistence scored **107.3703 / 167.9155**. Persistence was selected by validation MAE among the tested moving-average windows; the test results were not used for that selection. These metrics are in the proxy's numeric units, not verified RPS.

The saved error analysis and prediction CSV show repeated large XGBoost overpredictions: XGBoost is above the actual value on 16 of 17 validation rows and 10 of 18 test rows. The largest test errors are source row 87 (actual 244.0705, prediction 10,949.6660; signed error +10,705.5955), row 82 (actual 1,043.4854, prediction 8,877.0488; +7,833.5635), and row 85 (actual 493.0103, prediction 4,488.7485; +3,995.7382). Repeated extreme predictions and bursty observations are evidence of the observed error pattern, but this small experiment does not isolate its cause. Do not claim XGBoost improves accuracy or use it for scaling decisions based on these results.

Day 8's separate root-level [`day8_audit_report.md`](../day8_audit_report.md) and its unresolved source-schema questions are preserved. Forecast interpretation remains limited by the unverified dataset provenance/units and the unmeasured replica capacity. Live Kubernetes behavior, production scaling, real-world load distribution, and latency improvements remain unverified.

### Forecast error audit details

The Day 6 saved prediction CSV has one row per validation/test target, with the same timestamp and actual value alongside both XGBoost and moving-average predictions. The metrics JSON reports 52 training targets, 17 validation targets, and 18 test targets after three feature warm-up rows; the intervals are chronological. `train_model.build_lagged_features()` uses shifted lag values and a rolling mean over `rates.shift(1)`, so each target's features use only prior observations. The test rows are not used to fit the model; the persistence window was selected by validation MAE in the saved Day 7 window experiment.

The inference loader verifies the saved model's stored four feature names against `lag_1_rps`, `lag_2_rps`, `lag_3_rps`, and `rolling_mean_3_rps`. Training and inference use the shared feature-building functions. Training fits the target in its existing numeric scale without a target transformation or inverse scaling; large estimates are therefore not explained by a scale-conversion step in this pipeline. XGBoost and moving-average predictions in the saved CSV are aligned by construction on the same target rows.

Read-only recomputation on 2026-10-28 produced the following results from the saved predictions and processed proxy. Persistence is computed from the immediately prior observed workload value and was cross-checked against the Day 7 window-experiment CSV:

| Split | Model | MAE (proxy numeric units) | RMSE (proxy numeric units) | Targets |
|---|---|---:|---:|---:|
| Validation | Persistence, window 1 | 275.0973 | 478.2787 | 17 |
| Validation | XGBoost | 1418.9666 | 2404.4122 | 17 |
| Validation | Moving average, window 3 | 385.7248 | 695.3239 | 17 |
| Test | Persistence, window 1 | 107.3703 | 167.9155 | 18 |
| Test | XGBoost | 1873.3839 | 3499.1242 | 18 |
| Test | Moving average, window 3 | 205.4212 | 299.8802 | 18 |

XGBoost overpredicts 16 of 17 validation targets and 10 of 18 test targets. Its largest test estimates occur at source rows 82 (8,877.0488 against 1,043.4854), 83–85 (repeated 4,488.7485 against lower actuals), and 87 (10,949.6660 against 244.0705). The training targets include a peak of 18,421.2216 at source row 34, while the test target maximum is 1,043.4854. This training spike and the small sample are plausible contributors to the observed behavior, but the evidence does not isolate a cause. Hyperparameter suitability, tree behavior, and sparse examples of workload regimes remain untested hypotheses. No tuning or retraining was performed in this audit.

All forecast scores remain in unverified proxy numeric units. This comparison does not establish RPS accuracy, cluster-wide demand performance, or production suitability. Keep persistence as the current baseline to beat and do not use the existing XGBoost estimates for scaling decisions.

## Public-source follow-up and Kubernetes readiness plan (2026-10-28)

The official Alibaba repository's [issue #120](https://github.com/alibaba/clusterdata/issues/120) records a user report that `HTTP_MCR` and `HTTP_RT` values in an original `MSRTQps_0.csv` appeared equal. A repository collaborator confirmed the equality and said the trace would be fixed and updated. This is a useful warning that source trace quality can have metric-specific issues. It is not evidence about `consumerMQ_MCR`, Zenodo v2's extra field, the v2 transformation, or its duplicate policy. The Day 8 report records the limits of this evidence; Zenodo v2 source semantics remain unresolved.

### Safe future Kubernetes test sequence

No live Kubernetes integration test has been run. Before one is considered, first resolve or explicitly bound the workload semantics used by the forecast and measure a per-replica capacity for the exact test application. Then prepare a disposable, non-production namespace and deployment, verify its identity and starting replica count, and scope credentials to read and scale only that namespaced deployment. Keep the repository example at `dry_run: true` and begin with fake-client/offline coverage. Any later live test requires separate authorization and an explicitly reviewed test configuration; it must not silently change the example config or target a production workload.

For an authorized disposable-cluster test, acceptance should require: dry-run proposes the expected bounded target with zero patch calls; invalid, stale, or timezone-invalid forecasts hold without patching; a separately approved non-dry-run test patches only the named disposable deployment within min/max and per-step limits; cooldown and scale-down safeguards behave as configured; observed replicas converge and the workload remains healthy; and rollback restores the recorded starting replica count. Record the cluster/context, namespace, deployment, configuration, before/after replica counts, patch result, health checks, and rollback result. Stop and roll back if the target identity differs, health degrades, or the result leaves configured bounds. These are proposed future acceptance criteria, not tests performed or production-readiness claims.

### Current project summary

The Member 1 prototype includes preprocessing, moving-average and XGBoost experiments, an explicit saved-model loader, next-step inference with timestamp and forecast validation, a pure replica decision path, and controller safety rules. The saved-model-to-decision path has been exercised offline with synthetic history; the current full offline suite ran 178 tests successfully on 2026-10-28. The exploratory model underperformed persistence on the small selected-service proxy. Dataset semantics and units, measured replica capacity, live Kubernetes behavior, and production scaling remain unverified. The current safe demonstration is an offline prototype showing a labeled proxy forecast flowing into a bounded dry-run recommendation, with no live patch.

### Saved-model error consistency check (2026-10-28)

Read-only inference with `prediction/models/xgboost_service_proxy_day6_exploratory.joblib` reproduced all 35 saved validation/test XGBoost predictions from the saved processed series and the training feature builder. The maximum absolute difference from the prediction CSV was `5.684341886080802e-14`. The artifact reports the expected feature names and order: `lag_1_rps`, `lag_2_rps`, `lag_3_rps`, `rolling_mean_3_rps`. The source-row keys are valid positions in the 90-row input; the 52/17/18 split corresponds to chronological source rows 3–54, 55–71, and 72–89 after three feature warm-up rows. Training code fits only the first split; each feature for target row `t` is built from observations no later than `t-1`. This evidence does not show a feature-order, prediction-file alignment, target-scale conversion, or future-feature leakage error in the saved experiment.

The source rows 82–87 show a test workload increase followed by a decrease. XGBoost predicts 8,877.0488 at row 82 and repeats 4,488.7485 at rows 83–85, then predicts 10,949.6660 at row 87. The model artifact reproduces these values for the corresponding prior-only feature rows. This confirms the error pattern is present in the fitted model, not a transcription mismatch in the saved CSV. It does not identify the internal model cause; the series is small, the training period contains a much higher peak, and no controlled model or feature ablation was run. No retraining, tuning, or artifact writes were performed.

### Remaining measurement and demonstration work

Application capacity is **not measured**. The value in the example config remains a placeholder because this repository does not establish a selected target application, its representative request mix, resource limits, or acceptable latency/error thresholds. A capacity result requires an isolated benchmark of the intended application: set resource limits and service-level thresholds first, increase load in repeatable steps, record achieved throughput, p95/p99 latency, errors, CPU, memory, and restarts, and select a conservative sustained throughput only from runs that meet every threshold. Until that benchmark is completed, retain the placeholder label and do not treat replica recommendations as production guidance.

The current offline demonstration can use the existing saved-model integration test and fake-client controller tests to show synthetic history → model forecast → pure scaling decision → bounds/safety evaluation → dry-run result. The tests cover scale-up, scale-down, hold, invalid forecast handling, bounds, and zero dry-run patch calls. This is an offline software demonstration, not a workload benchmark or live Kubernetes demonstration. Any later cluster demo remains subject to the disposable-cluster, target-scoped access, explicit approval, health monitoring, and rollback conditions in the readiness plan above. No live cluster test was performed for this follow-up.

### Day 16 - Controlled Local Capacity Validation

**Environment:** Local FastAPI sample application at `127.0.0.1:8001`, endpoint `/work?n=1000`. The benchmark used concurrency levels of 5 and 10, with three runs at each level.

**Results:** All six runs completed with zero failed requests. Every run met the configured p95 latency acceptance criterion of 250 ms or less. Median throughput was 130.76 requests/second at concurrency 5 and 142.49 requests/second at concurrency 10. Median p95 latency was 69.36 ms and 109.71 ms, respectively.

**Latency outliers:** Maximum observed latency reached 9,918.22 ms in one concurrency-10 run. CPU sampling showed average total CPU usage of 22.2%-26.6% in concurrency-10 runs and 25.0%-47.8% in concurrency-5 runs. The worst run had 19 counter samples and the longest wall-clock duration (19.282 seconds).

**Limitations:** CPU counters were sampled approximately once per second, and individual request latency measurements were not timestamped for correlation. Therefore, the available evidence cannot establish the cause of the latency outlier. Passing the p95 criterion does not establish stable maximum latency or production capacity.

**Safety and scope:** This was a local sample-application benchmark, not a production or Kubernetes capacity test. No live Kubernetes API was used. The existing `dry_run: true` setting and capacity placeholder of `100.0` were not changed. Port 8000 was left untouched.

**Conclusion:** The controlled local benchmark and response-validation checks passed their stated acceptance criteria, but substantial maximum-latency outliers remain unexplained. The observed throughput must not be treated as a verified per-replica capacity value. Further capacity calibration and live-cluster validation remain pending.

### Day 16 - Step 6: Existing-Data Latency Outlier Investigation

**Scope:** Reviewed the six existing local benchmark runs. No new benchmark was executed.

**Findings:** All runs completed 1,200 planned requests successfully with zero failed requests. All p95 latencies were below 250 ms. The three concurrency-5 runs had maximum latencies of 140.18, 197.95, and 362.59 ms. The three concurrency-10 runs had maximum latencies of 5,019.30, 4,768.63, and 9,918.22 ms.

**Interpretation:** Multi-second maximum latency occurred in all three observed concurrency-10 runs, while the concurrency-5 runs had substantially lower maximum latency. This is an observed association in this small benchmark, not proof that concurrency caused the outliers.

**Limitations:** The benchmark did not record individual request timestamps for correlation with CPU samples. The available aggregate results cannot establish the root cause. Potential explanations remain unverified.

**Conclusion:** The existing-data investigation is complete, but the latency root cause remains undetermined. Further targeted measurements would be needed to investigate it. No files or settings other than this documentation were changed. The capacity placeholder remains `100.0`, and Kubernetes `dry_run: true` remains unchanged.

**Status:** Existing-data investigation complete; root-cause investigation pending.
