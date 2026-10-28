# CloudPulse

## Repeatable workload experiments

Install the project dependencies once in the active Python environment:

```powershell
python -m pip install -r requirements.txt
```

Start the FastAPI service in one terminal from the repository root:

```powershell
python -m uvicorn service.main:app --host 127.0.0.1 --port 8000
```

Open the live dashboard in your browser:

```text
http://127.0.0.1:8000/dashboard
```

The dashboard refreshes automatically, showing measurements from the active
Locust run's request records. Use **Start experiment** to launch a steady-load
or sudden-spike profile and **Stop experiment** to stop it. The backend rejects
a second start while an experiment is active. Throughput is a rolling
10-second request rate; request totals, failures, and latency statistics are
cumulative for the current run. Kubernetes metrics remain unavailable until a
real Kubernetes monitoring integration is configured.

Use **Saved experiments** to select a run recorded under `results/raw/`.
Historical selections are read-only and use the saved metadata and per-request
CSV; the dashboard remembers the selection in this browser and rediscovers the
run after the FastAPI service restarts. The live experiment controls remain
available while viewing history.

The read-only history APIs are:

- `GET /api/experiments/history` lists runs with valid metadata and request CSV
  pairs, including counts and latency summaries calculated from their request
  records.
- `GET /api/experiments/history/{run_id}` returns the selected run's summary
  metrics and measured per-second request-rate and latency series. Unknown,
  malformed, or incomplete runs return `404`.

Use **Compare saved experiments** on the dashboard to choose two saved runs and
compare their total requests, failures, average/median/P95 latency, and
average throughput. The request-rate and mean-latency charts use the saved
request CSVs and align each run's time axis to its own start time; scenario and
run ID labels identify each series. Comparison is read-only and does not change
the current live experiment.

The comparison API is:

- `GET /api/experiments/compare/{run_id_a}/{run_id_b}` returns the CSV-derived
  summaries and time series for two valid saved runs. Invalid IDs or runs with
  missing or malformed metadata/request files return `404`.

## Repeatable workload datasets

Generate a dataset from an existing, finished Locust run. This command reads
only that run's metadata and per-request CSV; it does not start an experiment:

```powershell
.\.venv-clean\Scripts\python.exe evaluation\generate_dataset.py --run-id locust_steady_20261028T063016_210669Z_01b7f5
```

The generator supports the runner's `steady`, `spike`, `ramp`, and `periodic`
scenarios. Output is written to a content-addressed directory under
`results/processed/datasets/<run_id>_<source-hash>/`. It contains:

- `request_records.csv`: one row per validated request. Its exact columns are
  `run_id`, `scenario`, `timestamp_utc`, `elapsed_seconds`, `method`, `name`,
  `url`, `status_code`, `response_time_ms`, `response_length`, `exception`,
  `failed`, `configured_peak_user_count`, and `test_duration_seconds`.
- `per_second.csv`: one row for each elapsed second with request records.
  `timestamp_utc` is the UTC bin start; `elapsed_second` is its integer offset
  from run start. `requests_per_second` and `request_count` are the actual
  request count in that one-second bin. Other exact columns are `run_id`,
  `scenario`, `mean_response_latency_ms`, `latency_sample_count`, `failures`,
  `error_rate`, `failure_sample_count`, `configured_peak_user_count`, and
  `test_duration_seconds`.
- `manifest.json`: schema version 1.1, run configuration/status, row counts,
  source hash, file names, and validation warnings.

Elapsed bins are anchored to each run's metadata start time, not wall-clock
calendar intervals. Empty seconds are omitted rather than filled with
fabricated zeroes. A missing latency remains blank; a per-second mean is blank
if any request in that second lacks latency. Failure count and error rate
remain blank when any request's failure outcome is unknown. Missing values are
reported in the manifest and CLI output. Empty files, missing columns,
malformed timestamps or measurements, and duplicate request records stop
generation with a validation error.

The configured user count is the run's configured peak; instantaneous
per-second active-user counts are not present in the request CSV and are not
inferred. The pipeline likewise does not derive CPU, memory, or Kubernetes
metrics. Requests recorded after the configured duration are retained rather
than clipped; the verified 60-second run above has its last request at elapsed
60.887614 seconds, so its aggregated bins run through elapsed second 60.

Generation is deterministic for unchanged source files and schema version.
Repeating the command reuses byte-identical output. Existing output is never
replaced: if a generated destination already exists but differs, generation
fails and preserves it. Raw experiment files and the existing processed
reports are untouched.

### Saved-run scenario coverage

The runner and dataset generator support all four scenarios. Completed saved
request-record runs were found only for steady and spike; no ramp or periodic
run is currently available to validate against real measurements. Datasets
were generated or deterministically reused for all seven completed runs:

| Scenario | Run ID | Raw requests | Dataset status |
|---|---|---:|---|
| steady | `locust_steady_20261027T171544_290429Z_8b0613` | 605 | generated |
| steady | `locust_steady_20261028T061519_500234Z_fd2164` | 305 | generated |
| steady | `locust_steady_20261028T062855_162198Z_cf73b6` | 605 | generated |
| steady | `locust_steady_20261028T063016_210669Z_01b7f5` | 605 | reused identical dataset |
| spike | `locust_spike_20261012T053617_095009Z` | 262 | generated |
| spike | `locust_spike_20261012T102804_682589Z_043fa4` | 45 | generated |
| spike | `locust_spike_20261028T061017_704805Z_4dfde1` | 4,358 | generated |

For each completed run, request rows and per-second request-count sums matched
the raw CSV; failure counts and latency sample totals reconciled, and the
weighted per-second mean latency matched the raw-record mean. Raw source-file
SHA-256 hashes were unchanged. One steady run is interrupted and another
failed without a request CSV; neither was treated as a completed run or
included in this coverage.

To collect real ramp-up and periodic records when approved, run these from the
repository root after starting the local FastAPI service:

```powershell
.\.venv-clean\Scripts\python.exe testing\run_load_test.py --scenario ramp --users 10 --duration 60 --spawn-rate 5
.\.venv-clean\Scripts\python.exe testing\run_load_test.py --scenario periodic --users 10 --duration 60 --spawn-rate 5
```

These commands start Locust experiments and have not been run as part of this
dataset-validation task. After each run completes, generate its dataset using
the new run ID printed by the runner:

```powershell
.\.venv-clean\Scripts\python.exe evaluation\generate_dataset.py --run-id <printed-run-id>
```

You can also run one scenario at a time from a second terminal. `--users` sets
peak concurrent users, `--duration` is in seconds, and `--spawn-rate` controls
how quickly Locust changes the active user count.

```powershell
python testing/run_load_test.py --scenario steady --users 10 --duration 120
python testing/run_load_test.py --scenario spike --users 10 --duration 120
python testing/run_load_test.py --scenario ramp --users 10 --duration 120
python testing/run_load_test.py --scenario periodic --users 10 --duration 120
```

The periodic scenario requires a duration of at least 8 seconds. Each run stops after its configured duration; press `Ctrl+C` to stop early. Stop the service with `Ctrl+C` in its terminal.

Locust writes uniquely named `locust_<scenario>_<UTC timestamp>_*.csv` statistics files under `results/raw/`. The files include summary statistics, full-history statistics, failures, and per-request records. A matching metadata JSON records the configuration and UTC start/end times. The service continues to append its request-level records to `results/raw/request_logs.csv`.

### Sudden-spike experiment

The spike runner uses a low-load warm-up, a jump to peak users, a sustained peak, and a return to low load. For example, this run uses 2 users during warm-up and recovery, then jumps to 10 peak users:

```powershell
python testing/run_load_test.py --scenario spike --users 10 --low-users 2 --warmup-duration 10 --peak-duration 20 --recovery-duration 10 --warmup-spawn-rate 2 --spike-spawn-rate 10 --recovery-spawn-rate 2
```

The three phase durations determine the total run time. The phase-specific spawn rates control how quickly Locust adds or removes users at each transition. Each run saves its configuration in a uniquely named metadata JSON alongside its CSV results.

Stop a manually started terminal experiment with `Ctrl+C`. Experiments started
from the dashboard can be stopped using its **Stop experiment** control.
