# Member 2 Implementation Summary (K. Devendra Reddy)

## 1. Local Service, Dashboard & Dataset Pipeline (Days 1-9)
- **FastAPI Service (service/main.py) & Dashboard (service/dashboard.html)**: Live experiment monitor, saved run history, and side-by-side comparison UI.
- **Load Runner (	esting/run_load_test.py)**: Supports steady, spike, amp, and periodic profiles with a standard-library loopback runner compatible with Windows 11 ARM64 Smart App Control.
- **Dataset Pipeline (evaluation/generate_dataset.py, evaluation/aggregate_workload.py)**: Produces deterministic SHA-256 hash-named datasets under esults/processed/datasets/.
- **Verification**: 22/22 unit tests passing in .venv315-dashboard.

## 2. Docker & Kubernetes Deployment (Days 10-21)
- **Containers**: service/Dockerfile (cloudpulse-service:v1) and prediction/Dockerfile (cloudpulse-prediction:v1).
- **Kubernetes Manifests (kubernetes/)**: 
amespace.yaml, deployment.yaml, service.yaml (NodePort 30080), and hpa.yaml (cpu: 50%, 1-5 replicas) on Docker Desktop kind (1.36.1) with metrics-server (--kubelet-insecure-tls).

## 3. Live Integration & Benchmark Results (Days 35-50)
- **Unit Tests**: 22 	esting/ tests + 178 prediction/ tests (200/200 passing).
- **Reactive HPA Spike (locust_spike_k8s_hpa_20261116174859)**: 4,188 requests, 0 failures, 15.73 ms avg, 25.77 ms P95; HPA scaled 1 -> 3 pods after CPU hit 115%/50%.
- **Predictive Controller Spike (locust_spike_k8s_predictive_20261116175826)**: HPA removed before test; PredictiveScalingController (dry_run=False) scaled 1 -> 3 pods proactively; 4,188 requests, 0 failures, 16.42 ms avg, 28.05 ms P95.

## Phase 7, 9, and 12 additions (2026-11-17)

### Phase 7: nine-column 10-second metrics contract

Aggregate an existing per-request Locust CSV without changing it:

```powershell
.\.venv315-dashboard\Scripts\python.exe evaluation\collect_metrics.py --run-id locust_spike_k8s_hpa_20261116174859
```

The output is `results/processed/<run_id>_10s_contract.csv` with columns
`timestamp,request_count,request_rate_rps,cpu_utilization_pct,memory_usage_bytes,replica_count,average_latency_ms,p95_latency_ms,error_count`.
CPU, memory, and replicas remain blank for saved request CSVs because those
values are not present in the request records. Sample live cluster metrics
every 10 seconds with `evaluation\collect_metrics.py --live-k8s`; stop with
Ctrl+C. For a one-sample check, use `--live-k8s --samples 1`.

The saved HPA run produced 8 chronological rows. Loading that CSV inside the
existing `cloudpulse-prediction:v1` image with
`prediction/src/preprocessing.py` returned 8 rows, retained all nine columns,
and reported no data-quality problems (10-second expected interval).
A live one-sample check reported 1 available replica, 38,797,312 bytes of
pod memory, and 0.8% CPU relative to declared container CPU limits.

### Phase 9: pod identity and request distribution

`GET /workload` now returns `pod_name` from the `HOSTNAME` environment
variable (or `local`) and the same value in `X-Pod-Name`. To compare new
connections with reused connections, use:

```powershell
kubectl scale deployment/cloudpulse-service -n cloudpulse --replicas=3
kubectl port-forward service/cloudpulse-service -n cloudpulse 18000:8000
.\.venv315-dashboard\Scripts\python.exe evaluation\measure_distribution.py --url http://127.0.0.1:18000/workload --requests 300 --workers 1 --mode both
kubectl scale deployment/cloudpulse-service -n cloudpulse --replicas=1
```

The measurement tool is restricted to loopback targets and writes JSON and
CSV reports under `results/processed/`. The `v2` image rollout returned the
pod name in both the response body and header. With three Ready replicas,
300 fresh and 300 pooled requests through `kubectl port-forward` all reached
the tunnel's selected pod (zero failures in either mode). This is expected:
`kubectl port-forward service/...` selects a Pod and does not measure Service
load balancing.

The in-cluster Service probe made 300 fresh requests and measured:
`cloudpulse-service-5dc7d97854-9wl7l`: 103,
`cloudpulse-service-5dc7d97854-ztrs9`: 104, and
`cloudpulse-service-5dc7d97854-xf8jn`: 93. All three pods were reached, with
100 requests as the ideal share and an imbalance ratio of 1.04. The persisted
port-forward JSON/CSV reports are
`results/processed/request_distribution_20261117T110345Z.json` and
`results/processed/request_distribution_20261117T110345Z.csv`. The Deployment
was then restored to one Ready replica.

### Phase 12: saved-run comparison

Generate a CSV summary and SVG chart from two or more saved request CSVs:

```powershell
.\.venv315-dashboard\Scripts\python.exe evaluation\compare_results.py locust_spike_k8s_hpa_20261116174859 locust_spike_k8s_predictive_20261116175826
```

Verified output: `results/processed/comparison_20261117T105734Z_summary.csv`
and `results/graphs/comparison_20261117T105734Z.svg`. Both recorded 4,188
requests, 0 failures, and 75 seconds (55.84 requests/second). CSV-derived
average latency was 15.732 ms for HPA and 16.424 ms for predictive scaling;
CSV-derived P95 was 25.814 ms and 28.074 ms respectively.

### Verification

The requested `.\.venv315-dashboard\Scripts\python.exe -m unittest discover
-s testing -p "test_*.py" -v` completed with 31 tests passing. The service
Docker build completed and `cloudpulse-service:v2` rolled out with image digest
`417f5e1bf7c0`. The three-replica request-distribution check completed, then
the Deployment returned to one Ready replica.