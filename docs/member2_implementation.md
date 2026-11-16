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