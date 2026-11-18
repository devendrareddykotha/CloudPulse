# CloudPulse telemetry collection

CloudPulse workload telemetry can be collected at a 10-second cadence through
three complementary paths:

1. `evaluation/collect_metrics.py --run-id <run_id>` aggregates actual request
   records into the shared 10-second CSV contract. Empty CPU, memory, or replica
   fields mean that the run did not record those measurements; they are not
   inferred from request data.
2. `evaluation/collect_metrics.py --live-k8s` samples the Deployment replica
   count and pod CPU/memory usage with `kubectl get deployment` and `kubectl
   top pods`. The latter requires Kubernetes Metrics Server to be installed and
   ready. This path does not invent request or latency values.
3. `prometheus.yml` optionally scrapes the CloudPulse `/metrics` endpoint every
   10 seconds, both through the in-cluster Service DNS name and from a local
   Prometheus process.

The local request-rate endpoint (`/metrics/request-rate`) reports a rolling
60-second workload request count and rate. Prometheus scraping of `/metrics`
does not replace the per-request CSV aggregation or Kubernetes Metrics Server.
