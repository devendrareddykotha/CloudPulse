import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import http.client
import ipaddress
import json
from datetime import datetime, timezone
from pathlib import Path
import threading
from urllib.parse import urlsplit


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "processed"
DEFAULT_URL = "http://127.0.0.1:8000/workload"
MAX_REQUESTS = 1000
MAX_WORKERS = 20
REPORT_COLUMNS = (
    "mode",
    "pod_name",
    "request_count",
    "total_responses",
    "failed_requests",
    "ideal_equal_share",
    "load_imbalance_ratio",
)


class DistributionError(ValueError):
    pass


def validate_workload_url(url):
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.path.rstrip("/") != "/workload"
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise DistributionError(
                "URL must be a loopback HTTP(S) URL ending in /workload."
            )
        if parsed.hostname.lower() != "localhost" and not ipaddress.ip_address(
            parsed.hostname
        ).is_loopback:
            raise DistributionError("Distribution probes are restricted to loopback.")
    except ValueError as error:
        if isinstance(error, DistributionError):
            raise
        raise DistributionError("Invalid loopback workload URL.") from error
    return parsed


def parse_response_identity(headers, body):
    response_headers = dict(headers)
    pod_name = response_headers.get("X-Pod-Name") or response_headers.get("x-pod-name")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = {}
    if not pod_name and isinstance(payload, dict):
        pod_name = payload.get("pod_name")
    return pod_name or "unknown"


def load_imbalance_ratio(pod_counts):
    if not pod_counts:
        return None
    total = sum(pod_counts.values())
    if total < 1:
        return None
    ideal_share = total / len(pod_counts)
    return max(pod_counts.values()) / ideal_share


def _request(parsed, connection, mode):
    connection.request(
        "GET",
        parsed.path or "/workload",
        headers={"Connection": "close" if mode == "fresh" else "keep-alive"},
    )
    response = connection.getresponse()
    body = response.read()
    identity = parse_response_identity(response.getheaders(), body)
    failed = response.status < 200 or response.status >= 400
    return identity, failed


def probe_connection_mode(url, request_count=300, workers=1, mode="both", timeout=5):
    parsed = validate_workload_url(url)
    if mode not in {"fresh", "pooled", "both"}:
        raise DistributionError(f"Unsupported connection mode: {mode}")
    if not 1 <= request_count <= MAX_REQUESTS:
        raise DistributionError(f"Request count must be between 1 and {MAX_REQUESTS}.")
    if not 1 <= workers <= MAX_WORKERS:
        raise DistributionError(f"Workers must be between 1 and {MAX_WORKERS}.")
    connection_type = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    modes = ("fresh", "pooled") if mode == "both" else (mode,)
    results = []

    for selected_mode in modes:
        pod_counts = {}
        failures = 0
        lock = threading.Lock()

        def worker(work_count):
            nonlocal failures
            connection = None
            try:
                for _ in range(work_count):
                    if selected_mode == "fresh":
                        connection = connection_type(host, port, timeout=timeout)
                    elif connection is None:
                        connection = connection_type(host, port, timeout=timeout)
                    try:
                        pod_name, failed = _request(
                            parsed, connection, selected_mode
                        )
                    except (OSError, http.client.HTTPException):
                        with lock:
                            failures += 1
                        if connection is not None:
                            connection.close()
                            connection = None
                        continue
                    with lock:
                        pod_counts[pod_name] = pod_counts.get(pod_name, 0) + 1
                        failures += int(failed)
                    if selected_mode == "fresh":
                        connection.close()
                        connection = None
            finally:
                if connection is not None:
                    connection.close()

        distribution, remainder = divmod(request_count, workers)
        work = [
            distribution + (index < remainder) for index in range(workers)
        ]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(worker, work))

        total_responses = sum(pod_counts.values())
        ideal_share = (
            total_responses / len(pod_counts) if pod_counts else None
        )
        results.append(
            {
                "mode": selected_mode,
                "pod_counts": dict(sorted(pod_counts.items())),
                "total_responses": total_responses,
                "failed_requests": failures,
                "ideal_equal_share": ideal_share,
                "load_imbalance_ratio": load_imbalance_ratio(pod_counts),
            }
        )
    return results


def write_report(results, output_dir=OUTPUT_DIR, stamp=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"request_distribution_{stamp}.json"
    csv_path = output_dir / f"request_distribution_{stamp}.csv"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    with json_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    with csv_path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for result in results:
            for pod_name, count in result["pod_counts"].items():
                writer.writerow(
                    {
                        "mode": result["mode"],
                        "pod_name": pod_name,
                        "request_count": count,
                        "total_responses": result["total_responses"],
                        "failed_requests": result["failed_requests"],
                        "ideal_equal_share": result["ideal_equal_share"],
                        "load_imbalance_ratio": result["load_imbalance_ratio"],
                    }
                )
    return json_path, csv_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure loopback CloudPulse per-pod request distribution."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mode", choices=("fresh", "pooled", "both"), default="both")
    parser.add_argument("--timeout", type=float, default=5)
    args = parser.parse_args(argv)
    try:
        results = probe_connection_mode(
            args.url, args.requests, args.workers, args.mode, args.timeout
        )
        json_path, csv_path = write_report(results)
    except (DistributionError, OSError) as error:
        parser.exit(1, f"Distribution measurement failed: {error}\n")
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    for result in results:
        print(
            f"{result['mode']}: responses={result['total_responses']}, "
            f"failures={result['failed_requests']}, "
            f"pods={result['pod_counts']}, "
            f"imbalance={result['load_imbalance_ratio']}"
        )


if __name__ == "__main__":
    main()
