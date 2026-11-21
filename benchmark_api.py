import statistics
import time

import httpx

API_URL = "http://127.0.0.1:8000/predict"

SAMPLE_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 6,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 510.0,
}


def run_benchmark(n_requests: int = 50) -> None:
    latencies_ms: list[float] = []
    errors = 0

    with httpx.Client(timeout=10.0) as client:
        # Warm-up request
        client.post(API_URL, json=SAMPLE_PAYLOAD)

        start_total = time.perf_counter()
        for _ in range(n_requests):
            t0 = time.perf_counter()
            resp = client.post(API_URL, json=SAMPLE_PAYLOAD)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                latencies_ms.append(elapsed_ms)
            else:
                errors += 1
        total_sec = time.perf_counter() - start_total

    latencies_ms.sort()
    p50 = statistics.median(latencies_ms)
    p95 = latencies_ms[int(0.95 * len(latencies_ms)) - 1]
    p99 = latencies_ms[int(0.99 * len(latencies_ms)) - 1]
    rps = round(len(latencies_ms) / total_sec, 2)

    print(f"\n--- FastAPI Inference Benchmark ({n_requests} requests) ---")
    print(f"Successful : {len(latencies_ms)} | Errors: {errors}")
    print(f"Throughput : {rps} req/sec")
    print(f"p50 Latency: {p50:.2f} ms")
    print(f"p95 Latency: {p95:.2f} ms")
    print(f"p99 Latency: {p99:.2f} ms")


if __name__ == "__main__":
    run_benchmark(50)