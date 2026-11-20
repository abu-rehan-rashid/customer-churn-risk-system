import sqlite3

import numpy as np
from fastapi.testclient import TestClient

from app import DB_PATH, app, init_db
from src.threshold_optimizer import find_optimal_threshold

client = TestClient(app)


def setup_module(module):
    init_db()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True


def test_valid_prediction_and_db_logging():
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 4,
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
        "MonthlyCharges": 89.5,
        "TotalCharges": 358.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert "latency_ms" in body
    assert body["latency_ms"] >= 0

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT tenure, contract_type FROM prediction_logs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row == (4, "Month-to-month")


def test_invalid_tenure_returns_422():
    response = client.post("/predict", json={"tenure": -5})
    assert response.status_code == 422


def test_invalid_contract_returns_422():
    response = client.post("/predict", json={"Contract": "Weekly"})
    assert response.status_code == 422


def test_cost_threshold_optimizer():
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    y_proba = np.array([0.85, 0.62, 0.45, 0.12, 0.78, 0.20, 0.15, 0.55, 0.30, 0.10])

    result = find_optimal_threshold(
        y_true=y_true,
        y_proba=y_proba,
        retention_cost=50.0,
        customer_ltv=600.0,
        offer_success_rate=0.5,
    )
    assert 0.05 <= result["optimal_threshold"] <= 0.95
    assert result["net_savings_usd"] > 0