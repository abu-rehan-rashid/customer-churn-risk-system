import os
import sqlite3

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as api_module
from src.predict import ChurnRiskEngine
from src.threshold_optimizer import find_optimal_threshold
from src.train import train

VALID_PAYLOAD = {
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


@pytest.fixture(scope="session", autouse=True)
def ensure_trained_model():
    if not os.path.exists("models/churn_pipeline.pkl"):
        train(data_path="data/churn_data.csv", models_dir="models")
    api_module.engine = ChurnRiskEngine("models/churn_pipeline.pkl")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    temp_db = tmp_path / "test_predictions.db"
    monkeypatch.setenv("PREDICTIONS_DB_PATH", str(temp_db))
    api_module.init_db()
    with TestClient(api_module.app) as test_client:
        yield test_client, temp_db


def test_health_endpoint(client):
    test_client, _ = client
    resp = test_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "online", "model_loaded": True}


def test_predict_valid_payload_and_logs_to_temp_db(client):
    test_client, temp_db = client
    resp = test_client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"Low Risk", "Medium Risk", "High Risk"}
    assert "recommended_action" in body
    assert body["latency_ms"] >= 0

    with sqlite3.connect(temp_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM prediction_logs").fetchone()[0]
    assert count == 1


def test_predict_rejects_empty_and_invalid_payloads(client):
    test_client, _ = client
    assert test_client.post("/predict", json={}).status_code == 422

    bad_charges = {**VALID_PAYLOAD, "TotalCharges": "abc"}
    assert test_client.post("/predict", json=bad_charges).status_code == 422

    negative_charges = {**VALID_PAYLOAD, "TotalCharges": -50.0}
    assert test_client.post("/predict", json=negative_charges).status_code == 422

    extra_field = {**VALID_PAYLOAD, "unknown_col": 123}
    assert test_client.post("/predict", json=extra_field).status_code == 422


def test_shap_explanation_shape_and_values():
    engine = ChurnRiskEngine("models/churn_pipeline.pkl")
    shap_df = engine.explain_instance(pd.DataFrame([VALID_PAYLOAD]))
    assert list(shap_df.columns) == ["Feature", "SHAP_Value"]
    assert len(shap_df) == 8
    assert not shap_df["SHAP_Value"].isna().any()


def test_threshold_optimizer_monotonicity():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_prob = np.array([0.90, 0.75, 0.60, 0.40, 0.25, 0.10])
    res = find_optimal_threshold(y_true, y_prob, retention_cost=50.0, customer_ltv=600.0, offer_success_rate=0.5)
    assert 0.1 <= res["optimal_threshold"] <= 0.9
    assert res["net_savings_usd"] > 0
