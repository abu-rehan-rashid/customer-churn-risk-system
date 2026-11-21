# 🛡️ Customer Churn Risk Prediction System

![CI Pipeline](https://github.com/abu-rehan-rashid/customer-churn-risk-system/actions/workflows/ci.yml/badge.svg)

Production-grade churn prediction system on the Kaggle Telco dataset featuring **calibrated XGBoost probabilities**, **SHAP explainability**, a **cost-based business ROI threshold optimizer**, a **Streamlit analytics workbench**, and a **FastAPI scoring service** with **SQLite telemetry** and **automated CI/CD**.

<p align="center">
  <img width="100%" alt="Real-Time Evaluator & SHAP Explainability" src="https://github.com/user-attachments/assets/9f68b9cc-38a5-4f0e-ae54-a779ed6633b9" />
</p>

<p align="center">
  <img width="48%" alt="What-if Sensitivity Simulator" src="https://github.com/user-attachments/assets/81b1247c-9e8d-4058-b4c2-b6b916b22130" />
  <img width="48%" alt="Model Health & Governance Metrics" src="https://github.com/user-attachments/assets/127108f3-2c41-46b3-9b47-108c022df74f" />
</p>

---

## Highlights & Production Engineering

- **Calibrated Probabilities & Baseline Benchmarking (`src/train.py`):** Wraps `XGBClassifier` in isotonic `CalibratedClassifierCV` (`Brier Score: 0.1348`) inside a leak-free scikit-learn `Pipeline`, benchmarked via 5-fold Stratified CV against a Logistic Regression baseline (`0.8463` vs `0.8451` CV ROC-AUC).
- **Explainable Predictions (`SHAP`):** SHAP values (log-odds, XGBoost `TreeExplainer`) are summed from one-hot columns back to original business features (e.g., all `Contract_*` columns → `Contract`) for both per-customer explanations and global holdout importances (`models/feature_importances.json`).
- **Cost-Based Threshold Optimizer (`src/threshold_optimizer.py`):** Evaluates out-of-fold validation probabilities (`models/oof_predictions.csv`) to calculate the probability cutoff (`0.23`) that maximizes net USD retention savings based on offer cost, customer LTV, and campaign save rate instead of relying on a static `0.50` threshold.
- **Strict Pydantic Guardrails (`app.py`):** Enforces required fields, `extra="forbid"`, domain bounds (`tenure` 0–120, `MonthlyCharges` > 0, `TotalCharges` ≥ 0), and literal contract/service categories, rejecting malformed inputs with `422 Unprocessable Entity`.
- **SQLite Inference & Drift Telemetry (`logs/predictions.db`):** Best-effort logging of every `/predict` payload, calibrated predicted probability, risk tier, and real-time inference latency (`latency_ms`) for production monitoring without blocking live responses.
- **Automated Testing & CI/CD (`.github/workflows/ci.yml`):** Every push runs `ruff` linting, trains the pipeline, executes real-model `pytest` unit/integration tests on an isolated temporary DB, and runs a `docker build` + `/health` verification on GitHub Actions.

| Risk tier | Probability | Recommended action |
|---|---|---|
| 🔴 High | ≥ 55% | Immediate retention offer / priority call |
| 🟠 Medium | 23% to 54% | Targeted email campaign and feedback survey |
| 🟢 Low | < 23% | No action needed |

## Model Performance & API Benchmarks

Evaluated on a stratified holdout set of 1,409 customers (Kaggle Telco Customer Churn, 26.5% base churn rate).

| Operating Point | ROC-AUC | PR-AUC | Brier Score | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| **Cost-Optimal Cutoff (`0.23`)** | 0.846 | 0.662 | 0.1348 | **83.9%** | 49.6% | 0.623 |
| **Default Cutoff (`0.50`)** | 0.846 | 0.662 | 0.1348 | 52.9% | **67.1%** | 0.592 |

**Holdout Confusion Matrix (`threshold = 0.50`):**

| | Predicted retained | Predicted churned |
|---|---|---|
| **Actually retained** | 938 | 97 |
| **Actually churned** | 176 | 198 |

### FastAPI Service Latency (`benchmark_api.py`, 50 Sequential Requests)

| Metric | Value | Notes |
|---|---|---|
| **p50 Latency** | **69.75 ms** | Includes preprocessing, calibrated XGBoost inference & SQLite telemetry write |
| **p95 Latency** | **109.80 ms** | Sub-110ms tail latency under local single-worker Uvicorn |
| **p99 Latency** | **170.03 ms** | 0% error rate across benchmark run |
| **Throughput** | **12.38 req/s** | Single-process local worker |

## Tech Stack

Python 3.11+, scikit-learn, XGBoost, SHAP, FastAPI, Pydantic, SQLite, Streamlit, Plotly, Pandas, Pytest, Ruff, Docker, GitHub Actions.

## Quick Start

```bash
git clone [https://github.com/abu-rehan-rashid/customer-churn-risk-system.git](https://github.com/abu-rehan-rashid/customer-churn-risk-system.git)
cd customer-churn-risk-system

python -m venv venv
venv\Scripts\activate          # Windows (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt

# Generate trained model artifacts (models/churn_pipeline.pkl, metrics.json, oof_predictions.csv & feature_importances.json)
python src/train.py

streamlit run ui_app.py        # Dashboard at http://localhost:8501
uvicorn app:app --reload       # API at http://localhost:8000 (Swagger docs at /docs)

Run Tests & Latency Benchmark:
python -m pytest -v tests/
python benchmark_api.py        # Requires uvicorn running on port 8000

Docker:
docker compose up --build

API Usage
GET / and GET /health return {"status": "online", "model_loaded": true} (or 503 if model artifacts are missing).
"http://127.0.0.1:8000/predict"a

POST /predict validates all 19 customer features (extra="forbid") against strict Pydantic schemas and returns calibrated churn probability, risk tier, action, and latency:
curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 510.0
  }'


  Architecture
  data/churn_data.csv ──> src/train.py ──> models/churn_pipeline.pkl
   (Kaggle Telco)        (calibration     models/metrics.json
                          + SHAP + OOF)   models/oof_predictions.csv
                                          models/feature_importances.json
                                                    │
                                                    ▼
                                  src/predict.py & threshold_optimizer.py
                                  predict_risk() · explain_instance() · ROI
                                                    │
                                  ┌─────────────────┴─────────────────┐
                                  ▼                                   ▼
                       app.py (FastAPI + SQLite)           ui_app.py (Streamlit)
                       /predict · /health · logs           4 Interactive Tabs



Project Structure
├── .github/workflows/
│   └── ci.yml                       # Automated Ruff, Pytest & Docker CI pipeline
├── data/
│   └── churn_data.csv               # Kaggle Telco Customer Churn dataset
├── models/
│   ├── churn_pipeline.pkl           # Serialized calibrated pipeline
│   ├── metrics.json                 # Holdout evaluation metrics
│   ├── oof_predictions.csv          # Out-of-fold validation probabilities
│   └── feature_importances.json     # Holdout mean |SHAP| feature importance shares
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Dataset preparation & cleaning
│   ├── train.py                     # Model training, calibration & holdout evaluation
│   ├── predict.py                   # ChurnRiskEngine: scoring + SHAP attribution
│   └── threshold_optimizer.py       # Cost-based decision threshold & net savings optimizer
├── tests/
│   └── test_api.py                  # Pytest suite for API, validation, DB logging & ROI math
├── app.py                           # FastAPI service with Pydantic guardrails & SQLite logging
├── benchmark_api.py                 # p50/p95/p99 latency & throughput benchmark script
├── ui_app.py                        # Streamlit dashboard (SHAP, What-If, Batch, ROI & Telemetry)
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt

Limitations & Future Improvements
Trained and validated on a single public dataset (Telco); domain adaptation is required before deploying on external telecom cohorts.

Uses SQLite for single-node telemetry logging; high-concurrency production deployments should migrate logs to PostgreSQL or Prometheus/Grafana.

Public endpoints do not enforce API key/JWT authentication yet.

Batch scoring is currently exposed in the Streamlit UI; a POST /predict/batch endpoint is planned for asynchronous bulk scoring.

Author
Abu Rehan · BSIT, The Islamia University of Bahawalpur

[LinkedIn](https://www.linkedin.com/in/abu-rehan-ml) · [GitHub](https://github.com/abu-rehan-rashid)