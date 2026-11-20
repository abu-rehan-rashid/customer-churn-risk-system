# 🛡️ Customer Churn Risk Prediction System

![CI Pipeline](https://github.com/abu-rehan-rashid/customer-churn-risk-system/actions/workflows/ci.yml/badge.svg)

Production-grade churn prediction system on the Kaggle Telco dataset featuring **SHAP explainability**, a **cost-based business ROI threshold optimizer**, a **Streamlit analytics workbench**, and a **FastAPI scoring service** with **SQLite drift/latency telemetry** and **automated CI/CD**.

<p align="center">
  <img width="100%" alt="Real-Time Evaluator & SHAP Explainability" src="https://github.com/user-attachments/assets/9f68b9cc-38a5-4f0e-ae54-a779ed6633b9" />
</p>

<p align="center">
  <img width="48%" alt="What-if Sensitivity Simulator" src="https://github.com/user-attachments/assets/81b1247c-9e8d-4058-b4c2-b6b916b22130" />
  <img width="48%" alt="Model Health & Governance Metrics" src="https://github.com/user-attachments/assets/127108f3-2c41-46b3-9b47-108c022df74f" />
</p>

---

## Highlights & Production Engineering

- **Explainable Predictions:** SHAP values (log-odds, XGBoost `TreeExplainer`) are summed from one-hot columns back to original business features (e.g., all `Contract_*` columns → `Contract`).
- **Cost-Based Threshold Optimizer (`src/threshold_optimizer.py`):** Dynamically calculates the probability cutoff that maximizes net USD retention savings based on offer cost, customer LTV, and campaign save rate instead of relying on a static `0.50` threshold.
- **Strict Pydantic Guardrails (`app.py`):** Enforces domain bounds (`tenure >= 0`, `MonthlyCharges >= 0`) and literal contract/service categories, rejecting malformed inputs with `422 Unprocessable Entity`.
- **SQLite Inference & Drift Telemetry (`logs/predictions.db`):** Logs every `/predict` payload, predicted probability, risk tier, and real-time inference latency (`latency_ms`) for production monitoring.
- **Automated Testing & CI/CD (`.github/workflows/ci.yml`):** Every push runs `ruff` linting, `pytest` unit/integration tests, and a `docker build` verification on GitHub Actions.

| Risk tier | Probability | Recommended action |
|---|---|---|
| 🔴 High | ≥ 70% | Immediate retention offer / priority call |
| 🟠 Medium | 35% to 69% | Targeted email campaign and feedback survey |
| 🟢 Low | < 35% | No action needed |

## Model Performance & API Benchmarks

Evaluated on a holdout set of 1,409 customers (Kaggle Telco Customer Churn).

| ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|
| 0.846 | 80.2% | 52.5% | 0.635 |

| | Predicted retained | Predicted churned |
|---|---|---|
| **Actually retained** | 764 | 271 |
| **Actually churned** | 74 | 300 |

### FastAPI Service Latency (`benchmark_api.py`, 50 Sequential Requests)

| Metric | Value | Notes |
|---|---|---|
| **p50 Latency** | **69.75 ms** | Includes preprocessing, XGBoost inference & SQLite telemetry write |
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

# Generate trained model artifacts (models/churn_pipeline.pkl & models/metrics.json)
python src/train.py

streamlit run ui_app.py        # Dashboard at http://localhost:8501
uvicorn app:app --reload       # API at http://localhost:8000 (Swagger docs at /docs)
```

**Run Tests & Latency Benchmark:**

```bash
python -m pytest -v tests/
python benchmark_api.py        # Requires uvicorn running on port 8000
```

**Docker:**

```bash
docker compose up --build
```

## API Usage

`GET /` returns `{"status": "online", "model_loaded": true}`.

`POST /predict` validates inputs against strict Pydantic schemas and returns risk tier, action, and latency:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 6, "Contract": "Month-to-month", "InternetService": "Fiber optic", "MonthlyCharges": 85.0}'
```

Response shape:

```json
{
  "churn_probability": 0.9131,
  "risk_level": "High Risk",
  "recommended_action": "Immediate Retention Offer / Priority Call",
  "latency_ms": 69.75
}
```

## Architecture

```text
data/churn_data.csv ──> src/train.py ──> models/churn_pipeline.pkl
   (Kaggle Telco)        (preprocessing      models/metrics.json
                          + XGBoost)                │
                                                    ▼
                                  src/predict.py & threshold_optimizer.py
                                  predict_risk() · explain_instance() · ROI
                                                    │
                                  ┌─────────────────┴─────────────────┐
                                  ▼                                   ▼
                       app.py (FastAPI + SQLite)           ui_app.py (Streamlit)
                       /predict · /health · logs           4 Interactive Tabs
```

## Project Structure

```text
├── .github/workflows/
│   └── ci.yml                  # Automated Ruff, Pytest & Docker CI pipeline
├── data/
│   └── churn_data.csv          # Kaggle Telco Customer Churn dataset
├── models/
│   └── metrics.json            # Holdout evaluation metrics
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset preparation & cleaning
│   ├── train.py                # Model training & holdout evaluation
│   ├── predict.py              # ChurnRiskEngine: scoring + SHAP attribution
│   └── threshold_optimizer.py  # Cost-based decision threshold & net savings optimizer
├── tests/
│   └── test_api.py             # Pytest suite for API, validation, DB logging & ROI math
├── app.py                      # FastAPI service with Pydantic guardrails & SQLite logging
├── benchmark_api.py            # p50/p95/p99 latency & throughput benchmark script
├── ui_app.py                   # Streamlit dashboard (SHAP, What-If, Batch, ROI & Telemetry)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## Limitations & Future Improvements

- Trained and validated on a single public dataset (Telco); domain adaptation is required before deploying on external telecom cohorts.
- Uses SQLite for single-node telemetry logging; high-concurrency production deployments should migrate logs to PostgreSQL or Prometheus/Grafana.
- Public endpoints do not enforce API key/JWT authentication yet.
- Batch scoring is currently exposed in the Streamlit UI; a `POST /predict/batch` endpoint is planned for asynchronous bulk scoring.

## Author

**Abu Rehan** · BSIT, The Islamia University of Bahawalpur  
[LinkedIn](https://www.linkedin.com/in/abu-rehan-ml) · [GitHub](https://github.com/abu-rehan-rashid)