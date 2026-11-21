# 🛡️ Customer Churn Risk Prediction System

![CI Pipeline](https://github.com/abu-rehan-rashid/customer-churn-risk-system/actions/workflows/ci.yml/badge.svg)

Production-style churn prediction system on the IBM Telco Customer Churn dataset (also published on Kaggle), featuring **calibrated XGBoost probabilities**, **SHAP explanations**, a **cost-based ROI threshold optimizer**, a **Streamlit analytics workbench**, and a **FastAPI scoring service** with **SQLite request logging** and **automated CI** (lint, train, test, Docker smoke test).

<p align="center">
  <img width="100%" alt="Real-Time Evaluator & SHAP Explainability" src="https://github.com/user-attachments/assets/9f68b9cc-38a5-4f0e-ae54-a779ed6633b9" />
</p>

<p align="center">
  <img width="48%" alt="What-if Sensitivity Simulator" src="https://github.com/user-attachments/assets/81b1247c-9e8d-4058-b4c2-b6b916b22130" />
  <img width="48%" alt="Model Health & Governance Metrics" src="https://github.com/user-attachments/assets/127108f3-2c41-46b3-9b47-108c022df74f" />
</p>

---

## Highlights

- **Calibrated probabilities and a baseline (`src/train.py`):** `XGBClassifier` wrapped in an isotonic `CalibratedClassifierCV` inside a leak-free scikit-learn `Pipeline`. 5-fold stratified CV ROC-AUC is `0.8463` for the calibrated XGBoost model versus `0.8451` for a Logistic Regression baseline, so the two are statistically very close on this dataset. Out-of-fold Brier score is `0.1348`, and predicted probabilities match observed churn rates within about 2 percentage points per bin.
- **Explainable predictions (SHAP):** SHAP values from an XGBoost `TreeExplainer` are summed from one-hot columns back to the original business features (for example, all `Contract_*` columns become `Contract`). Per-customer explanations are approximate: they explain one of the three XGBoost models inside the calibrated ensemble, not the final averaged probability. Global importances (`models/feature_importances.json`) come from a separate XGBoost model refit on the training split and scored on the holdout.
- **Cost-based threshold optimizer (`src/threshold_optimizer.py`):** uses the saved out-of-fold probabilities (`models/oof_predictions.csv`) to find the cutoff that maximizes net savings for a given offer cost, saved-customer value and campaign save rate. With the dashboard defaults ($50 offer, $600 value, 35% save rate) the optimum is about `0.23`, instead of a fixed `0.50`. These dollar figures depend entirely on the assumptions you enter.
- **Strict Pydantic validation (`app.py`):** all 19 fields are required, unknown fields are rejected (`extra="forbid"`), numeric ranges are enforced (`tenure` 0–120, `MonthlyCharges` > 0 and ≤ 500, `TotalCharges` 0–100,000) and categorical fields are restricted to the known values. Invalid requests return `422`. Prediction failures return `500` with a reference ID and no internal details.
- **SQLite request logging (`logs/predictions.db`):** every `/predict` request is logged with tenure, monthly charges, contract, calibrated probability, risk tier, latency and the raw payload. Logging is best-effort: a failed write is recorded as a warning and never causes the prediction to fail. This is request logging only; drift statistics are not computed yet (see Limitations).
- **Automated tests and CI (`.github/workflows/ci.yml`):** every push and pull request to `main` runs `ruff`, trains the pipeline, runs the `pytest` suite against the real trained model using a temporary database, then builds the Docker image and checks `/health` in a running container.

### Risk tiers

| Risk tier | Probability | Recommended action |
|---|---|---|
| 🔴 High | ≥ 55% | Immediate retention offer / priority call |
| 🟠 Medium | 23% to < 55% | Targeted email campaign and feedback survey |
| 🟢 Low | < 23% | No action needed |

On the out-of-fold predictions (n = 7,043), observed churn was about 70% in the High tier (17% of customers), 37% in the Medium tier (28%) and 8% in the Low tier (55%).

## Model Performance

Dataset: Telco Customer Churn, 7,043 customers, 26.5% base churn rate. Two evaluation sets are used and are kept separate below.

### Out-of-fold (5-fold CV, n = 7,043)

| Metric | Value |
|---|---|
| ROC-AUC | 0.846 |
| PR-AUC (baseline = 0.265) | 0.660 |
| Brier score | 0.1348 |

| Operating point | Customers flagged | Recall | Precision | F1 |
|---|---|---|---|---|
| Cost-optimal cutoff `0.23` | 44.9% | **83.9%** | 49.6% | 0.623 |
| Default cutoff `0.50` | 20.5% | 51.6% | **66.9%** | 0.583 |

### Stratified holdout (20%, n = 1,409, 374 churners), cutoff 0.50

| | Predicted retained | Predicted churned |
|---|---|---|
| **Actually retained** | 938 | 97 |
| **Actually churned** | 176 | 198 |

Recall 52.9% (198 / 374), precision 67.1% (198 / 295), F1 0.592, accuracy 80.6%. With only 374 churners, these holdout recall and precision values carry roughly ±4 percentage points of sampling uncertainty. Exact values can differ slightly between machines and library versions when the model is retrained; `models/metrics.json` always holds the values from the most recent training run.

Because the model is calibrated, a `0.50` cutoff is conservative for this problem: recall at `0.50` is about 52%, while the cost-optimal cutoff recovers about 84% of churners at roughly 50% precision. The cutoff should be chosen from your own offer cost, customer value and save rate. The cutoff here was selected and scored on the same out-of-fold predictions, so the savings estimate is slightly optimistic.

### FastAPI latency (`benchmark_api.py`, 50 sequential requests)

| Metric | Value |
|---|---|
| p50 | 69.75 ms |
| p95 | 109.80 ms |
| p99 | 170.03 ms |
| Throughput | 12.38 req/s |
| Error rate | 0% |

Measured on a local development machine with a single Uvicorn worker, one client, over HTTP, including the SQLite write. The `latency_ms` field in the API response is different: it measures only the model scoring time inside the server, so it is much smaller than the end-to-end figures above. Re-run `python benchmark_api.py` after retraining to refresh these numbers.

## Tech Stack

Python 3.11+, scikit-learn, XGBoost, SHAP, FastAPI, Pydantic, SQLite, Streamlit, Plotly, Pandas, Pytest, Ruff, Docker, GitHub Actions.

## Quick Start

```bash
git clone https://github.com/abu-rehan-rashid/customer-churn-risk-system.git
cd customer-churn-risk-system

python -m venv venv
venv\Scripts\activate          # Windows (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt

# Train and generate artifacts in models/ (the .pkl is not committed to git)
python src/train.py

streamlit run ui_app.py        # Dashboard at http://localhost:8501
uvicorn app:app --reload       # API at http://localhost:8000 (Swagger docs at /docs)
```

Run tests and the latency benchmark:

```bash
python -m pytest -v tests/
python benchmark_api.py        # requires the API running on port 8000
```

Docker (the image trains the model during the build):

```bash
docker build -t churn-risk-api .
docker run -p 8000:8000 churn-risk-api
```

## API Usage

`GET /` and `GET /health` return `{"status": "online", "model_loaded": true}`, or `503` if the model artifact could not be loaded.

`POST /predict` validates all 19 customer fields and returns the calibrated churn probability, risk tier, recommended action and model scoring latency:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
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
```

Response shape (values are illustrative and depend on the trained model):

```json
{
  "churn_probability": 0.65,
  "risk_level": "High Risk",
  "recommended_action": "Immediate Retention Offer / Priority Call",
  "latency_ms": 14.2
}
```

## Architecture

```text
data/churn_data.csv ──> src/train.py ──> models/churn_pipeline.pkl
   (Telco dataset)       (calibration     models/metrics.json
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
                       /predict · /health · logs           4 interactive tabs
```

## Project Structure

```text
├── .github/workflows/
│   └── ci.yml                       # Ruff, train, pytest and Docker smoke test
├── data/
│   └── churn_data.csv               # Telco Customer Churn dataset
├── models/
│   ├── churn_pipeline.pkl           # Generated by train.py (gitignored, not committed)
│   ├── metrics.json                 # Holdout metrics, CV scores, dataset hash, timestamp
│   ├── oof_predictions.csv          # Out-of-fold probabilities used by the ROI optimizer
│   └── feature_importances.json     # Holdout mean |SHAP| importance shares
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Downloads and inspects the raw dataset
│   ├── train.py                     # CV, calibration, holdout evaluation, artifact export
│   ├── predict.py                   # ChurnRiskEngine: scoring, risk tiers, SHAP attribution
│   └── threshold_optimizer.py       # Cost-based decision threshold and net-savings search
├── tests/
│   └── test_api.py                  # API, validation, DB logging, SHAP and ROI tests
├── app.py                           # FastAPI service with Pydantic validation and SQLite logging
├── benchmark_api.py                 # p50/p95/p99 latency and throughput script
├── ui_app.py                        # Streamlit dashboard (SHAP, what-if, batch, ROI, telemetry)
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## Limitations & Future Improvements

- Trained and validated on a single public dataset with no timestamps, so time-based validation and drift behaviour cannot be tested. Adaptation and re-validation are required before using it on another customer base.
- The model ranks customers by churn risk; it does not estimate who would respond to an offer (uplift). Savings figures are what-if estimates based on the assumed offer cost, customer value and save rate.
- SHAP explanations are approximate (see Highlights), and the what-if simulator shows model associations, not causal effects of changing a contract.
- Telemetry is request logging in SQLite. Feature-drift statistics (for example PSI) and alerting are not implemented; high-concurrency deployments should move to PostgreSQL or Prometheus/Grafana.
- Dependencies in `requirements.txt` are not pinned to exact versions.
- No API authentication (API key or JWT) yet.
- Batch scoring is available only in the Streamlit UI; a `POST /predict/batch` endpoint is planned.

## Author

Abu Rehan Rashid · BSIT, The Islamia University of Bahawalpur · [GitHub](https://github.com/abu-rehan-rashid)