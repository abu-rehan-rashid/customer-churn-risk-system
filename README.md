# 🛡️ Customer Churn Risk Prediction System

Churn prediction on the Kaggle Telco dataset with **SHAP explainability**, a **Streamlit dashboard** and a **FastAPI scoring endpoint**. Given a customer's profile, it returns a churn probability, a risk tier, a recommended retention action, and which features pushed the risk up or down.

<p align="center">
  <img width="100%" height="832" alt="Real-Time Evaluator & SHAP Explainability" src="https://github.com/user-attachments/assets/9f68b9cc-38a5-4f0e-ae54-a779ed6633b9" />
</p>

<p align="center">
  <img width="48%" height="834" alt="What-if Sensitivity Simulator" src="https://github.com/user-attachments/assets/81b1247c-9e8d-4058-b4c2-b6b916b22130" />
  <img width="48%" height="783" alt="Model Health & Governance Metrics" src="https://github.com/user-attachments/assets/127108f3-2c41-46b3-9b47-108c022df74f" />
</p>

---

## Highlights

- **Explainable predictions:** SHAP values (log-odds, XGBoost TreeExplainer) are summed from one-hot columns back to the original features (e.g. all `Contract_*` columns → `Contract`), so the chart reads in business terms.
- **Risk tiers with actions:** each probability maps to a tier and a suggested action (see table below).
- **What-if simulator:** vary tenure and see how churn probability changes.
- **Batch scoring:** upload a CSV in the dashboard and score every row.
- **REST API:** `POST /predict` with Pydantic validation and auto-generated docs at `/docs`.

| Risk tier | Probability | Recommended action |
|---|---|---|
| 🔴 High | ≥ 70% | Immediate retention offer / priority call |
| 🟠 Medium | 35% to 69% | Targeted email campaign and feedback survey |
| 🟢 Low | < 35% | No action needed |

## Model Performance

Evaluated on a holdout set of 1,409 customers (Kaggle Telco Customer Churn).

| ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|
| 0.846 | 80.2% | 52.5% | 0.635 |

| | Predicted retained | Predicted churned |
|---|---|---|
| **Actually retained** | 764 | 271 |
| **Actually churned** | 74 | 300 |

**Reading the trade-off:** the model catches 300 of 374 churners (80.2%) but also raises 271 false alarms (precision 52.5%). That suits cases where a retention offer is cheap compared with losing a customer. If offers are expensive, a higher decision threshold would trade recall for precision.

## Tech Stack

Python 3.10+, scikit-learn, XGBoost, SHAP, FastAPI, Pydantic, Streamlit, Plotly, Pandas, Docker.

## Quick Start

```bash
git clone https://github.com/abu-rehan-rashid/customer-churn-risk-system.git
cd customer-churn-risk-system

python -m venv venv
venv\Scripts\activate          # Windows (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt

# The trained model is not committed, so generate it first
python src/train.py            # creates models/churn_pipeline.pkl and models/metrics.json

streamlit run ui_app.py        # dashboard at http://localhost:8501
uvicorn app:app --reload       # API at http://localhost:8000 (docs at /docs)
```

**Docker:**

```bash
docker compose up --build
```

## API Usage

`GET /` returns `{"status": "online", "model_loaded": true}`.

`POST /predict` accepts the customer fields (all have defaults, so you can send only the ones you care about):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 6, "Contract": "Month-to-month", "InternetService": "Fiber optic", "MonthlyCharges": 85.0}'
```

Response shape:

```json
{
  "churn_probability": 0.8690,
  "risk_level": "High Risk",
  "recommended_action": "Immediate Retention Offer / Priority Call"
}
```

## Architecture

```text
data/churn_data.csv ──> src/train.py ──> models/churn_pipeline.pkl
   (Kaggle Telco)        (preprocessing      models/metrics.json
                          + XGBoost)                │
                                                    ▼
                                         src/predict.py (ChurnRiskEngine)
                                         predict_risk() · explain_instance()
                                                    │
                                  ┌─────────────────┴─────────────────┐
                                  ▼                                   ▼
                       app.py (FastAPI)                    ui_app.py (Streamlit)
```

## Project Structure

```text
├── app.py                  # FastAPI service
├── ui_app.py               # Streamlit dashboard (4 tabs)
├── src/
│   ├── data_loader.py      # dataset preparation
│   ├── train.py            # training + evaluation
│   └── predict.py          # ChurnRiskEngine: scoring + SHAP
├── data/churn_data.csv
├── models/metrics.json
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Limitations

- Trained and validated on a single public dataset (Telco); not tested on other customer data.
- The API has no authentication, and missing fields silently fall back to default values.
- No automated tests yet.
- Batch scoring is available in the dashboard only, not as an API endpoint.

## Author

**Abu Rehan** · BSIT, The Islamia University of Bahawalpur
[LinkedIn](https://www.linkedin.com/in/abu-rehan-ml) · [GitHub](https://github.com/abu-rehan-rashid)
