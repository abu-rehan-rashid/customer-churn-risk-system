# 🛡️ Enterprise Customer Churn Intelligence & Explainability Platform

An end-to-end Machine Learning system engineered to predict customer churn risk, deliver explainable AI (XAI) feature attributions using SHAP, run interactive sensitivity simulations, execute batch scoring, and serve real-time predictions via FastAPI and Streamlit.

---

## 📌 Executive Summary
* **Business Purpose:** Identify high-risk subscription accounts prior to churn to deploy targeted retention offers and reduce revenue loss.
* **Tech Stack:** Python 3.10, Scikit-Learn, XGBoost, SHAP, FastAPI, Streamlit, Plotly, Pandas.
* **Key Innovation:** Solves One-Hot Encoding interpretability degradation by aggregating SHAP log-odds contributions back to raw categorical features for clear business stakeholder insight.
* **Performance Benchmark:** **0.846 ROC-AUC**, **80.2% Recall**, **52.5% Precision**, and **0.635 F1-Score** on holdout test evaluation.

---

## ⭐ Key Features

* **Real-Time Churn Risk Scoring:** Categorizes accounts into actionable risk tiers based on model probability output:
  * 🔴 **High Risk ($\ge 70\%$):** Immediate Retention Offer / Priority Call
  * 🟠 **Medium Risk ($35\%\text{--}69\%$):** Targeted Email Campaign & Feedback Survey
  * 🟢 **Low Risk ($< 35\%$):** No Action Needed
* **Mathematical SHAP Attribution:** Re-aggregates encoded dummy variables back to parent columns, displaying exact directionality (Red = Risk Increase, Green = Retention Support).
* **Interactive Sensitivity Simulator:** Allows account managers to run "what-if" scenarios across variable tenure ranges to project churn risk trajectories in real time.
* **Enterprise Batch Scoring Engine:** Processes multi-record CSV files, generating row-level risk predictions and visual distribution breakdowns.
* **Production REST API:** Serves validated prediction payloads via FastAPI with Pydantic request modeling and OpenAPI documentation.
* **Model Governance & Monitoring:** Displays holdout confusion matrices, cross-validation metrics, and global feature importance metrics inside a centralized dashboard.

---

## 🏗️ System Architecture & Data Flow

```text
[ IBM / Kaggle Dataset ] ──> [ src/data_loader.py ] ──> [ data/churn_data.csv ]
                                                               │
                                                               ▼
[ models/metrics.json ] <── [ src/train.py ] <─── [ Preprocessing & XGBoost ]
[ models/churn_pipeline.pkl ]                                  │
          │                                                    │
          └─────────────────────┬──────────────────────────────┘
                                │
                                ▼
                     [ src/predict.py ]
                     (ChurnRiskEngine)
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
  [ REST API: app.py ]                       [ UI: ui_app.py ]
  (FastAPI Endpoint)                        (Streamlit Dashboard)