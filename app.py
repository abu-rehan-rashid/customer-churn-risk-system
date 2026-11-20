import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.predict import ChurnRiskEngine

DB_PATH = Path("logs/predictions.db")

app = FastAPI(
    title="Customer Churn Risk Scoring API",
    description="Production API for real-time customer churn prediction, risk assessment, and drift logging.",
    version="1.1.0",
)

# Initialize Engine at startup
try:
    engine = ChurnRiskEngine()
except Exception:
    engine = None


# --- SQLite Prediction & Drift Logger ---
def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                tenure INTEGER NOT NULL,
                monthly_charges REAL NOT NULL,
                contract_type TEXT NOT NULL,
                prediction_output TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                raw_payload TEXT NOT NULL
            )
            """
        )
        conn.commit()


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


@app.on_event("startup")
def startup_event() -> None:
    init_db()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# --- Strict Pydantic Schema ---
class CustomerPayload(BaseModel):
    gender: Literal["Female", "Male"] = "Female"
    SeniorCitizen: int = Field(0, ge=0, le=1, description="0 for No, 1 for Yes")
    Partner: Literal["Yes", "No"] = "Yes"
    Dependents: Literal["Yes", "No"] = "No"
    tenure: int = Field(1, ge=0, le=120, description="Months with company (0-120)")
    PhoneService: Literal["Yes", "No"] = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = "No"
    OnlineBackup: Literal["Yes", "No", "No internet service"] = "No"
    DeviceProtection: Literal["Yes", "No", "No internet service"] = "No"
    TechSupport: Literal["Yes", "No", "No internet service"] = "No"
    StreamingTV: Literal["Yes", "No", "No internet service"] = "No"
    StreamingMovies: Literal["Yes", "No", "No internet service"] = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: Literal["Yes", "No"] = "Yes"
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = "Electronic check"
    MonthlyCharges: float = Field(70.35, gt=0.0, le=500.0)
    TotalCharges: float | str | None = 70.35


@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "online", "model_loaded": engine is not None}


@app.post("/predict")
def predict_churn_risk(payload: CustomerPayload):
    if not engine:
        raise HTTPException(status_code=500, detail="Model engine not loaded.")

    start_time = time.perf_counter()

    try:
        input_data = pd.DataFrame([payload.model_dump()])
        results = engine.predict_risk(input_data)
        record = results.to_dict(orient="records")[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e!s}")

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    record["latency_ms"] = latency_ms

    # Log request and prediction to SQLite for production drift monitoring
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO prediction_logs (
                timestamp_utc, tenure, monthly_charges, contract_type,
                prediction_output, latency_ms, raw_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                payload.tenure,
                payload.MonthlyCharges,
                payload.Contract,
                json.dumps(record),
                latency_ms,
                json.dumps(payload.model_dump()),
            ),
        )
        conn.commit()

    return record