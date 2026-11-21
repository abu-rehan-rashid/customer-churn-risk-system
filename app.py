import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.predict import ChurnRiskEngine

logger = logging.getLogger("churn_api")
logging.basicConfig(level=logging.INFO)


def get_db_path() -> Path:
    return Path(os.getenv("PREDICTIONS_DB_PATH", "logs/predictions.db"))


def init_db() -> None:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                tenure INTEGER NOT NULL,
                monthly_charges REAL NOT NULL,
                contract_type TEXT NOT NULL,
                churn_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                prediction_output TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                raw_payload TEXT NOT NULL
            )
            """
        )
        conn.commit()


@contextmanager
def get_db():
    conn = sqlite3.connect(get_db_path())
    try:
        yield conn
    finally:
        conn.close()


try:
    engine: ChurnRiskEngine | None = ChurnRiskEngine()
except Exception as exc:
    logger.exception(
        "Failed to initialize ChurnRiskEngine at startup: %s", exc)
    engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Customer Churn Risk Scoring API",
    description="Scoring API for real-time customer churn prediction, risk assessment, and request logging.",
    version="1.2.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_id = uuid.uuid4().hex[:8]
    logger.exception("Unhandled server error [%s]: %s", error_id, exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "reference_id": error_id},
    )


class CustomerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(..., ge=0, le=1,
                               description="0 for No, 1 for Yes")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=120,
                        description="Months with company (0-120)")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., gt=0.0, le=500.0)
    TotalCharges: float = Field(..., ge=0.0, le=100000.0)


@app.get("/")
@app.get("/health")
def health_check():
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "model_loaded": False},
        )
    return {"status": "online", "model_loaded": True}


@app.post("/predict")
def predict_churn_risk(payload: CustomerPayload):
    if engine is None:
        raise HTTPException(
            status_code=503, detail="Model engine unavailable.")

    start_time = time.perf_counter()

    try:
        input_data = pd.DataFrame([payload.model_dump()])
        results = engine.predict_risk(input_data)
        record = results.to_dict(orient="records")[0]
    except Exception:
        error_id = uuid.uuid4().hex[:8]
        logger.exception("Prediction failure [%s]", error_id)
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed (ref: {error_id}).",
        )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    record["latency_ms"] = latency_ms

    # Best-effort telemetry write (never drops a valid prediction if DB write fails)
    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO prediction_logs (
                    timestamp_utc, tenure, monthly_charges, contract_type,
                    churn_probability, risk_level, prediction_output, latency_ms, raw_payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    payload.tenure,
                    payload.MonthlyCharges,
                    payload.Contract,
                    float(record["churn_probability"]),
                    str(record["risk_level"]),
                    json.dumps(record),
                    latency_ms,
                    json.dumps(payload.model_dump()),
                ),
            )
            conn.commit()
    except Exception as db_exc:
        logger.warning("Failed to write prediction telemetry: %s", db_exc)

    return record
