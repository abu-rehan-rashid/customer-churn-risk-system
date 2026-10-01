from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Union
import pandas as pd
from src.predict import ChurnRiskEngine

app = FastAPI(
    title="Customer Churn Risk Scoring API",
    description="Production API for real-time customer churn prediction and risk assessment.",
    version="1.0.0"
)

# Initialize Engine at startup
try:
    engine = ChurnRiskEngine()
except Exception as e:
    engine = None


class CustomerPayload(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.35
    TotalCharges: Optional[Union[float, str]] = 70.35


@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": engine is not None}


@app.post("/predict")
def predict_churn_risk(payload: CustomerPayload):
    if not engine:
        raise HTTPException(status_code=500, detail="Model engine not loaded.")
    
    try:
        input_data = pd.DataFrame([payload.model_dump()])
        results = engine.predict_risk(input_data)
        return results.to_dict(orient="records")[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")