import os
import warnings

import joblib
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore", category=UserWarning)

REQUIRED_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


class ChurnRiskEngine:
    def __init__(self, model_path: str = "models/churn_pipeline.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact missing at {model_path}. Run train.py first."
            )

        self.pipeline = joblib.load(model_path)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model = self.pipeline.named_steps["model"]
        self._explainer = None

    @property
    def explainer(self):
        if self._explainer is None:
            # Unwrap CalibratedClassifierCV to use fast exact TreeExplainer on the underlying XGBoost model
            if hasattr(self.model, "calibrated_classifiers_") and self.model.calibrated_classifiers_:
                base_est = self.model.calibrated_classifiers_[0].estimator
                self._explainer = shap.TreeExplainer(base_est)
            elif hasattr(self.model, "get_booster"):
                self._explainer = shap.TreeExplainer(self.model.get_booster())
            else:
                self._explainer = shap.Explainer(self.model)
        return self._explainer

    @staticmethod
    def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        cols_to_drop = [c for c in ["customerID", "Churn"] if c in df_clean.columns]
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)

        missing = [c for c in REQUIRED_FEATURES if c not in df_clean.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        for num_col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
            df_clean[num_col] = pd.to_numeric(df_clean[num_col], errors="coerce")
            if df_clean[num_col].isna().any():
                raise ValueError(f"Column '{num_col}' contains non-numeric or missing values.")

        return df_clean[REQUIRED_FEATURES]

    def predict_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = self._prepare_features(df)
        probabilities = self.pipeline.predict_proba(df_clean)[:, 1]

        results = []
        for prob in probabilities:
            if prob >= 0.70:
                risk_level = "High Risk"
                action = "Immediate Retention Offer / Priority Call"
            elif prob >= 0.35:
                risk_level = "Medium Risk"
                action = "Targeted Email Campaign & Feedback Survey"
            else:
                risk_level = "Low Risk"
                action = "No Action Needed"

            results.append(
                {
                    "churn_probability": round(float(prob), 4),
                    "risk_level": risk_level,
                    "recommended_action": action,
                }
            )

        return pd.DataFrame(results)

    def explain_instance(self, input_data: pd.DataFrame, row_index: int = 0) -> pd.DataFrame:
        df_clean = self._prepare_features(input_data)
        if row_index < 0 or row_index >= len(df_clean):
            raise IndexError(f"row_index {row_index} out of bounds for {len(df_clean)} rows.")

        X_transformed = self.preprocessor.transform(df_clean)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        try:
            encoded_feature_names = self.preprocessor.get_feature_names_out()
        except AttributeError:
            encoded_feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        shap_values = self.explainer.shap_values(X_transformed)
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values

        if isinstance(shap_values, list):
            single_shap = (
                shap_values[1][row_index]
                if len(shap_values) > 1
                else shap_values[0][row_index]
            )
        elif len(shap_values.shape) == 3:
            single_shap = (
                shap_values[row_index, :, 1]
                if shap_values.shape[2] > 1
                else shap_values[row_index, :, 0]
            )
        elif len(shap_values.shape) == 2:
            single_shap = shap_values[row_index]
        else:
            single_shap = np.asarray(shap_values)

        raw_columns = df_clean.columns.tolist()
        aggregated_shap = {col: 0.0 for col in raw_columns}

        for feature_name, value in zip(encoded_feature_names, single_shap):
            clean_name = feature_name.replace("num__", "").replace("cat__", "")
            matched = False
            for raw_col in sorted(raw_columns, key=len, reverse=True):
                if clean_name == raw_col or clean_name.startswith(f"{raw_col}_"):
                    aggregated_shap[raw_col] += float(value)
                    matched = True
                    break
            if not matched:
                aggregated_shap[clean_name] = float(value)

        shap_df = pd.DataFrame(list(aggregated_shap.items()), columns=["Feature", "SHAP_Value"])
        shap_df["Abs_Impact"] = shap_df["SHAP_Value"].abs()
        shap_df = (
            shap_df.sort_values(by="Abs_Impact", ascending=True)
            .tail(8)
            .drop(columns=["Abs_Impact"])
        )

        return shap_df