import os
import warnings
import joblib
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings('ignore', category=UserWarning)


class ChurnRiskEngine:
    def __init__(self, model_path: str = "models/churn_pipeline.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact missing at {model_path}. Run train.py first.")

        self.pipeline = joblib.load(model_path)
        self.preprocessor = self.pipeline.named_steps['preprocessor']
        self.model = self.pipeline.named_steps['model']
        self._explainer = None

    @property
    def explainer(self):
        if self._explainer is None:
            try:
                booster = self.model.get_booster()
                self._explainer = shap.TreeExplainer(booster)
            except Exception:
                self._explainer = shap.Explainer(self.model)
        return self._explainer

    def predict_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        
        # Drop non-feature identifiers if present
        cols_to_drop = [c for c in ['customerID', 'Churn'] if c in df_clean.columns]
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)

        # Clean numeric columns to prevent NaN/String errors
        for num_col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            if num_col in df_clean.columns:
                df_clean[num_col] = pd.to_numeric(
                    df_clean[num_col], errors='coerce'
                ).fillna(0)

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

            results.append({
                "churn_probability": round(float(prob), 4),
                "risk_level": risk_level,
                "recommended_action": action
            })

        return pd.DataFrame(results)

    def explain_instance(self, input_data: pd.DataFrame) -> pd.DataFrame:
        df_clean = input_data.copy()
        
        cols_to_drop = [c for c in ['customerID', 'Churn'] if c in df_clean.columns]
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)

        for num_col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            if num_col in df_clean.columns:
                df_clean[num_col] = pd.to_numeric(
                    df_clean[num_col], errors='coerce'
                ).fillna(0)

        # 1. Preprocess features
        X_transformed = self.preprocessor.transform(df_clean)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        # 2. Extract feature names
        try:
            encoded_feature_names = self.preprocessor.get_feature_names_out()
        except AttributeError:
            encoded_feature_names = [
                f"feature_{i}" for i in range(X_transformed.shape[1])]

        # 3. Calculate SHAP values
        try:
            shap_values = self.explainer.shap_values(X_transformed)
        except Exception:
            explainer = shap.Explainer(self.model)
            shap_values = explainer(X_transformed)

        if hasattr(shap_values, 'values'):
            shap_values = shap_values.values

        if isinstance(shap_values, list):
            single_shap = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 3:
            single_shap = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
        elif len(shap_values.shape) == 2:
            single_shap = shap_values[0]
        else:
            single_shap = shap_values

        # 4. Aggregate feature contributions back to raw input columns
        raw_columns = df_clean.columns.tolist()
        aggregated_shap = {col: 0.0 for col in raw_columns}

        for feature_name, value in zip(encoded_feature_names, single_shap):
            clean_name = feature_name.replace('num__', '').replace('cat__', '')
            matched = False
            for raw_col in sorted(raw_columns, key=len, reverse=True):
                if clean_name == raw_col or clean_name.startswith(f"{raw_col}_"):
                    aggregated_shap[raw_col] += float(value)
                    matched = True
                    break
            if not matched:
                aggregated_shap[clean_name] = float(value)

        # 5. Output DataFrame
        shap_df = pd.DataFrame(list(aggregated_shap.items()), columns=['Feature', 'SHAP_Value'])
        shap_df['Abs_Impact'] = shap_df['SHAP_Value'].abs()
        shap_df = shap_df.sort_values(by='Abs_Impact', ascending=True).tail(8).drop(columns=['Abs_Impact'])

        return shap_df