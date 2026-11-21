import hashlib
import json
import os
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
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
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def _compute_real_shap_importances(
    xgb_pipeline: Pipeline, X_sample: pd.DataFrame
) -> list[dict[str, float | str]]:
    """Compute real mean(|SHAP|) share aggregated back to raw business features."""
    preprocessor = xgb_pipeline.named_steps["preprocessor"]
    model = xgb_pipeline.named_steps["model"]

    X_trans = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_trans)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    mean_abs = np.abs(shap_vals).mean(axis=0)
    raw_totals: dict[str, float] = {feat: 0.0 for feat in ALL_FEATURES}

    for enc_name, val in zip(feature_names, mean_abs):
        clean_name = enc_name.split("__", 1)[-1]
        matched = False
        for raw_feat in ALL_FEATURES:
            if clean_name == raw_feat or clean_name.startswith(f"{raw_feat}_"):
                raw_totals[raw_feat] += float(val)
                matched = True
                break
        if not matched:
            raw_totals[clean_name] = raw_totals.get(clean_name, 0.0) + float(val)

    total_sum = sum(raw_totals.values()) or 1.0
    ranked = sorted(
        [
            {"Feature": feat, "Importance Weight": round(val / total_sum, 4)}
            for feat, val in raw_totals.items()
        ],
        key=lambda item: item["Importance Weight"],
        reverse=True,
    )
    return ranked


def train(data_path: str = "data/churn_data.csv", models_dir: str = "models") -> dict:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file missing at {data_path}. Run data_loader.py first.")

    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    y = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    X = df[ALL_FEATURES].copy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Baseline comparison: Logistic Regression vs Calibrated XGBoost
    lr_pipeline = Pipeline(
        [
            ("preprocessor", _build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    lr_cv = cross_validate(lr_pipeline, X, y, cv=cv, scoring=["roc_auc", "average_precision"])

    base_xgb = XGBClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="logloss",
    )
    calibrated_pipeline = Pipeline(
        [
            ("preprocessor", _build_preprocessor()),
            ("model", CalibratedClassifierCV(estimator=base_xgb, method="isotonic", cv=3)),
        ]
    )

    print("Running 5-Fold Stratified Cross-Validation...")
    xgb_cv = cross_validate(
        calibrated_pipeline, X, y, cv=cv, scoring=["roc_auc", "average_precision", "neg_brier_score"]
    )
    oof_probs = cross_val_predict(calibrated_pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    cv_roc_auc = float(np.mean(xgb_cv["test_roc_auc"]))
    cv_pr_auc = float(np.mean(xgb_cv["test_average_precision"]))
    cv_brier = float(-np.mean(xgb_cv["test_neg_brier_score"]))
    lr_cv_auc = float(np.mean(lr_cv["test_roc_auc"]))

    print(f"5-Fold CV ROC-AUC (Calibrated XGBoost): {cv_roc_auc:.4f} | Baseline LR: {lr_cv_auc:.4f}")
    print(f"5-Fold CV PR-AUC: {cv_pr_auc:.4f} | Brier Score: {cv_brier:.4f}")

    # 2. Train / Test Split for Holdout Benchmarking
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    calibrated_pipeline.fit(X_train, y_train)
    y_pred = calibrated_pipeline.predict(X_test)
    y_prob = calibrated_pipeline.predict_proba(X_test)[:, 1]

    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Fit an uncalibrated reference XGB pipeline on X_train strictly for exact TreeExplainer SHAP shares
    shap_ref_pipeline = Pipeline(
        [
            ("preprocessor", _build_preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=120,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    eval_metric="logloss",
                ),
            ),
        ]
    )
    shap_ref_pipeline.fit(X_train, y_train)
    real_importances = _compute_real_shap_importances(shap_ref_pipeline, X_test)

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "churn_pipeline.pkl")
    metrics_path = os.path.join(models_dir, "metrics.json")
    oof_path = os.path.join(models_dir, "oof_predictions.csv")
    imp_path = os.path.join(models_dir, "feature_importances.json")

    joblib.dump(calibrated_pipeline, model_path)

    # Save real out-of-fold probabilities for cost-based threshold optimization
    pd.DataFrame(
        {
            "y_true": y.values,
            "y_prob": np.round(oof_probs, 6),
            "MonthlyCharges": df["MonthlyCharges"].values,
        }
    ).to_csv(oof_path, index=False)

    with open(imp_path, "w") as f:
        json.dump(real_importances, f, indent=4)

    data_hash = hashlib.sha256(open(data_path, "rb").read()).hexdigest()[:16]
    metrics_data = {
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "dataset_sha256": data_hash,
        "roc_auc": round(roc_auc, 3),
        "pr_auc": round(pr_auc, 3),
        "brier_score": round(brier, 4),
        "recall": round(float(recall), 3),
        "precision": round(float(precision), 3),
        "f1_score": round(float(f1), 3),
        "cv_roc_auc_xgb": round(cv_roc_auc, 4),
        "cv_roc_auc_logreg_baseline": round(lr_cv_auc, 4),
        "confusion_matrix": cm,
        "features": ALL_FEATURES,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Artifacts saved to {models_dir}/ (pipeline, metrics.json, oof_predictions.csv, feature_importances.json)")
    return metrics_data


if __name__ == "__main__":
    train()