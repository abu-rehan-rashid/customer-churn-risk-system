import os
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_fscore_support


def train():
    data_path = "data/churn_data.csv"
    model_path = "models/churn_pipeline.pkl"
    metrics_path = "models/metrics.json"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file missing at {data_path}. Run data_loader.py first.")

    df = pd.read_csv(data_path)

    # Clean data types
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Target encoding
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['Churn', 'customerID'], errors='ignore')

    # Separate numeric and categorical features
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Preprocessor (OHE without drop='first' for clean SHAP feature attribution)
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

    # Calculate class imbalance ratio
    imbalance_ratio = float(sum(y == 0) / sum(y == 1))

    # Full Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            scale_pos_weight=imbalance_ratio,
            random_state=42,
            eval_metric='logloss'
        ))
    ])

    # 1. Stratified K-Fold Cross Validation
    print("Running 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=['roc_auc', 'recall', 'precision', 'f1'])

    print(f"Mean CV ROC-AUC:   {np.mean(cv_results['test_roc_auc']):.4f}")
    print(f"Mean CV Recall:    {np.mean(cv_results['test_recall']):.4f}")
    print(f"Mean CV Precision: {np.mean(cv_results['test_precision']):.4f}")

    # 2. Train / Test Split for Holdout Governance Benchmarking
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Fitting model on training set...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = float(roc_auc_score(y_test, y_prob))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"Holdout Test ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save Pipeline & Export Governance Metrics JSON
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, model_path)
    
    metrics_data = {
        "roc_auc": round(roc_auc, 3),
        "recall": round(float(recall), 3),
        "precision": round(float(precision), 3),
        "f1_score": round(float(f1), 3),
        "confusion_matrix": cm
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"Model saved to {model_path}")
    print(f"Metrics metadata saved to {metrics_path}")


if __name__ == "__main__":
    train()