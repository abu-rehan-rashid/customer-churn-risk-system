import json
import os
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.predict import ChurnRiskEngine
from src.threshold_optimizer import find_optimal_threshold

# Page Config
st.set_page_config(
    page_title="Enterprise Churn & Revenue Retention Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Executive CSS for crisp 100% zoom fit, single-line header, and clean KPI cards
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 0.8rem;
        max-width: 96%;
    }
    h1 {
        font-size: 1.75rem !important;
        padding-bottom: 0.15rem !important;
    }
    section[data-testid="stSidebar"] {
        width: 275px !important;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.10);
        padding: 8px 12px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_engine():
    return ChurnRiskEngine()


try:
    engine = load_engine()
except Exception as e:
    st.error(f"Failed to load ML Engine: {e}. Please run 'python src/train.py' first.")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR: Customer Account Profile Inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📋 Customer Profile")
    st.caption("Adjust account attributes to score churn & financial exposure:")

    with st.form("customer_input_form"):
        tenure = st.slider("Tenure (Months)", 1, 72, 6)
        contract = st.selectbox(
            "Contract Type", ["Month-to-month", "One year", "Two year"]
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", 18.0, 150.0, 85.0, step=1.0
        )
        internet_service = st.selectbox(
            "Internet Service", ["Fiber optic", "DSL", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        tech_support = st.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"]
        )
        online_security = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"]
        )

        with st.expander("More Services & Demographics"):
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            gender = st.selectbox("Gender", ["Female", "Male"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["No", "Yes", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["No", "Yes", "No internet service"]
            )

        total_charges = tenure * monthly_charges
        st.form_submit_button("⚡ Evaluate Account", use_container_width=True)

payload = pd.DataFrame(
    [
        {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
    ]
)

# Header
st.title("🛡️ Enterprise Churn & Revenue Retention Platform")
st.caption(
    "Calibrated Churn Scoring · Mathematical SHAP Attribution · Cost-Based ROI Optimization · Live API Telemetry"
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎯 Executive Risk & Revenue",
        "🧪 Retention Simulator",
        "📁 Batch Scoring & Budget",
        "📊 ROI Optimizer & Telemetry",
    ]
)

# -----------------------------------------------------------------------------
# TAB 1: Executive Risk & Revenue Evaluator
# -----------------------------------------------------------------------------
with tab1:
    res = engine.predict_risk(payload).iloc[0]
    prob = float(res["churn_probability"])
    risk_level = res["risk_level"]
    action = res["recommended_action"]

    annual_arr = monthly_charges * 12.0
    expected_revenue_loss = annual_arr * prob
    max_retention_budget = expected_revenue_loss * 0.25

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Churn Probability", f"{prob * 100:.1f}%", delta=risk_level, delta_color="inverse")
    k2.metric("Annual Contract Value (ARR)", f"${annual_arr:,.0f}")
    k3.metric("Expected Revenue at Risk", f"${expected_revenue_loss:,.0f}")
    k4.metric("Recommended Offer Cap", f"${max_retention_budget:,.0f}")

    color = (
        "#FF4B4B"
        if risk_level == "High Risk"
        else ("#FFA500" if risk_level == "Medium Risk" else "#00CC96")
    )

    col_left, col_right = st.columns([1, 1.45], gap="medium")

    with col_left:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 32}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 23], "color": "rgba(0, 204, 150, 0.15)"},
                        {"range": [23, 55], "color": "rgba(255, 165, 0, 0.15)"},
                        {"range": [55, 100], "color": "rgba(255, 75, 75, 0.15)"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(height=180, margin={"l": 15, "r": 15, "t": 15, "b": 5})
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(
            f"""
            <div style="background-color: {color}18; border-left: 5px solid {color}; padding: 12px 14px; border-radius: 8px;">
                <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.7px; color:#A0A0A0;">Executive Action Playbook</div>
                <div style="font-size:19px; font-weight:700; color:{color}; margin: 2px 0;">{risk_level}</div>
                <div style="font-size:13px; color:#EAEAEA;">{action}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("**🔍 Key Churn & Retention Drivers (SHAP Attribution)**")
        try:
            shap_df = engine.explain_instance(payload)
        except Exception as exc:
            st.warning(f"SHAP attribution unavailable: {exc}")
            shap_df = None

        if shap_df is not None and not shap_df.empty and "SHAP_Value" in shap_df.columns:
            colors = [
                "#FF4B4B" if val >= 0 else "#00CC96" for val in shap_df["SHAP_Value"]
            ]
            fig_shap = go.Figure(
                go.Bar(
                    x=shap_df["SHAP_Value"],
                    y=shap_df["Feature"],
                    orientation="h",
                    marker={"color": colors},
                    width=0.55,
                )
            )
            fig_shap.update_layout(
                height=290,
                showlegend=False,
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
                xaxis={
                    "title": "SHAP Log-Odds Impact (Red = Risk | Green = Retention)",
                    "zeroline": True,
                    "zerolinewidth": 2,
                    "zerolinecolor": "white",
                    "showgrid": True,
                    "gridcolor": "rgba(255, 255, 255, 0.12)",
                },
                yaxis={
                    "type": "category",
                    "categoryorder": "array",
                    "categoryarray": shap_df["Feature"].tolist(),
                },
            )
            st.plotly_chart(fig_shap, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: Retention Intervention Simulator (Observational What-If)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown(
        "Observational model what-if comparing predicted churn trajectories across **Month-to-month**, **One year**, and **Two year** contracts over tenure."
    )
    sim_range = st.slider("Tenure Horizon (Months)", 1, 72, (1, 36))
    tenures = list(range(sim_range[0], sim_range[1] + 1))

    batch_sim_rows = []
    for contract_opt in ["Month-to-month", "One year", "Two year"]:
        for t in tenures:
            row = payload.iloc[0].to_dict()
            row["Contract"] = contract_opt
            row["tenure"] = t
            row["TotalCharges"] = t * row["MonthlyCharges"]
            batch_sim_rows.append(row)

    sim_input_df = pd.DataFrame(batch_sim_rows)
    sim_preds = engine.predict_risk(sim_input_df)
    sens_df = pd.DataFrame(
        {
            "Tenure (Months)": sim_input_df["tenure"],
            "Contract Type": sim_input_df["Contract"],
            "Churn Probability (%)": sim_preds["churn_probability"] * 100.0,
        }
    )

    fig_sens = px.line(
        sens_df,
        x="Tenure (Months)",
        y="Churn Probability (%)",
        color="Contract Type",
        title="Model Churn Trajectory by Contract Lock-In Profile",
        color_discrete_map={
            "Month-to-month": "#FF4B4B",
            "One year": "#FFA500",
            "Two year": "#00CC96",
        },
    )
    fig_sens.add_hline(
        y=55, line_dash="dash", line_color="red", annotation_text="High Risk Cutoff (55%)"
    )
    fig_sens.update_layout(height=340, margin={"l": 15, "r": 15, "t": 40, "b": 15})
    st.plotly_chart(fig_sens, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: Portfolio Batch Scoring & Budget Planner
# -----------------------------------------------------------------------------
with tab3:
    uploaded_file = st.file_uploader("Upload Customer Cohort (CSV)", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        if batch_df.empty:
            st.warning("Uploaded CSV file is empty.")
        else:
            if st.button("🚀 Run Portfolio Risk & Budget Analysis"):
                try:
                    with st.spinner("Scoring portfolio..."):
                        results_df = engine.predict_risk(batch_df)
                        final_batch = pd.concat(
                            [
                                batch_df.reset_index(drop=True),
                                results_df.reset_index(drop=True),
                            ],
                            axis=1,
                        )

                        high_risk_df = final_batch[final_batch["risk_level"] == "High Risk"]
                        arr_at_risk = (
                            high_risk_df["MonthlyCharges"].sum() * 12.0
                            if "MonthlyCharges" in final_batch.columns
                            else len(high_risk_df) * 600.0
                        )
                        est_campaign_cost = len(high_risk_df) * 50.0

                        b1, b2, b3, b4 = st.columns(4)
                        b1.metric("Total Accounts Scored", f"{len(final_batch):,}")
                        b2.metric("High-Risk Accounts", f"{len(high_risk_df):,}")
                        b3.metric("High-Risk Annual ARR", f"${arr_at_risk:,.0f}")
                        b4.metric("Est. Retention Budget", f"${est_campaign_cost:,.0f}")

                        c_tbl, c_pie = st.columns([1.4, 1])
                        with c_tbl:
                            cols_to_show = [
                                c
                                for c in [
                                    "customerID",
                                    "tenure",
                                    "Contract",
                                    "MonthlyCharges",
                                    "churn_probability",
                                    "risk_level",
                                ]
                                if c in final_batch.columns
                            ]
                            st.dataframe(final_batch[cols_to_show].head(12), use_container_width=True)
                            st.download_button(
                                "⬇️ Download Scored Portfolio CSV",
                                data=final_batch.to_csv(index=False).encode("utf-8"),
                                file_name="scored_churn_portfolio.csv",
                                mime="text/csv",
                            )
                        with c_pie:
                            fig_pie = px.pie(
                                final_batch,
                                names="risk_level",
                                title="Risk Tier Breakdown",
                                color="risk_level",
                                color_discrete_map={
                                    "High Risk": "#FF4B4B",
                                    "Medium Risk": "#FFA500",
                                    "Low Risk": "#00CC96",
                                },
                            )
                            fig_pie.update_layout(height=310, margin={"l": 10, "r": 10, "t": 35, "b": 10})
                            st.plotly_chart(fig_pie, use_container_width=True)
                except Exception as exc:
                    st.error(f"Batch scoring failed: {exc}")


# -----------------------------------------------------------------------------
# TAB 4: Financial ROI Optimizer & Live Telemetry
# -----------------------------------------------------------------------------
with tab4:
    roi_c1, roi_c2, roi_c3 = st.columns(3)
    with roi_c1:
        retention_cost = st.number_input(
            "Retention Offer Cost ($)", min_value=5.0, max_value=500.0, value=50.0, step=5.0
        )
    with roi_c2:
        customer_ltv = st.number_input(
            "Assumed Saved Customer Value ($)", min_value=100.0, max_value=5000.0, value=600.0, step=50.0
        )
    with roi_c3:
        offer_success_rate = (
            st.slider("Assumed Campaign Save Rate (%)", min_value=10, max_value=90, value=50) / 100.0
        )

    oof_path = "models/oof_predictions.csv"
    if os.path.exists(oof_path):
        oof_df = pd.read_csv(oof_path)
        eval_y_true = oof_df["y_true"].to_numpy(dtype=int)
        eval_y_prob = oof_df["y_prob"].to_numpy(dtype=float)

        opt_res = find_optimal_threshold(
            eval_y_true,
            eval_y_prob,
            retention_cost=retention_cost,
            customer_ltv=customer_ltv,
            offer_success_rate=offer_success_rate,
        )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Optimal Probability Cutoff", f"{opt_res['optimal_threshold']:.2f}")
        r2.metric("Projected Net Savings (OOF)", f"${opt_res['net_savings_usd']:,.0f}")
        r3.metric("Customers Targeted", f"{opt_res['customers_targeted']:,}")
        r4.metric("Expected Churn Recall", f"{opt_res['recall'] * 100:.1f}%")

        thresholds_grid = np.linspace(0.10, 0.90, 41)
        curve_savings = []
        for th in thresholds_grid:
            preds = (eval_y_prob >= th).astype(int)
            tp = int(((preds == 1) & (eval_y_true == 1)).sum())
            targeted = int(preds.sum())
            net = (tp * offer_success_rate * customer_ltv) - (targeted * retention_cost)
            curve_savings.append(
                {"Probability Threshold": round(float(th), 2), "Net Savings ($)": net}
            )

        curve_df = pd.DataFrame(curve_savings)
        fig_roi = px.area(
            curve_df,
            x="Probability Threshold",
            y="Net Savings ($)",
            title="Out-of-Fold Net Financial Savings vs. Decision Threshold (Assumption-Driven What-If)",
        )
        fig_roi.add_vline(
            x=opt_res["optimal_threshold"],
            line_dash="dash",
            line_color="#00CC96",
            annotation_text=f"Optimal ({opt_res['optimal_threshold']:.2f})",
        )
        fig_roi.update_layout(height=250, margin={"l": 15, "r": 15, "t": 35, "b": 10})
        st.plotly_chart(fig_roi, use_container_width=True)
    else:
        st.warning("Out-of-fold predictions missing. Run 'python src/train.py' to enable cost-threshold simulation.")

    with st.expander("📡 Live FastAPI Inference & Request Telemetry (logs/predictions.db)", expanded=True):
        db_path = "logs/predictions.db"
        if os.path.exists(db_path):
            try:
                with sqlite3.connect(db_path) as conn:
                    logs_df = pd.read_sql_query(
                        "SELECT * FROM prediction_logs ORDER BY id DESC LIMIT 10",
                        conn,
                    )
                if not logs_df.empty:
                    t1, t2, t3 = st.columns(3)
                    t1.metric("Recent Logged Calls", f"{len(logs_df)}")
                    if "latency_ms" in logs_df.columns:
                        t2.metric("Mean API Latency", f"{logs_df['latency_ms'].mean():.1f} ms")
                    if "risk_level" in logs_df.columns:
                        high_pct = (logs_df["risk_level"] == "High Risk").mean() * 100
                        t3.metric("High-Risk Traffic Share", f"{high_pct:.1f}%")
                    st.dataframe(logs_df, use_container_width=True)
                else:
                    st.info("No API inference logs recorded yet.")
            except Exception as exc:
                st.warning(f"Could not read telemetry database: {exc}")
        else:
            st.info("Telemetry database will appear automatically once FastAPI `/predict` receives requests.")

    with st.expander("📊 Holdout Model Validation & Governance Metrics", expanded=False):
        metrics_path = "models/metrics.json"
        imp_path = "models/feature_importances.json"

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Holdout ROC-AUC", f"{float(metrics['roc_auc']):.3f}")
            m2.metric("Holdout PR-AUC", f"{float(metrics.get('pr_auc', 0.0)):.3f}")
            m3.metric("Brier Score", f"{float(metrics.get('brier_score', 0.0)):.4f}")
            m4.metric("Recall @0.5", f"{float(metrics['recall']) * 100:.1f}%")
            m5.metric("Precision @0.5", f"{float(metrics['precision']) * 100:.1f}%")

            col_a, col_b = st.columns(2)
            with col_a:
                cm_data = np.array(metrics["confusion_matrix"])
                fig_cm = px.imshow(
                    cm_data,
                    text_auto=True,
                    title="Holdout Confusion Matrix",
                    labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                    x=["Retained (0)", "Churned (1)"],
                    y=["Retained (0)", "Churned (1)"],
                    color_continuous_scale="Blues",
                )
                fig_cm.update_layout(height=280)
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_b:
                if os.path.exists(imp_path):
                    with open(imp_path, "r") as f:
                        imp_records = json.load(f)
                    global_imp = pd.DataFrame(imp_records).head(8).sort_values("Importance Weight")
                    fig_imp = px.bar(
                        global_imp,
                        x="Importance Weight",
                        y="Feature",
                        orientation="h",
                        title="Holdout Mean |SHAP| Feature Importance Share",
                    )
                    fig_imp.update_layout(height=280)
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("Run 'python src/train.py' to compute feature importances.")
        else:
            st.warning("Governance metrics missing. Run 'python src/train.py' first.")