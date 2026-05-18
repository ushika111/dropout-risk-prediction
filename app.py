import streamlit as st
import pandas as pd
import pickle
#import matplotlib.pyplot as plt
import xgboost as xgb

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Dropout Risk System", layout="wide")

st.markdown("""
# 🎓 Student Dropout Risk Prediction System  
AI-powered early detection of at-risk students using Machine Learning
""")

st.caption("Built using XGBoost + Streamlit + Student Engagement Data")

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# =========================
# UPLOAD DATA
# =========================
file = st.file_uploader(" Upload Student Dataset (CSV)", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.subheader(" Data Preview")
    st.dataframe(df.head())

    # =========================
    # PREPROCESSING
    # =========================
    if "student_id" in df.columns:
        df = df.drop("student_id", axis=1)

    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    # =========================
    # PREDICTION
    # =========================
    if st.button(" Predict Dropout Risk"):

        preds = model.predict(df)
        probs = model.predict_proba(df).max(axis=1)

        df["Predicted Risk (Encoded)"] = preds
        df["Dropout Probability"] = probs

        # Risk labels
        risk_map = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}
        df["Risk Level"] = df["Predicted Risk (Encoded)"].map(risk_map)

        # =========================
        # INSIGHT ENGINE
        # =========================
        def explain(row):
            if row["Risk Level"] == "High Risk":
                return "⚠️ High risk — immediate intervention required"
            elif row["Risk Level"] == "Medium Risk":
                return "⚠️ Medium risk — monitor student engagement"
            else:
                return "✅ Low risk — performing well"

        df["Insight"] = df.apply(explain, axis=1)

        # =========================
        # RESULTS
        # =========================
        st.subheader("📊 Prediction Results")
        st.dataframe(df)

        # =========================
        # RISK SUMMARY
        # =========================
        st.subheader("Risk Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("🔴 High Risk", len(df[df["Risk Level"] == "High Risk"]))
        col2.metric("🟠 Medium Risk", len(df[df["Risk Level"] == "Medium Risk"]))
        col3.metric("🟢 Low Risk", len(df[df["Risk Level"] == "Low Risk"]))

        # =========================
        # PROBABILITY VISUAL
        # =========================
        st.subheader(" Dropout Probability Distribution")
        st.bar_chart(df["Dropout Probability"])

        # =========================
        # FEATURE IMPORTANCE
        # =========================
        st.subheader("📊 Feature Importance (Model Explainability)")

        fig, ax = plt.subplots()
        xgb.plot_importance(model, ax=ax)
        st.pyplot(fig)

        # =========================
        # INSIGHTS TABLE
        # =========================
        st.subheader(" AI Insights")
        st.dataframe(df[["Risk Level", "Dropout Probability", "Insight"]])

        # =========================
        # DOWNLOAD
        # =========================
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            " Download Full Report",
            csv,
            "dropout_risk_report.csv",
            "text/csv"
        )

# =========================
# PROJECT INFO
# =========================
st.markdown("""
---
### About This System
- Predicts student dropout risk using ML
- Uses behavioural + performance data
- Helps institutions identify at-risk students early
""")
