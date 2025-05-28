import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px

# --- Page config ---
st.set_page_config(page_title="Agentic Loan Fraud Detection", layout="wide")

# --- Title & instructions ---
st.title("🤖 Agentic Loan Collateral Fraud Detection Dashboard")
st.markdown("""
This AI-powered **agent** analyzes loan booking data, detects potentially fraudulent transactions, and recommends actions.  
It simulates **real-time collateral verification** and **adapts dynamically** based on data context.  
Upload your **CSV file** below to begin.
""")

# --- File uploader ---
uploaded_file = st.file_uploader("📂 Upload your CSV file", type="csv")

if uploaded_file is not None:
    # --- Load data ---
    df = pd.read_csv(uploaded_file)

    # --- Clean data & feature engineering ---
    df['loan_amount'] = pd.to_numeric(df['loan_amount'], errors='coerce')
    df['collateral_value'] = pd.to_numeric(df['collateral_value'], errors='coerce')
    df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce')
    df = df[df['collateral_value'] > 0]
    df['ltv_ratio'] = df['loan_amount'] / df['collateral_value']
    df.fillna(0, inplace=True)

    # --- Dynamic threshold based on LTV context ---
    average_ltv = df['ltv_ratio'].mean()
    dynamic_threshold = 0.2 if average_ltv > 0.8 else 0.1
    st.write(f"🤖 Dynamic fraud threshold: **{dynamic_threshold*100:.1f}%** based on LTV context (average LTV: {average_ltv:.2f})")

    # --- Fraud detection model ---
    model = IsolationForest(contamination=dynamic_threshold, random_state=42)
    model.fit(df[['loan_amount', 'collateral_value', 'credit_score', 'ltv_ratio']])
    df['fraud_risk'] = model.predict(df[['loan_amount', 'collateral_value', 'credit_score', 'ltv_ratio']])
    df['fraud_risk_label'] = df['fraud_risk'].apply(lambda x: "Fraudulent" if x == -1 else "Normal")

    # --- Mock external collateral verification API ---
    def mock_api_check(loan_id):
        return "Valid" if loan_id.endswith("5") else "Suspicious"

    df['collateral_check'] = df['loan_id'].apply(mock_api_check)

    # --- Metrics ---
    total_records = len(df)
    total_fraud = (df['fraud_risk_label'] == "Fraudulent").sum()
    fraud_rate = (total_fraud / total_records) * 100

    # --- Display summary metrics ---
    metric_cols = st.columns(3)
    metric_cols[0].metric("🔢 Total Records", total_records)
    metric_cols[1].metric("🚨 Fraudulent Records", total_fraud)
    metric_cols[2].metric("📊 Fraud Rate (%)", f"{fraud_rate:.2f}")

    # --- Decision logic: escalate or not ---
    if fraud_rate > 20:
        st.error("⚠️ High fraud risk! 🚨 Suggesting **human review** for flagged transactions.")
    else:
        st.success("✅ Fraud risk within acceptable range. No immediate action required.")

    st.markdown("---")

    # --- Layout: Pie chart and data tables side by side ---
    col1, col2 = st.columns([1, 1.5])

    # Pie chart of fraud risk distribution
    fraud_count = df['fraud_risk_label'].value_counts().reset_index()
    fraud_count.columns = ['Risk Label', 'Count']
    fig = px.pie(
        fraud_count,
        values='Count',
        names='Risk Label',
        title="Fraud Risk Distribution",
        color='Risk Label',
        color_discrete_map={"Fraudulent": "red", "Normal": "green"}
    )
    fig.update_traces(textinfo='percent+label', pull=[0.1, 0])
    col1.plotly_chart(fig, use_container_width=True)

    # --- Data tables ---
    col2.markdown("#### 🚨 Flagged Fraudulent Transactions")
    fraudulent_df = df[df['fraud_risk_label'] == "Fraudulent"]
    col2.dataframe(fraudulent_df, use_container_width=True, height=300)

    col2.markdown("#### 📋 Full Dataset with Risk & API Check")
    col2.dataframe(df, use_container_width=True, height=400)

    # --- Human feedback loop simulation ---
    st.markdown("### 🔍 Human Feedback for Fraud Review")
    if not fraudulent_df.empty:
        selected_indices = st.multiselect("✅ Select transactions confirmed as fraud:", fraudulent_df.index)
        if st.button("💾 Submit Feedback"):
            feedback_df = df.loc[selected_indices]
            feedback_df['confirmed_fraud'] = True
            feedback_df.to_csv("human_feedback.csv", mode='a', header=False, index=False)
            st.success("✅ Feedback saved for retraining.")
    else:
        st.info("No fraudulent transactions to review.")

    # --- Download results ---
    st.markdown("---")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results CSV", data=csv,
                       file_name="agentic_fraud_detection_results.csv",
                       mime="text/csv")
else:
    st.warning("Please upload a CSV file to start the AI agent’s analysis.")

# --- Footer ---
st.markdown("""
---
✅ **Agentic AI-powered Fraud Detection**  
✅ **Real-time collateral verification (simulated)**  
✅ **Decision logic & human review loop**  
✅ **Dynamic fraud threshold adapts to data context**  
""")
