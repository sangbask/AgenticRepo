import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px

# --- Page config & theme ---
st.set_page_config(page_title="Agentic Loan Collateral Risk Insights", layout="wide")

# --- Optional: Replace with relevant fraud detection logo ---
st.markdown("""
<div style='text-align: center;'>
    <img src='https://cdn-icons-png.flaticon.com/512/4213/4213249.png' width='80' />
</div>
""", unsafe_allow_html=True)


# --- Title only, smaller ---
st.markdown("""
<h3 style='text-align: center; font-size: 1.5rem;'>Agentic Loan Collateral Fraud Risk Detector</h3>
""", unsafe_allow_html=True)

# --- Sidebar: Dynamic threshold slider ---
st.sidebar.header("🔧 Adjust Detection Parameters")
threshold_input = st.sidebar.slider(
    "Dynamic Fraud Threshold (%)",
    min_value=1,
    max_value=50,
    value=10,
    step=1,
    help="Set the fraud risk threshold. Higher values increase sensitivity (more likely to flag fraud)."
) / 100.0

# --- Sidebar: Real-time transaction check inputs ---
st.sidebar.header("🔍 Real-Time Transaction Check")
loan_amount_input = st.sidebar.number_input("Loan Amount", min_value=0.0, value=100000.0, step=1000.0)
collateral_value_input = st.sidebar.number_input("Collateral Value", min_value=1.0, value=120000.0, step=1000.0)
credit_score_input = st.sidebar.slider("Credit Score", min_value=300, max_value=850, value=650)

if collateral_value_input > 0:
    ltv_ratio_input = loan_amount_input / collateral_value_input
else:
    ltv_ratio_input = 0.0

# --- File uploader ---
uploaded_file = st.file_uploader("📂 Upload your CSV file", type="csv")

model = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- Clean data & feature engineering ---
    df['loan_amount'] = pd.to_numeric(df['loan_amount'], errors='coerce')
    df['collateral_value'] = pd.to_numeric(df['collateral_value'], errors='coerce')
    df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce')
    df = df[df['collateral_value'] > 0]
    df['ltv_ratio'] = df['loan_amount'] / df['collateral_value']
    df.fillna(0, inplace=True)

    # --- Model training with user-adjusted threshold ---
    st.write(f"🤖 Using your selected fraud threshold: **{threshold_input*100:.1f}%**")
    model = IsolationForest(contamination=threshold_input, random_state=42)
    model.fit(df[['loan_amount', 'collateral_value', 'credit_score', 'ltv_ratio']])
    df['fraud_risk'] = model.predict(df[['loan_amount', 'collateral_value', 'credit_score', 'ltv_ratio']])
    df['fraud_risk_label'] = df['fraud_risk'].apply(lambda x: "Fraudulent" if x == -1 else "Normal")

    # --- Mock collateral check ---
    def mock_api_check(loan_id):
        return "Valid" if loan_id.endswith("5") else "Suspicious"
    df['collateral_check'] = df['loan_id'].apply(mock_api_check)

    # --- Metrics ---
    total_records = len(df)
    total_fraud = (df['fraud_risk_label'] == "Fraudulent").sum()
    fraud_rate = (total_fraud / total_records) * 100

    
    metric_cols = st.columns(3)
    metric_cols[0].metric("Total Records", total_records)
    metric_cols[1].metric("Potential Fraudulent Cases", total_fraud)
    metric_cols[2].metric("Fraud Rate (%)", f"{fraud_rate:.2f}")

    if fraud_rate > 20:
        st.error("⚠️ Elevated fraud risk! Recommend manual review of flagged cases.")
    else:
        st.success("✅ Fraud risk is within acceptable limits.")

    st.markdown("---")

    # --- Charts layout ---
    charts_col1, charts_col2 = st.columns(2)

    fraud_count = df['fraud_risk_label'].value_counts().reset_index()
    fraud_count.columns = ['Risk Label', 'Count']
    pie_chart = px.pie(fraud_count, values='Count', names='Risk Label', title="Fraud Risk Distribution",
                       color='Risk Label', color_discrete_map={"Fraudulent": "red", "Normal": "green"})
    pie_chart.update_traces(textinfo='percent+label', pull=[0.1, 0])
    charts_col1.plotly_chart(pie_chart, use_container_width=True)

    if total_fraud > 0:
        fraud_df = df[df['fraud_risk_label'] == "Fraudulent"]
        feature_variance = fraud_df[['loan_amount', 'collateral_value', 'credit_score', 'ltv_ratio']].var().reset_index()
        feature_variance.columns = ['Feature', 'Variance']
        bar_chart = px.bar(feature_variance, x='Feature', y='Variance', color='Variance',
                           title="Feature Variance in Potential Fraud Cases", color_continuous_scale="reds")
        charts_col2.plotly_chart(bar_chart, use_container_width=True)
    else:
        charts_col2.info("No fraudulent transactions detected to show feature variance.")

    st.markdown("---")

    # --- Data tables ---
    tables_col1, tables_col2 = st.columns([1, 1.5])
    tables_col1.markdown("<h4 style='font-size: 1rem;'>🔍 Potential Fraudulent Cases</h4>", unsafe_allow_html=True)
    fraudulent_df = df[df['fraud_risk_label'] == "Fraudulent"]
    tables_col1.dataframe(fraudulent_df, use_container_width=True, height=400)

    tables_col2.markdown("<h4 style='font-size: 1rem;'>📋 Complete Loan Transaction Records</h4>", unsafe_allow_html=True)
    tables_col2.dataframe(df, use_container_width=True, height=400)

    # --- Analyst review feedback ---
    st.markdown("<h4 style='font-size: 1rem;'>👥 Analyst Review and Feedback</h4>", unsafe_allow_html=True)
    if not fraudulent_df.empty:
        selected_indices = st.multiselect("Select transactions confirmed as fraud:", fraudulent_df.index)
        if st.button("💾 Submit Feedback"):
            feedback_df = df.loc[selected_indices]
            feedback_df['confirmed_fraud'] = True
            feedback_df.to_csv("human_feedback.csv", mode='a', header=False, index=False)
            st.success("✅ Feedback saved for retraining.")
    else:
        st.info("No fraudulent transactions to review.")

    # --- Download full dataset ---
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Risk Analysis Report (CSV)", data=csv,
                       file_name="agentic_fraud_detection_results.csv",
                       mime="text/csv")
else:
    st.warning("Please upload a CSV file to start the analysis.")

# --- Sidebar: Real-time transaction fraud check ---
if st.sidebar.button("🔎 Check Real-Time Transaction"):
    if model is not None:
        single_data = pd.DataFrame({
            'loan_amount': [loan_amount_input],
            'collateral_value': [collateral_value_input],
            'credit_score': [credit_score_input],
            'ltv_ratio': [ltv_ratio_input]
        })
        fraud_prediction = model.predict(single_data)[0]
        if fraud_prediction == -1:
            st.sidebar.error("🚨 This transaction is flagged as **Fraudulent**.")
        else:
            st.sidebar.success("✅ This transaction appears **Normal**.")
    else:
        st.sidebar.warning("Please upload a CSV file first to initialize the model.")
