import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px
from openai import OpenAI
from dotenv import load_dotenv
import os
import yfinance as yf
import numpy as np

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found in .env.")
    st.stop()

client = OpenAI(api_key=api_key)

def generate_gpt_summary(fraud_rate, avg_ltv, avg_credit_score):
    prompt = f"""
    Provide a short, professional summary of the fraud risk in this dataset.
    - Fraud rate: {fraud_rate:.2f}%
    - Average LTV ratio (fraudulent): {avg_ltv:.2f}
    - Average credit score (fraudulent): {avg_credit_score:.2f}
    Summarize key patterns and business implications.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a professional fraud risk analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# --- Page config & theme ---
st.set_page_config(page_title="Agentic Loan Collateral Risk Insights", layout="wide")

# --- Logo ---
st.markdown("""
<div style='text-align: center;'>
    <img src='https://cdn-icons-png.flaticon.com/512/4213/4213249.png' width='80' />
</div>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("""
<h3 style='text-align: center; font-size: 1.5rem;'>Agentic Loan Collateral Fraud Risk Detector</h3>
""", unsafe_allow_html=True)

# --- Sidebar: Dynamic threshold slider ---
st.sidebar.header("🔧 Adjust Detection Parameters")
threshold_input = st.sidebar.slider(
    "Dynamic Fraud Threshold (%)",
    min_value=1, max_value=50, value=10, step=1,
    help="ℹ️ Sets how sensitive the fraud detection is. Higher thresholds flag more transactions."
) / 100.0

# --- Sidebar: Real-time transaction check inputs ---
st.sidebar.header("🔍 Real-Time Transaction Check")

loan_amount_input = st.sidebar.number_input(
    "Loan Amount", min_value=0.0, value=100000.0, step=1000.0,
    help="ℹ️ Amount of the loan being analyzed."
)

collateral_value_input = st.sidebar.number_input(
    "Collateral Value", min_value=1.0, value=120000.0, step=1000.0,
    help="ℹ️ Value of the asset pledged for the loan."
)

credit_score_input = st.sidebar.slider(
    "Credit Score", min_value=300, max_value=850, value=650,
    help="ℹ️ Numeric measure of the borrower's creditworthiness."
)

credit_history_input = st.sidebar.number_input(
    "Credit History (years)", min_value=0, max_value=30, value=5,
    help="ℹ️ Number of years the borrower has credit history."
)

collateral_type_input = st.sidebar.selectbox(
    "Collateral Type", ["Bonds", "Equities", "Cash"],
    help="ℹ️ Type of asset used as collateral (Bonds, Equities, Cash)."
)

aml_score_input = st.sidebar.slider(
    "AML Risk Score", min_value=0, max_value=100, value=50,
    help="ℹ️ Anti-Money Laundering risk score based on checks."
)

# --- Real-time market adjustment for Equities ---
if collateral_type_input == "Equities":
    st.sidebar.markdown("💹 **Fetching Real-Time Market Data for Equities...**")
    sp500 = yf.Ticker("^GSPC")
    sp500_data = sp500.history(period="2d")
    if not sp500_data.empty and len(sp500_data["Close"]) >= 2:
        previous_close = sp500_data["Close"].iloc[-2]
        latest_close = sp500_data["Close"].iloc[-1]
        market_multiplier = 1.0 + (latest_close - previous_close) / previous_close
        st.sidebar.write(f"📈 S&P 500 Index: {latest_close:,.2f} ({(latest_close - previous_close)/previous_close*100:.2f}%)")
    else:
        market_multiplier = 1.0
        st.sidebar.write("⚠️ Could not fetch market data. Using default multiplier of 1.0.")
else:
    market_multiplier = 1.0

adjusted_collateral_value = collateral_value_input * market_multiplier
st.sidebar.write(f"📊 Adjusted Collateral Value: {adjusted_collateral_value:,.2f}")

# --- Calculate LTV Ratio ---
ltv_ratio_input = loan_amount_input / adjusted_collateral_value if adjusted_collateral_value > 0 else 0.0

# --- Convert Collateral Type to Numeric ---
collateral_type_map = {"Bonds": 1, "Equities": 2, "Cash": 3}
collateral_type_num = collateral_type_map.get(collateral_type_input, 0)

# --- File uploader ---
uploaded_file = st.file_uploader(
    "📂 Upload your CSV file",
    type="csv",
    help="ℹ️ Upload a CSV file with loan transaction data for analysis."
)

model = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['loan_amount'] = pd.to_numeric(df['loan_amount'], errors='coerce')
    df['collateral_value'] = pd.to_numeric(df['collateral_value'], errors='coerce')
    df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce')
    df['credit_history'] = pd.to_numeric(df.get('credit_history', 5), errors='coerce')
    df['aml_score'] = pd.to_numeric(df.get('aml_score', 50), errors='coerce')
    df['collateral_type'] = df.get('collateral_type', 'Cash')
    df['collateral_type_num'] = df['collateral_type'].map(collateral_type_map).fillna(0)
    df = df[df['collateral_value'] > 0]
    df['ltv_ratio'] = df['loan_amount'] / df['collateral_value']
    df.fillna(0, inplace=True)

    st.write(f"🤖 Using your selected fraud threshold: **{threshold_input*100:.1f}%**")
    model = IsolationForest(contamination=threshold_input, random_state=42)
    model.fit(df[['loan_amount', 'collateral_value', 'credit_score', 'ltv_ratio', 'credit_history', 'collateral_type_num', 'aml_score']])
    df['fraud_risk'] = model.predict(df[['loan_amount', 'collateral_value', 'credit_score', 'ltv_ratio', 'credit_history', 'collateral_type_num', 'aml_score']])
    df['fraud_risk_label'] = df['fraud_risk'].apply(lambda x: "Fraudulent" if x == -1 else "Normal")

    total_records = len(df)
    total_fraud = (df['fraud_risk_label'] == "Fraudulent").sum()
    fraud_rate = (total_fraud / total_records) * 100

    metric_cols = st.columns(3)
    metric_cols[0].metric("Total Records", total_records, help="ℹ️ Number of loan transactions analyzed.")
    metric_cols[1].metric("Potential Fraudulent Cases", total_fraud, help="ℹ️ Number of transactions flagged as potentially fraudulent.")
    metric_cols[2].metric("Fraud Rate (%)", f"{fraud_rate:.2f}", help="ℹ️ Percentage of transactions flagged as fraudulent.")

    if total_fraud > 0:
        fraud_df = df[df['fraud_risk_label'] == "Fraudulent"]
        avg_ltv = fraud_df['ltv_ratio'].mean()
        avg_credit_score = fraud_df['credit_score'].mean()
        gpt_summary = generate_gpt_summary(fraud_rate, avg_ltv, avg_credit_score)
        st.markdown("### 💡 AI-Powered Risk Insights Summary")
        st.write(gpt_summary)

    st.markdown("---")
    charts_col1, charts_col2 = st.columns(2)

    charts_col1.markdown("📊 **Fraud Risk Distribution**")
    charts_col1.markdown("_ℹ️ Pie chart showing the proportion of fraudulent vs normal transactions._")
    fraud_count = df['fraud_risk_label'].value_counts().reset_index()
    fraud_count.columns = ['Risk Label', 'Count']
    pie_chart = px.pie(fraud_count, values='Count', names='Risk Label', title="Fraud Risk Distribution",
                       color='Risk Label', color_discrete_map={"Fraudulent": "red", "Normal": "green"})
    pie_chart.update_traces(textinfo='percent+label', pull=[0.1, 0])
    charts_col1.plotly_chart(pie_chart, use_container_width=True)

    if total_fraud > 0:
        charts_col2.markdown("📈 **Log-Scaled Feature Variance**")
        charts_col2.markdown("_ℹ️ Highlights features with the highest variability in fraudulent transactions._")
        feature_variance = fraud_df[['loan_amount', 'collateral_value', 'credit_score', 'ltv_ratio', 'credit_history', 'collateral_type_num', 'aml_score']].var()
        feature_variance_log = np.log1p(feature_variance).reset_index()
        feature_variance_log.columns = ['Feature', 'Log Variance']
        log_chart = px.bar(feature_variance_log, x='Feature', y='Log Variance', color='Log Variance',
                           title="Log-Scaled Feature Variance", color_continuous_scale="reds")
        charts_col2.plotly_chart(log_chart, use_container_width=True)
    else:
        charts_col2.info("No fraudulent transactions detected to show feature variance.")

# The rest of your file remains unchanged
