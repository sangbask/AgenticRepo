import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from openai import OpenAI
from dotenv import load_dotenv
import os
import yfinance as yf
import plotly.graph_objects as go

# Import the risk modules
from risk_modules.fraud_module import FraudRiskModule
from risk_modules.credit_module import CreditRiskModule
from risk_modules.collateral_module import CollateralRiskModule
from risk_modules.combined_risk_module import CombinedRiskModule

# Initialize the risk modules
fraud_module = FraudRiskModule()
credit_module = CreditRiskModule()
collateral_module = CollateralRiskModule()
combined_module = CombinedRiskModule()

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found in .env.")
    st.stop()

client = OpenAI(api_key=api_key)

def generate_gpt_summary(fraud_rate, avg_ltv, avg_credit_score, model_name):
    prompt = f"""
    Provide a short, professional summary of the fraud risk in this dataset for {model_name}.
    - Fraud rate: {fraud_rate:.2f}%
    - Average LTV ratio (fraudulent): {avg_ltv:.2f}
    - Average credit score (fraudulent): {avg_credit_score:.2f}
    Summarize key patterns and business implications.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a professional risk analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def highlight_fraudulent_rows(row, fraud_label_col):
    color = 'background-color: #FFCDD2' if row[fraud_label_col] == "Fraudulent" else ''
    return [color] * len(row)

# --- Page config & theme ---
st.set_page_config(page_title="Agentic Loan Collateral Risk Insights", layout="wide")
st.markdown("""
<div style='text-align: center;'>
    <img src='https://cdn-icons-png.flaticon.com/512/4213/4213249.png' width='80' />
</div>
""", unsafe_allow_html=True)
st.markdown("""
<h3 style='text-align: center;'>Agentic Loan Collateral Risk & Exposure Dashboard</h3>
""", unsafe_allow_html=True)

# --- Sidebar: Dynamic threshold slider ---
st.sidebar.header(" Adjust Detection Parameters")
threshold_input = st.sidebar.slider("Dynamic Fraud Threshold (%)", 1, 50, 10, step=1) / 100.0

# --- Sidebar: Real-time transaction check inputs ---
st.sidebar.header(" Real-Time Transaction Check")
loan_amount_input = st.sidebar.number_input("Loan Amount", 0.0, 1e9, 100000.0, step=1000.0)
collateral_value_input = st.sidebar.number_input("Collateral Value", 1.0, 1e9, 120000.0, step=1000.0)
credit_score_input = st.sidebar.slider("Credit Score", 300, 850, 650)
credit_history_input = st.sidebar.number_input("Credit History (years)", 0, 30, 5)
collateral_type_input = st.sidebar.selectbox("Collateral Type", ["Bonds", "Equities", "Cash"])
aml_score_input = st.sidebar.slider("AML Risk Score", 0, 100, 50)

# Real-time market adjustment for Equities
if collateral_type_input == "Equities":
    st.sidebar.markdown(" **Fetching Real-Time Market Data for Equities...**")
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
collateral_type_map = {"Bonds": 1, "Equities": 2, "Cash": 3}
collateral_type_num = collateral_type_map.get(collateral_type_input, 0)

# --- File uploader ---
uploaded_file = st.file_uploader(" Upload your CSV file with Dummy Labels", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    for col in ['loan_amount', 'collateral_value', 'credit_score', 'credit_history', 'aml_score']:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce')
    df['collateral_type'] = df.get('collateral_type', 'Cash')
    df['collateral_type_num'] = df['collateral_type'].map(collateral_type_map).fillna(0)
    df = df[df['collateral_value'] > 0]
    df['ltv_ratio'] = df['loan_amount'] / df['collateral_value']
    df.fillna(0, inplace=True)

    # --- Fraud Risk ---
    df = fraud_module.predict_isolation_forest(df)
    fraud_module.train_xgb_model(df)
    df = fraud_module.predict_xgboost(df)

    # --- Credit Risk ---
    df = credit_module.predict_credit_risk(df)

    # --- Collateral Risk ---
    df = collateral_module.predict_collateral_risk(df)

    # --- Combined Risk ---
    df['fraud_risk_score'] = df['fraud_risk_label'].apply(lambda x: 1 if x == "Fraudulent" else 0)
    df = combined_module.compute_combined_risk(df)

    # --- Tabs ---
    tab_fraud, tab_credit, tab_collateral, tab_combined = st.tabs([
        "🔍 Fraud Risk", "📊 Credit Risk", "🏦 Collateral Risk", "🌐 Combined Risk Score"
    ])
    with tab_fraud:
        st.subheader("Fraud Risk Analysis (IsolationForest vs XGBoost)")
        subtab_iso, subtab_xgb = st.tabs(["UnSupervised", "Supervised"])
        for subtab, label, model in [
            (subtab_iso, 'fraud_risk_label', 'UnSupervised'),
            (subtab_xgb, 'xgb_fraud_label', 'Supervised')
        ]:
            with subtab:
                fraud_df = df[df[label] == "Fraudulent"]
                total_fraud = len(fraud_df)
                fraud_rate = (total_fraud / len(df)) * 100
                st.metric("Potential Fraudulent Cases", total_fraud)
                col1, col2 = st.columns(2)
                with col1:
                    pie_chart = px.pie(df, names=label, title="Fraud Risk Distribution",
                                       color=label, color_discrete_map={"Fraudulent": "red", "Normal": "green"})
                    st.plotly_chart(pie_chart, key=f"pie_{model}")
                with col2:
                    if not fraud_df.empty:
                        feature_variance = fraud_df[['loan_amount', 'collateral_value', 'credit_score', 'ltv_ratio',
                                                     'credit_history', 'collateral_type_num', 'aml_score']].var()
                        feature_variance_log = np.log1p(feature_variance).reset_index()
                        feature_variance_log.columns = ['Feature', 'Log Variance']
                        bar_chart = px.bar(feature_variance_log, x='Feature', y='Log Variance', color='Log Variance',
                                           title="Log-Scaled Feature Variance", color_continuous_scale="reds")
                        st.plotly_chart(bar_chart, key=f"bar_{model}")
                    else:
                        st.info("No fraudulent transactions detected to show feature variance.")
                if not fraud_df.empty:
                    avg_ltv, avg_credit = fraud_df['ltv_ratio'].mean(), fraud_df['credit_score'].mean()
                    gpt_summary = generate_gpt_summary(fraud_rate, avg_ltv, avg_credit, model)
                    st.markdown("### 📝 AI-Powered Risk Insights Summary")
                    st.write(gpt_summary)
                st.markdown("### 📊 Full Loan Portfolio Overview")
                styled_df = df.style.apply(lambda row: highlight_fraudulent_rows(row, label), axis=1)
                st.dataframe(styled_df, height=400)
                st.markdown("### 👥 Analyst Feedback")
                if not fraud_df.empty:
                    selected_indices = st.multiselect(f"Select transactions confirmed as fraud in {model}:", fraud_df.index, key=f"feedback_{model}")
                    if st.button(f"Submit Feedback for {model}"):
                        feedback_df = df.loc[selected_indices]
                        feedback_df['confirmed_fraud'] = True
                        feedback_df.to_csv("human_feedback.csv", mode='a', header=False, index=False)
                        st.success("✅ Feedback saved for retraining.")
                else:
                    st.info("No fraudulent transactions to review.")

    # Fraud Risk Tab (unchanged)
    # Credit Risk Tab (unchanged, but ensure radar chart created)
    with tab_credit:
        st.subheader("Credit Risk Analysis")
        radar_data = pd.DataFrame({
            'Feature': ['Loan Amount', 'Credit Score', 'LTV Ratio', 'Credit History', 'DTI Ratio'],
            'Low Risk': [0.1, 0.8, 0.3, 0.7, 0.2],
            'High Risk': [0.7, 0.3, 0.8, 0.4, 0.9]
        })
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=radar_data['Low Risk'], theta=radar_data['Feature'], fill='toself', name='Low Risk'))
        fig_radar.add_trace(go.Scatterpolar(r=radar_data['High Risk'], theta=radar_data['Feature'], fill='toself', name='High Risk'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Credit Risk Feature Profiles")
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(fig_radar, use_container_width=True)
        with col2: st.write(df[['loan_amount', 'credit_score', 'ltv_ratio', 'credit_history', 'dti_ratio', 'credit_risk_score']])
        st.info("Credit risk scores range from 0 (low risk) to 1 (high risk).")

    # Collateral Risk Tab with spline chart and metrics
    with tab_collateral:
        st.subheader("Collateral Risk Analysis")
    
        # Prepare data for donut chart
        collateral_risk_data = pd.DataFrame({
            'Collateral Type': ['Bonds', 'Equities', 'Cash'],
            'Collateral Risk Score': [
                df[df['collateral_type'] == 'Bonds']['collateral_risk_score'].mean(),
                df[df['collateral_type'] == 'Equities']['collateral_risk_score'].mean(),
                df[df['collateral_type'] == 'Cash']['collateral_risk_score'].mean()
            ]
        })
    
        # Create donut chart with a balanced modern color palette
        fig_donut = px.pie(
            collateral_risk_data,
            names='Collateral Type',
            values='Collateral Risk Score',
            hole=0.5,  # balanced donut size
            color='Collateral Type',
            color_discrete_sequence=px.colors.qualitative.Set2,  # softer but not too light
            title="Collateral Risk Across Types"
        )
        
        # Reduce text size slightly, keep good contrast
        fig_donut.update_traces(
            textinfo='percent+label',
            textfont_size=13,
            pull=[0.05, 0, 0],
            rotation=90,
            marker=dict(line=dict(color='#FFFFFF', width=1))  # clean white border
        )
        
        # Keep chart compact and tidy
        fig_donut.update_layout(
            margin=dict(t=30, l=30, r=30, b=30),
            height=350
        )
        
        # Two-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Collateral Risk Distribution")
            st.plotly_chart(fig_donut, use_container_width=True)
        
        with col2:
            st.markdown("#### Collateral Risk Details Table")
            st.write(df[['collateral_type', 'market_volatility', 'ltv_ratio', 'collateral_risk_score']])
        
        st.info("Collateral risk scores indicate potential collateral devaluation.")
    with tab_combined:
        st.subheader("Combined Risk Score Overview")

        # --- Metrics ---
        avg_combined_risk = df['combined_risk_score'].mean()
        max_combined_risk = df['combined_risk_score'].max()
        min_combined_risk = df['combined_risk_score'].min()
    
        st.markdown("### 📊 Combined Risk Metrics")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Average Combined Risk", f"{avg_combined_risk:.2f}")
        metric_cols[1].metric("Highest Combined Risk", f"{max_combined_risk:.2f}")
        metric_cols[2].metric("Lowest Combined Risk", f"{min_combined_risk:.2f}")
    
        # --- Bubble (scatter) chart for combined risk scores ---
        st.markdown("### 🎯 Combined Risk Distribution")
        fig_bubble = px.scatter(
            df,
            x='credit_risk_score',
            y='collateral_risk_score',
            size='combined_risk_score',
            color='combined_risk_score',
            hover_data=['loan_amount', 'credit_score', 'collateral_type'],
            color_continuous_scale='Viridis',
            title="Combined Risk  Chart: Credit vs Collateral Risk"
        )
        fig_bubble.update_layout(height=400)
        st.plotly_chart(fig_bubble, use_container_width=True)
    
        # Compact, professional table view
        st.markdown("### 🗂️ Detailed Risk Breakdown")
        table_data = df[['loan_amount', 'credit_score', 'collateral_type', 
                         'fraud_risk_score', 'credit_risk_score', 'collateral_risk_score', 'combined_risk_score']]
        st.dataframe(table_data.style.background_gradient(cmap='Oranges').format({
            'loan_amount': '₹{:,.0f}',
            'credit_score': '{:,.0f}',
            'fraud_risk_score': '{:.2f}',
            'credit_risk_score': '{:.2f}',
            'collateral_risk_score': '{:.2f}',
            'combined_risk_score': '{:.2f}'
        }), height=400)
    
        st.info("This polished view showcases key risk metrics alongside the bubble chart for an engaging client pitch.")

# --- Real-time transaction prediction ---
    if st.sidebar.button("🔎 Check Real-Time Transaction"):
        single_data = pd.DataFrame({
            'loan_amount': [loan_amount_input],
            'collateral_value': [adjusted_collateral_value],
            'credit_score': [credit_score_input],
            'ltv_ratio': [ltv_ratio_input],
            'credit_history': [credit_history_input],
            'collateral_type_num': [collateral_type_num],
            'aml_score': [aml_score_input]
        })
        iso_pred = fraud_module.isolation_forest_model.predict(single_data)[0]
        xgb_pred = fraud_module.xgb_model.predict(single_data)[0]
        st.sidebar.write("UnSupervised (IsolationForest): " + ("Fraudulent" if iso_pred == -1 else "Normal"))
        st.sidebar.write("Supervised (XGBoost): " + ("Fraudulent" if xgb_pred == 1 else "Normal"))
else:
    st.warning("Please upload a CSV file to start the analysis.")
