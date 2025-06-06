
#  Agentic Loan Collateral Risk & Fraud Detection Dashboard

This Streamlit-powered application provides real-time, AI-driven risk analysis on loan and collateral datasets.  
It empowers financial analysts with a single dashboard to detect:

-  **Fraud Risk** (using Isolation Forest and XGBoost models)
- **Credit Risk** (via credit score and LTV ratio)
-  **Collateral Risk** (with market-aware logic)
-  **Combined Risk Scoring** (fraud + credit + collateral)

---

### Quickstart Guide

### Clone the Repository

```bash
git clone https://github.com/sangbask/AgenticRepo.git
cd AgenticRepo
```

---

### Set Up Python Environment

You can use `virtualenv` or `conda`. Example with `conda`:

```bash
conda create -n agentic_env python=3.9
conda activate agentic_env
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

---

###  Set Up Environment Variables

>  This app uses the OpenAI API. You’ll need your own API key to use the GPT-powered features.

####  Steps:

1. Create a `.env` file based on the provided template:

```bash
cp .env.example .env
```

2. Open `.env` and replace the placeholder:

```
OPENAI_API_KEY=sk-your-api-key-here
```

> Do **not** commit your actual API key to GitHub.

---

###  Launch the Dashboard

```bash
streamlit run FraudDetectorAIAgents.py
```

Then open your browser to:  
**http://localhost:8501**

---

##  Project Structure

```
AgenticRepo/
├── FraudDetectorAIAgents.py          # Main Streamlit app
├── .env.example                      # Placeholder for environment secrets
├── requirements.txt                  # Python dependencies
├── risk_modules/                     # Modular risk logic
│   ├── fraud_module.py
│   ├── credit_module.py
│   ├── collateral_module.py
│   └── combined_risk_module.py
└── sample_data/                      # (Optional) demo CSV files
```

---

##  Security Notice

- Your API key must **never** be committed to `.env`.
- `.env.example` is included as a safe starter file.
- `.env` is already included in `.gitignore`.

---

## Sample CSV Format (Expected Columns)

Make sure your uploaded CSV file includes:

```csv
loan_amount,collateral_value,credit_score,credit_history,collateral_type,aml_score
100000,120000,700,6,Equities,45
```
There is a sampel loan portfolio csv file in teh repo for sample data upload.
---

## 📞 Contact

Maintainer: **@sangbask**  
For issues, raise a GitHub issue or contact via internal review channels.
