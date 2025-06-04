import joblib
import pandas as pd

class CreditRiskModule:
    def __init__(self):
        # Load the trained Logistic Regression model and scaler
        self.model = joblib.load("risk_modules/credit_risk_model.pkl")
        self.scaler = joblib.load("risk_modules/credit_risk_scaler.pkl")

    def predict_credit_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        # ✅ Ensure dti_ratio exists (with default if missing)
        if 'dti_ratio' not in df.columns:
            df['dti_ratio'] = 0.3  # or another default
    
        features = ['loan_amount', 'credit_score', 'ltv_ratio', 'credit_history', 'dti_ratio']
    
        # ✅ Now safely scale the data
        X_scaled = self.scaler.transform(df[features])
        credit_risk_score = self.model.predict_proba(X_scaled)[:, 1]
    
        # ✅ Add risk score
        df['credit_risk_score'] = credit_risk_score.clip(0, 1)
        return df

