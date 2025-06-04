import joblib
import pandas as pd
import os

class CollateralRiskModule:
    def __init__(self):
        # Load the trained model and scaler from the same directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = joblib.load(os.path.join(current_dir, "collateral_risk_model.pkl"))
        self.scaler = joblib.load(os.path.join(current_dir, "collateral_risk_scaler.pkl"))

    def predict_collateral_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        # ✅ Ensure all required features are present
        if 'market_volatility' not in df.columns:
            df['market_volatility'] = 0.2  # default fallback
        features = ['collateral_type_num', 'market_volatility', 'ltv_ratio']
        
        # ✅ Scale and predict
        X_scaled = self.scaler.transform(df[features])
        collateral_risk_score = self.model.predict(X_scaled)
        df['collateral_risk_score'] = collateral_risk_score.clip(0, 1)
        return df
