# risk_modules/fraud_module.py
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest

class FraudRiskModule:
    def __init__(self):
        self.isolation_forest_model = None
        self.xgb_model = None

    def predict_isolation_forest(self, df: pd.DataFrame):
        self.isolation_forest_model = IsolationForest(contamination=0.1, random_state=42)
        self.isolation_forest_model.fit(df[['loan_amount', 'collateral_value', 'credit_score', 'ltv_ratio',
                                             'credit_history', 'collateral_type_num', 'aml_score']])
        df['fraud_risk'] = self.isolation_forest_model.predict(df[['loan_amount', 'collateral_value', 'credit_score',
                                                                    'ltv_ratio', 'credit_history', 'collateral_type_num',
                                                                    'aml_score']])
        df['fraud_risk_label'] = df['fraud_risk'].apply(lambda x: "Fraudulent" if x == -1 else "Normal")
        return df

    def train_xgb_model(self, df: pd.DataFrame):
        X = df[['loan_amount', 'collateral_value', 'credit_score', 'ltv_ratio',
                 'credit_history', 'collateral_type_num', 'aml_score']]
        y = df['confirmed_fraud'] if 'confirmed_fraud' in df.columns else (df['fraud_risk_label'] == "Fraudulent").astype(int)
        self.xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        self.xgb_model.fit(X, y)

    def predict_xgboost(self, df: pd.DataFrame):
        if self.xgb_model is not None:
            X = df[['loan_amount', 'collateral_value', 'credit_score', 'ltv_ratio',
                     'credit_history', 'collateral_type_num', 'aml_score']]
            df['xgb_fraud'] = self.xgb_model.predict(X)
            df['xgb_fraud_label'] = df['xgb_fraud'].apply(lambda x: "Fraudulent" if x == 1 else "Normal")
        return df
