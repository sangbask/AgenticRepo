import pytest
import pandas as pd
from risk_modules.fraud_module import FraudRiskModule

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'loan_amount': [100000],
        'collateral_value': [120000],
        'credit_score': [700],
        'ltv_ratio': [0.83],
        'credit_history': [5],
        'collateral_type_num': [2],
        'aml_score': [45]
    })

def test_isolation_forest(sample_df):
    module = FraudRiskModule()
    df_result = module.predict_isolation_forest(sample_df.copy())
    assert 'fraud_risk_label' in df_result.columns

def test_xgboost_prediction(sample_df):
    module = FraudRiskModule()
    df = module.predict_isolation_forest(sample_df.copy())
    module.train_xgb_model(df)
    df_result = module.predict_xgboost(df.copy())
    assert 'xgb_fraud_label' in df_result.columns