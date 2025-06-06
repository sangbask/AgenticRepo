import pytest
import pandas as pd
from risk_modules.credit_module import CreditRiskModule

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'loan_amount': [100000],
        'credit_score': [700],
        'ltv_ratio': [0.83],
        'credit_history': [5],
        'dti_ratio': [0.3]
    })

def test_credit_risk(sample_df):
    module = CreditRiskModule()
    df_result = module.predict_credit_risk(sample_df.copy())
    assert 'credit_risk_score' in df_result.columns