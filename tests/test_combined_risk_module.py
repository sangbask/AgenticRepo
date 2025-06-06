import pytest
import pandas as pd
from risk_modules.combined_risk_module import CombinedRiskModule

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'fraud_risk_score': [0.6],
        'credit_risk_score': [0.7],
        'collateral_risk_score': [0.8]
    })

def test_combined_risk_score(sample_df):
    module = CombinedRiskModule()
    df_result = module.compute_combined_risk(sample_df.copy())
    assert 'combined_risk_score' in df_result.columns