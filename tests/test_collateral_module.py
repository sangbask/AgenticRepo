import pytest
import pandas as pd
from risk_modules.collateral_module import CollateralRiskModule

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'collateral_type_num': [2],
        'market_volatility': [0.2],
        'ltv_ratio': [0.83]
    })

def test_collateral_risk(sample_df):
    module = CollateralRiskModule()
    df_result = module.predict_collateral_risk(sample_df.copy())
    assert 'collateral_risk_score' in df_result.columns