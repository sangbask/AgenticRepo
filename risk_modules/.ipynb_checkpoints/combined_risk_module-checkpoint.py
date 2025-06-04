class CombinedRiskModule:
    def __init__(self):
        pass

    def compute_combined_risk(self, df):
        df['combined_risk_score'] = (
            0.4 * df['fraud_risk_score'] +
            0.4 * df['credit_risk_score'] +
            0.2 * df['collateral_risk_score']
        ).clip(0, 1)
        return df
