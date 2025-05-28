import pandas as pd
import numpy as np

# Set seed
np.random.seed(42)

# Create 100 sample loan records
num_samples = 100
data = {
    'loan_id': [f'LN{i:04}' for i in range(num_samples)],
    'loan_amount': np.random.randint(50000, 500000, num_samples),
    'collateral_value': np.random.randint(40000, 600000, num_samples),
    'credit_score': np.random.randint(300, 850, num_samples),
    'collateral_rating': np.random.choice([1, 2, 3], num_samples)
}

df = pd.DataFrame(data)

# Simulate ~15% fraudulent cases
fraud_ratio = 0.15
num_fraud = int(fraud_ratio * num_samples)
fraud_indices = np.random.choice(df.index, size=num_fraud, replace=False)

# Update loan_id to mark as fraudulent
df.loc[fraud_indices, 'loan_id'] = df.loc[fraud_indices, 'loan_id'].apply(lambda x: x[:-1] + '5')

# --- ADDING VARIANCE TO FRAUDULENT CASES ---
# Add noise to credit_score
df.loc[fraud_indices, 'credit_score'] = np.random.randint(300, 850, size=num_fraud)
# Add noise to loan_amount and collateral_value to ensure ltv_ratio variance
df.loc[fraud_indices, 'loan_amount'] += np.random.randint(-10000, 10000, size=num_fraud)
df.loc[fraud_indices, 'collateral_value'] += np.random.randint(-10000, 10000, size=num_fraud)

# Ensure no zero or negative collateral values
df['collateral_value'] = df['collateral_value'].clip(10000, 600000)

# Save to CSV
csv_path = '/Users/sangeetha/PythonProjects/GoAgentic/FraudDetectionAgent/mock_loan_data_fraud_variance_final.csv'
df.to_csv(csv_path, index=False)

print(f"✅ Final mock dataset with strong variance created: {csv_path}")