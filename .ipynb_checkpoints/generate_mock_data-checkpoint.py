import pandas as pd
import numpy as np

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

# Save to CSV in your desired directory
csv_path = '/Users/sangeetha/PythonProjects/GoAgentic/FraudDetectionAgent/mock_loan_data.csv'
df.to_csv(csv_path, index=False)

print(f"✅ Mock dataset created: {csv_path}")
