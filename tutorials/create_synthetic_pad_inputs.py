import numpy as np
import pandas as pd

NUM_PATIENTS: int = 4452

np.random.seed(0)

# Create patients
df = pd.DataFrame({ 'id' : range(NUM_PATIENTS)})

# True PAD label
df['y'] = np.random.choice([0,1], size = df.shape[0])

# Model predictions
df['y_hat_dl'] = np.random.beta(df['y'] * 200 + (1 - df['y']) * 1, 8)
df['y_hat_rf'] = np.random.beta(df['y'] * 30 + (1 - df['y']) * 4, 8)
df['y_hat_lr'] = np.random.beta(df['y'] * 40 + (1 - df['y']) * 6, 8)

# ABI test prediction
df['abi_test_pred'] = np.random.normal(0.65 * df['y'] + (1 - df['y']) * 1.09, 0.15 * df['y'] + (1 - df['y']) * 0.11)

# Resource priority
df['random_resource_priority'] = np.random.choice(range(df.shape[0]), replace=False, size=df.shape[0])

# Save to CSV
df.to_csv('input/synthetic_pad_inputs.csv')