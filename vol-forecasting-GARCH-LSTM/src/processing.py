import datetime
import pandas as pd
import numpy as np
import os
from raw import load_data

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, 'data', 'processed')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

raw_data = load_data("1990-01-01", datetime.date.today())
SP500_data = raw_data["^GSPC"]
SP500_data = SP500_data.dropna()

log_returns = np.log(SP500_data / SP500_data.shift(1))
log_returns = log_returns.dropna()

realized_variance = log_returns ** 2

df_processed = pd.DataFrame({
    'Price': SP500_data,
    'Log_Returns': log_returns,
    'Realized_Variance': realized_variance
})

df_processed.to_csv(os.path.join(output_dir, "processed_data.csv"), index=True)
