import os
import time
import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from processing import df_processed

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_dir = os.path.join(base_dir, 'data', 'processed')
metrics_dir = os.path.join(base_dir, 'results', 'metrics')

for d in [processed_dir, metrics_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

log_returns = df_processed['Log_Returns']
tscv = TimeSeriesSplit(n_splits=5)

start_time = time.time()

fold_results = []

for fold, (train_index, test_index) in enumerate(tscv.split(log_returns)):
    train, test = log_returns.iloc[train_index], log_returns.iloc[test_index]
    
    model = arch_model(train, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')

    forecasts = model_fit.forecast(horizon=len(test), reindex=False)
    pred_variances = forecasts.variance.values[-1, :]
    
    realized_variances = test**2

    rmse = np.sqrt(mean_squared_error(realized_variances, pred_variances))
    mae = mean_absolute_error(realized_variances, pred_variances)
    
    fold_results.append({'Fold': fold + 1, 'RMSE': rmse, 'MAE': mae})
    print(f"Fold {fold + 1} completato.")

total_training_time = time.time() - start_time

df_metrics = pd.DataFrame(fold_results)

mean_row = pd.DataFrame([{'Fold': 'Average', 
                          'RMSE': df_metrics['RMSE'].mean(), 
                          'MAE': df_metrics['MAE'].mean()}])
df_metrics = pd.concat([df_metrics, mean_row], ignore_index=True)

df_metrics.to_csv(os.path.join(metrics_dir, "garch_metrics.csv"), index=False)


full_model = arch_model(log_returns, vol='Garch', p=1, q=1, dist='Normal')
full_res = full_model.fit(disp='off')

garch_variances = pd.DataFrame({
    'garch_conditional_vol': full_res.conditional_volatility ** 2 
}, index=log_returns.index)

garch_variances.to_csv(os.path.join(processed_dir, "garch_variances.csv"), index=True)

print("-" * 30)
print(f"Training Time Totale: {total_training_time:.4f} secondi")
print(f"Metriche salvate in: {metrics_dir}")
print(f"Feature GARCH salvata in: {processed_dir}")