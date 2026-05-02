import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, 'results', 'plots')
prediction_dir = os.path.join(base_dir, 'results', 'predictions')
metrics_dir = os.path.join(base_dir, 'results', 'metrics')


for d in [output_dir, metrics_dir, prediction_dir]:
    if not os.path.exists(d):
        os.makedirs(d)



garch_metrics = pd.read_csv(os.path.join(metrics_dir, "garch_metrics.csv"))
lstm_metrics = pd.read_csv(os.path.join(metrics_dir, "lstm_metrics.csv"))

average_garch = garch_metrics[garch_metrics['Fold'] == 'Average'].iloc[0]
average_lstm = lstm_metrics[lstm_metrics['Fold'] == 'Average'].iloc[0]

data = {
    'Model': ['GARCH', 'LSTM'],
    'RMSE': [average_garch['RMSE'], average_lstm['RMSE']],
    'MAE': [average_garch['MAE'], average_lstm['MAE']]
}

table = pd.DataFrame(data)

improvement_rmse = ((average_garch['RMSE'] - average_lstm['RMSE']) / average_garch['RMSE']) * 100

barchart = table.plot(x='Model', y=['RMSE', 'MAE'], kind='bar', figsize=(10, 6), title='Model Performance Comparison')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
plt.close()


print("-" * 40)
print(f"GARCH  — RMSE: {average_garch['RMSE']:.6f} | MAE: {average_garch['MAE']:.6f}")
print(f"LSTM   — RMSE: {average_lstm['RMSE']:.6f} | MAE: {average_lstm['MAE']:.6f}")
print(f"Miglioramento RMSE: {improvement_rmse:.2f}%")


garch_predictions = pd.read_csv(os.path.join(prediction_dir, "garch_predictions_fold_5.csv"), index_col=0, parse_dates=True)
lstm_predictions = pd.read_csv(os.path.join(prediction_dir, "lstm_predictions_fold_5.csv"), index_col=0, parse_dates=True)
predictions = pd.DataFrame({
    'y_true': garch_predictions['y_true'],
    'y_pred_garch': garch_predictions['y_pred_garch'], 
    'y_pred_lstm': lstm_predictions['y_pred_lstm']
}, index=garch_predictions.index)
predictions = predictions.dropna()

plt.figure(figsize=(14, 6))
plt.plot(predictions.index, predictions['y_true'], color='black', label='Realized Variance')
plt.plot(predictions.index, predictions['y_pred_garch'], color='blue', label='GARCH')
plt.plot(predictions.index, predictions['y_pred_lstm'], color='red', label='LSTM')
plt.title('GARCH vs LSTM — Volatility Forecasting (Fold 5)')
plt.xlabel('Date')
plt.ylabel('Variance')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "predictions_comparison.png"))
plt.close()


processed_data = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'processed_data.csv'), index_col=0, parse_dates=True)
mean_vol = np.mean(processed_data['Log_Returns'])
dev_std = np.std(processed_data['Log_Returns'])
plt.figure(figsize=(10, 6))
plt.hist(processed_data['Log_Returns'], bins=50, density=True, alpha=0.6, color='g', label='Log Returns')
x = np.linspace(min(processed_data['Log_Returns']), max(processed_data['Log_Returns']), 100) 
plt.plot(x, stats.norm.pdf(x, mean_vol, dev_std), color='red', label='Normale')
plt.title('Distribution of Log Returns')
plt.xlabel('Log Returns')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "log_returns_distribution.png"))
plt.close()