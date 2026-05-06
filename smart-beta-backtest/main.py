import pandas as pd
from src.data_loader import load_prices, load_risk_free
from src.metrics import compute_all
from src.signals import compute_momentum, compute_low_volatility
from src.backtest import run_backtest
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
metrics_dir = os.path.join(base_dir, 'results', 'metrics')
metrics_path = os.path.join(metrics_dir, 'metrics.csv')

prices = load_prices()
risk_free = load_risk_free()
risk_free_mean = risk_free.mean().values[0]

monthly_prices = prices.resample('ME').last()
monthly_returns = monthly_prices.pct_change()
momentum = compute_momentum(monthly_prices)
low_volatility = compute_low_volatility(monthly_returns)

signals = ['momentum', 'low_volatility']
methods = ['equal', 'markowitz', 'risk_parity']
cost_bps = 10

results_dict = {}
for signal_name in signals:
    signal = momentum if signal_name == 'momentum' else low_volatility
    for method in methods:
        print(f"Running {signal_name} - {method}...")
        results = run_backtest(prices, signal, method=method, cost_bps=cost_bps)
        metrics = compute_all(results['returns'], risk_free_mean)
        results_dict[f"{signal_name}_{method}"] = metrics
    
results_table = pd.DataFrame(results_dict).T
print(results_table.round(4))
results_table.to_csv(metrics_path)