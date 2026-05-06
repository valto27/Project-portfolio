import pandas as pd
import os
from src.portfolio import equal_weights, markowitz_weights, risk_parity_weights

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
results_path = os.path.join(data_dir, 'results.csv')

def run_backtest(prices, signal, method, cost_bps):
    prices = prices.resample('ME').last()
    yields = prices.pct_change()

    results = []
    prev_weights_long = None

    for date in signal.index:
        signal_t = signal.loc[date].dropna()
        if len(signal_t) == 0:
            continue

        n_long = max(1, int(len(signal_t) * 0.1))
        n_short = max(1, int(len(signal_t) * 0.1))

        historical_returns = yields.loc[:date]

        if method == 'equal':
            weights_long = equal_weights(signal_t.nlargest(n_long).index)
            weights_short = equal_weights(signal_t.nsmallest(n_short).index) * -1
        elif method == 'markowitz':
            weights_long = markowitz_weights(historical_returns[signal_t.nlargest(n_long).index])
            weights_short = markowitz_weights(historical_returns[signal_t.nsmallest(n_short).index]) * -1
        elif method == 'risk_parity':
            weights_long = risk_parity_weights(historical_returns[signal_t.nlargest(n_long).index])
            weights_short = risk_parity_weights(historical_returns[signal_t.nsmallest(n_short).index]) * -1
        
        if prev_weights_long is None:
            turnover = 1.0
        else:
            diff = weights_long.sub(prev_weights_long, fill_value=0)
            turnover = diff.abs().sum() / 2

        date_idx = yields.index.get_loc(date)
        if date_idx + 1 >= len(yields):
            break
        next_returns = yields.iloc[date_idx + 1]

        ret_long = (weights_long * next_returns[weights_long.index]).sum()
        ret_short = (weights_short * next_returns[weights_short.index]).sum()
        ret_gross = ret_long + ret_short

        cost = turnover * cost_bps / 10000
        ret_net = ret_gross - cost

        results.append({
            'date': date,
            'returns': ret_net,
            'turnover': turnover})
        
        prev_weights_long = weights_long
        
    results_df = pd.DataFrame(results).set_index('date')
    results_df.to_csv(results_path)
    return results_df
    

if __name__ == "__main__":
    from src.data_loader import load_prices
    prices = load_prices()
    
    yields = prices.pct_change()

    from src.signals import compute_momentum
    signal = compute_momentum(prices.resample('ME').last())

    results = run_backtest(prices, signal, method='equal', cost_bps=10)
    print(results.shape)
    print(results.head())