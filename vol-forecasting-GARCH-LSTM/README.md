## Volatility forecasting GARCH vs LSTM
GARCH(1,1) vs LSTM for daily volatility forecasting on S&P 500 (1990–2025).

## What this is

A benchmark comparing a classical statistical model against a neural network on the same task: predict next-day variance from S&P 500 returns. The LSTM loses. GARCH trains in 0.5 seconds, scores better on RMSE, and requires no GPU. The neural network takes several minutes and underperforms by 2.67%. Whether that gap is fixable with more tuning is an open question — but in a latency-sensitive environment, the answer probably doesn't matter.

## Dataset

S&P 500 daily closes via `yfinance` (`^GSPC`, auto-adjusted). Log-returns: `ln(Pt / Pt-1)`. Target variable: squared returns as a proxy for realized variance.

## Models

**GARCH(1,1)** — `arch` library, Normal distribution. Conditional variances from the full-series fit are passed as a feature to the LSTM. Training: ~0.5 seconds.

**LSTM** — 1 layer, hidden size 64, Softplus activation. 20-day input window with log-returns and GARCH conditional variance. Adam optimizer, MSELoss, early stopping with patience 10.

## Validation

`TimeSeriesSplit` with 5 folds, defined once and shared between both models. No shuffling, no leakage.

## Results

| Model | RMSE     | MAE      |
|-------|----------|----------|
| GARCH | 0.000473 | 0.000168 |
| LSTM  | 0.000485 | 0.000226 |

GARCH wins. "The LSTM reduces RMSE by X%" was supposed to go here — it didn't work out that way. The result is more useful for that reason.

## Limitations

- `r²` is a noisy variance proxy; tick-based realized variance would be cleaner
- GARCH has one convergence warning across 5 folds (t-Student may fix it)
- LSTM hyperparameters were not tuned beyond basic defaults

## Structure

data/raw            <- SP500_raw.csv
data/processed      <- processed_data.csv, garch_variances.csv
models/             <- lstm_volatility_model.pth
results/metrics     <- garch_metrics.csv, lstm_metrics.csv
results/predictions <- fold 5 predictions for both models
results/plots       <- comparison charts
src/                <- raw.py, processing.py, garch_model.py, lstm_model.py, evaluate.py

## How to run

pip install -r requirements.txt
cd src
python raw.py && python processing.py && python garch_model.py && python lstm_model.py && python evaluate.py