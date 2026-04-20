TICKER_MAP = {
    "ACWI US Equity":  "ACWI",
    "S&P 500":         "^GSPC",
    "MSCI Emerging":   "EEM",
    "MSCI Europe":     "IEUR",
    "VIX Index":       "^VIX",
    "EURUSD Curncy":   "EURUSD=X",
    "USGG10YR Index":  "^TNX",
    "USGG3M Index":    "^IRX",
    "USGG1M Index":    "^IRX",
    "MXWD Index":      "ACWI",
    "VXEEM":           "VXEEM",
}

CONFIG = {
    # Data source
    "use_yfinance":    True,
    "yf_start":        "2000-01-01",
    "yf_end":          None,
    "excel_path":      "Dati 1.xlsx",
    "sheet_names": [
        "ACWI US Equity", "S&P 500", "MSCI Emerging",
        "MSCI Europe", "VIX Index", "EURUSD Curncy",
        "USGG10YR Index", "USGG3M Index",
    ],
    "date_col_letter": "B",
    "price_col_letter": "C",
    "start_row":       3,

    # Scope
    "target_assets":   ["ACWI US Equity"],

    # Returns
    "use_log_returns":      True,
    "dropna_after_returns": True,

    # Risk parameters
    "alpha":       0.99,
    "window_size": 400,
    "horizons":    [1, 10],

    # Computation
    "n_jobs":         1,
    "progress_every": 200,
    "REFIT_FREQ":     50,

    # EWMA / GARCH
    "ewma_lambda":     0.94,
    "garch_min_omega": 1e-8,
    "garch_bounds":    ((1e-7, None), (0.01, 0.99), (0.01, 0.99)),

    # Weighted Historical
    "hist_lambda_grid": [0.90, 0.93, 0.95, 0.97, 0.98, 0.985, 0.99, 0.995],
    "hist_val_frac":    0.2,

    # Monte Carlo
    "mc_sims_1d":  5000,
    "mc_sims_10d": 10000,
    "mc_seed":     1234,

    # EVT
    "use_evt":        True,
    "evt_u_quantile": 0.95,
    "evt_min_excess": 80,
    "evt_tail_mix":   0.30,

    # Student-t
    "param_t_sims": 5000,

    # Output
    "output_dir": "results",
    "plots":      True,
}