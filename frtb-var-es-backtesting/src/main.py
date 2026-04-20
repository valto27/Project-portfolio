import os
import numpy as np
import pandas as pd
from typing import Dict
from joblib import Parallel, delayed
from src.config import CONFIG
from src.data import load_prices, to_returns
from src.models import rolling_forecasts, ForecastResult
from src.backtests import kupiec_pof, christoffersen_independence, christoffersen_cc, traffic_light_zone, duration_based_test, quantile_score, fz_score, acerbi_szekely_test
from src.plotting import plot_var_es_professional, sanitize_name

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_for_asset(name: str, price: pd.Series, cfg: Dict) -> None:
    out_dir = os.path.join(cfg["output_dir"], sanitize_name(name))
    ensure_dir(out_dir)
    
    ret = to_returns(price, log=cfg["use_log_returns"])
    if cfg["dropna_after_returns"]: ret = ret.dropna()
    
    alpha, window_size = cfg["alpha"], cfg["window_size"]
    methods = ["param_ewma", "param_t", "hist_plain", "hist_weighted", "hist_fhs_garch", "evt_only"]
    results = []

    for h in cfg["horizons"]:
        for method in methods:
            try: 
                results.append(rolling_forecasts(ret, method, alpha, window_size, h, cfg, progress_label=f"[{name}]"))
            except Exception as e: 
                print(f"[WARN] {name}: method={method}, h={h} failed: {e}")

    summary_rows = []
    for fr in results:
        y, q, e, ex = fr.realized_loss.values, fr.VaR.values, fr.ES.values, fr.exceptions.values
        pof=kupiec_pof(ex,alpha); ind=christoffersen_independence(ex); cc=christoffersen_cc(ex,alpha)
        dur=duration_based_test(ex,alpha); qsc=quantile_score(y,q,alpha); fzs=fz_score(y,q,e,alpha); esb=acerbi_szekely_test(y, q, e, alpha)

        n_obs_valid = int(np.isfinite(q).sum())
        n_exceptions = int(np.nansum(ex))
        zone = traffic_light_zone(n_exceptions, n_obs_valid, alpha)
        
        summary_rows.append({
            "Asset":name,"Method":fr.name,"Horizon":fr.horizon,
            "Obs":int(np.isfinite(q).sum()),"Exceptions":int(np.nansum(ex)),
            "Kupiec_LR":pof["LR"],"Kupiec_p":pof["pval"],
            "Christof_IND_LR":ind["LR"],"Christof_IND_p":ind["pval"],
            "Christof_CC_LR":cc["LR"],"Christof_CC_p":cc["pval"],
            "Traffic_Light": zone,
            "Duration_LR":dur["LR"],"Duration_p":dur["pval"],
            "QuantileScore":qsc,"FZ_score":fzs,
            "Acerbi_Z2_stat": esb["stat"],
            "Acerbi_Z2_p": esb["pval"]
        })

        df_out = pd.concat([fr.VaR, fr.ES, fr.realized_loss, fr.exceptions], axis=1)
        df_out.to_excel(os.path.join(out_dir, f"{sanitize_name(fr.name)}_h{fr.horizon}_series.xlsx"))

        if cfg["plots"]:
            try:
                plot_var_es_professional(fr, name, out_dir, summary_rows[-1])
            except Exception as e:
                print(f"[WARN] Could not plot for {fr.name}: {e}")

    pd.DataFrame(summary_rows).to_excel(os.path.join(out_dir, "summary_backtests.xlsx"), index=False)
    print(f"Finished {name}")

def main():
    cfg = CONFIG
    ensure_dir(cfg["output_dir"])
    
    print("Loading prices...")
    all_prices = load_prices(cfg)
    print(f"Loaded {len(all_prices)} assets: {list(all_prices.keys())}")
    
    tasks = [(a, all_prices[a]) for a in cfg["target_assets"]]
    print(f"Running for {len(tasks)} assets...")

    for name, price in tasks:
        run_for_asset(name, price, cfg)

if __name__ == "__main__":
    main()
