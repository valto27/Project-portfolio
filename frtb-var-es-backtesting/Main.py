#Packages and Variables
import os
import re
import math
import warnings
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, binom
from scipy.optimize import minimize
from joblib import Parallel, delayed
from numba import jit

CONFIG = {
    "excel_path": "Dati 1.xlsx",
    "sheet_names": [
        "ACWI US Equity", "MXWD Index", "USGG1M Index", "VIX Index", "EURUSD Curncy",
        "S&P 500", "MSCI Emerging", "USGG3M Index", "MSCI Europe", "USGG10YR Index", "VXEEM",
    ],
    "date_col_letter": "B",
    "price_col_letter": "C",
    "start_row": 3,
    "target_assets": ["ACWI US Equity"], 

    "use_log_returns": True,
    "dropna_after_returns": True,

    "alpha": 0.99,          
    "window_size": 400,     
    "horizons": [1, 10],    

    "n_jobs": -1,           
    "progress_every": 200,  
    "REFIT_FREQ": 50,       

    "ewma_lambda": 0.94,
    "garch_min_omega": 1e-8,
    "garch_bounds": ((1e-7, None), (0.01, 0.99), (0.01, 0.99)), 
    
    "hist_lambda_grid": [0.90, 0.93, 0.95, 0.97, 0.98, 0.985, 0.99, 0.995],
    "hist_val_frac": 0.2,
    "mc_sims_1d": 5000,
    "mc_sims_10d": 10000,   
    "mc_seed": 1234,
    
    "use_evt": True,
    "evt_u_quantile": 0.95,
    "evt_min_excess": 80,
    "evt_tail_mix": 0.30,
    
    "param_t_sims": 5000,

    "output_dir": "risk_output_v3_stable",
    "plots": True
}
#FUNCTIONS
@jit(nopython=True)
def _numba_ewma_loop(ret_arr: np.ndarray, lam: float, start_var: float) -> np.ndarray:
    n = len(ret_arr)
    sigmas = np.zeros(n)
    var = start_var
    for i in range(n):
        r = ret_arr[i]
        var = lam * var + (1 - lam) * r * r
        if var < 1e-12: var = 1e-12
        sigmas[i] = math.sqrt(var)
    return sigmas

@jit(nopython=True)
def _numba_garch_recursion(ret: np.ndarray, omega: float, alpha: float, beta: float, start_var: float) -> np.ndarray:
    n = len(ret)
    sig2 = np.zeros(n)
    sig2[0] = start_var
    for t in range(1, n):
        val = omega + alpha * ret[t-1]**2 + beta * sig2[t-1]
        sig2[t] = val if val > 1e-12 else 1e-12
    return sig2

@jit(nopython=True)
def _numba_garch_loglike(params: np.ndarray, ret: np.ndarray, start_var: float) -> float:
    omega, alpha, beta = params[0], params[1], params[2]
    if alpha + beta >= 0.999: return 1e10 
    
    n = len(ret)
    sig2 = np.zeros(n)
    sig2[0] = start_var
    ll = 0.0
    
    for t in range(1, n):
        val = omega + alpha * ret[t-1]**2 + beta * sig2[t-1]
        s2 = val if val > 1e-12 else 1e-12
        sig2[t] = s2
        ll += -0.5 * (math.log(2 * 3.1415926535) + math.log(s2) + (ret[t]**2) / s2)
        
    return -ll 

@jit(nopython=True)
def _numba_student_t_mle(params: np.ndarray, residuals: np.ndarray) -> float:
    df = params[0]
    if df <= 2.01 or df > 100: return 1e10

    n = len(residuals)
    c = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(math.pi * (df - 2))
    
    ll = 0.0
    for i in range(n):
        term = 1 + (residuals[i]**2) / (df - 2)
        ll += c - ((df + 1) / 2) * math.log(term)
        
    return -ll

@jit(nopython=True)
def _numba_simulate_paths_garch(horizon: int, sims: int, last_sig2: float, last_ret: float, 
                                omega: float, alpha: float, beta: float, 
                                innovations: np.ndarray) -> np.ndarray:
    losses = np.zeros(sims)
    for i in range(sims):
        curr_sig2 = last_sig2
        curr_ret = last_ret
        path_loss = 0.0
        for h in range(horizon):
            next_sig2 = omega + alpha * (curr_ret**2) + beta * curr_sig2
            if next_sig2 < 1e-12: next_sig2 = 1e-12
            next_sig = math.sqrt(next_sig2)
            z = innovations[i, h]
            ret_sim = z * next_sig
            path_loss += -ret_sim
            curr_sig2 = next_sig2
            curr_ret = ret_sim
        losses[i] = path_loss
    return losses

@jit(nopython=True)
def _numba_simulate_paths_ewma(horizon: int, sims: int, last_sig2: float, 
                               lam: float, innovations: np.ndarray) -> np.ndarray:
    losses = np.zeros(sims)
    for i in range(sims):
        curr_sig2 = last_sig2
        path_loss = 0.0
        for h in range(horizon):
            curr_sig = math.sqrt(curr_sig2)
            z = innovations[i, h]
            ret_sim = z * curr_sig
            path_loss += -ret_sim
            curr_sig2 = lam * curr_sig2 + (1 - lam) * (ret_sim**2)
            if curr_sig2 < 1e-12: curr_sig2 = 1e-12
        losses[i] = path_loss
    return losses

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def sanitize_name(name: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]", "_", name)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s

def read_sheet_series(path: str, sheet_name: str, date_col: str, price_col: str, start_row: int) -> pd.Series:
    skip = max(0, start_row - 1)
    usecols = f"{date_col}:{price_col}"
    df = pd.read_excel(path, sheet_name=sheet_name, usecols=usecols, skiprows=skip, header=None, names=["Date", "Price"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    if not np.issubdtype(df["Price"].dtype, np.number):
        df["Price"] = pd.to_numeric(
            df["Price"].astype(str).str.replace(",", ".", regex=False).str.replace(r"[^\d.-]", "", regex=True),
            errors="coerce"
        )
    df = df.dropna().drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
    series = df.set_index("Date")["Price"].astype(float)
    if series.empty: raise ValueError(f"Sheet '{sheet_name}' produced an empty series.")
    return series

def to_returns(price: pd.Series, log: bool = True) -> pd.Series:
    r = np.log(price / price.shift(1)) if log else price.pct_change()
    return r.rename("ret")

def ewma_sigma_path(ret_window: np.ndarray, lam: float) -> Tuple[np.ndarray, float]:
    if len(ret_window) == 0: return np.array([]), np.nan
    var_start = np.nanvar(ret_window[:50], ddof=1) if len(ret_window) >= 50 else np.nanvar(ret_window, ddof=1)
    var_start = max(var_start, 1e-12)
    sigmas = _numba_ewma_loop(ret_window, lam, var_start)
    last_var = sigmas[-1]**2
    next_var = lam * last_var + (1 - lam) * ret_window[-1]**2
    return sigmas, math.sqrt(next_var)

def garch_fit_opt(ret: np.ndarray, min_omega: float) -> Tuple[float, float, float]:
    if len(ret) < 50: 
        return max(min_omega, 0.01 * np.nanvar(ret)), 0.05, 0.90
    start_var = np.nanvar(ret, ddof=1)
    x0 = [max(min_omega, 0.05*start_var), 0.1, 0.85]
    cons = ({'type': 'ineq', 'fun': lambda x:  0.999 - x[1] - x[2]})
    bounds = CONFIG["garch_bounds"]
    try:
        res = minimize(
            _numba_garch_loglike, x0, args=(ret, start_var),
            method='SLSQP', bounds=bounds, constraints=cons,
            tol=1e-6, options={'disp': False, 'maxiter': 50}
        )
        if res.success: return res.x[0], res.x[1], res.x[2]
        else: return x0[0], x0[1], x0[2]
    except Exception: return x0[0], x0[1], x0[2]

def garch_in_sample_vol(ret: np.ndarray, omega: float, a: float, b: float) -> np.ndarray:
    start_var = np.nanvar(ret, ddof=1)
    sig2 = _numba_garch_recursion(ret, omega, a, b, start_var)
    return np.sqrt(sig2)

def garch_forecast_path(last_sig2: float, last_ret: float, omega: float, a: float, b: float, horizon: int) -> np.ndarray:
    sig2_path = np.zeros(horizon)
    sig2_path[0] = max(omega + a * (last_ret**2) + b * last_sig2, 1e-12)
    for k in range(1, horizon): 
        sig2_path[k] = max(omega + (a + b) * sig2_path[k-1], 1e-12)
    return sig2_path

def gpd_fit_pwm(excesses: np.ndarray) -> Tuple[float, float]:
    y = np.sort(excesses[np.isfinite(excesses)]); n = len(y)
    if n < 3: return np.nan, np.nan
    b0 = np.mean(y); i = np.arange(1, n + 1); w = (i - 0.35) / n; b1 = np.sum(w * y) / np.sum(w)
    if not np.isfinite(b0) or not np.isfinite(b1) or b1 == b0: return np.nan, np.nan
    xi = (2 * b1 - b0) / (b0 - b1); beta = (2 * b0 * b1) / (b0 - b1)
    if beta <= 0 or not np.isfinite(xi) or not np.isfinite(beta): return np.nan, np.nan
    return float(xi), float(beta)

def gpd_var_es(u: float, xi: float, beta: float, p_exceed: float, alpha: float) -> Tuple[float, float]:
    p_tail = 1 - alpha
    if p_exceed <= 0 or not np.isfinite(xi) or not np.isfinite(beta): return np.nan, np.nan
    if xi == 0: q = u - beta * math.log(p_tail / p_exceed) 
    else: q = u + (beta / xi) * (((p_tail / p_exceed)**(-xi)) - 1.0)
    if xi >= 0.98: es = np.nan 
    else: es = (q + (beta - xi * u)) / (1 - xi)
    return float(q), float(es)

def hist_var_es(loss_window: np.ndarray, alpha: float) -> Tuple[float, float]:
    x = loss_window[np.isfinite(loss_window)];
    if x.size < 5: return np.nan, np.nan
    q = np.quantile(x, alpha); es = x[x >= q].mean() if (x >= q).any() else np.nan
    return float(q), float(es)

def weighted_quantile(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    order = np.argsort(values); v, w = values[order], weights[order]; cw = np.cumsum(w)
    total = cw[-1] if cw.size > 0 else 1.0; cw /= total
    idx = int(np.clip(np.searchsorted(cw, alpha, side="left"), 0, len(v) - 1)) if len(v) > 0 else 0
    return float(v[idx]) if len(v) > 0 else np.nan

def weighted_hist_var_es(loss_window: np.ndarray, alpha: float, lam: float) -> Tuple[float, float]:
    x = loss_window[np.isfinite(loss_window)]; n = x.size
    if n < 5: return np.nan, np.nan
    exponents = np.arange(n - 1, -1, -1); w = (1 - lam) * np.power(lam, exponents); var = weighted_quantile(x, w, alpha)
    order = np.argsort(x); xv, wv = x[order], w[order]; mask = xv >= var
    es = (xv[mask] * wv[mask]).sum() / wv[mask].sum() if mask.any() else np.nan
    return float(var), float(es)

def tune_lambda_weighted(loss_window: np.ndarray, alpha: float, grid: List[float], val_frac: float) -> float:
    x = loss_window[np.isfinite(loss_window)];
    if len(x) < 50: return 0.97
    split = int(max(30, len(x) * (1 - val_frac))); train, val = x[:split], x[split:]
    best_score, best_lambda = np.inf, 0.97
    for lam in grid:
        exponents = np.arange(len(train) - 1, -1, -1); w = (1 - lam) * np.power(lam, exponents)
        q = weighted_quantile(train, w, alpha); u = val - q; score = np.mean((alpha - (u < 0).astype(float)) * u)
        if score < best_score: best_score, best_lambda = score, lam
    return float(best_lambda)

def simulate_student_t_unit_var(df: float, size: tuple, rng: np.random.Generator) -> np.ndarray:
    Z = rng.normal(0.0, 1.0, size=size)
    G = rng.gamma(shape=df/2.0, scale=2.0, size=size)
    T = Z / np.sqrt(G / df) 
    if df > 2: T = T / np.sqrt(df / (df - 2.0))
    return T

def normal_var_es(sigma: float, alpha: float) -> Tuple[float, float]:
    z_alpha = norm.ppf(alpha); var = z_alpha * sigma; es = sigma * norm.pdf(z_alpha) / (1 - alpha)
    return float(var), float(es)

def build_evt_params(window: np.ndarray, garch_params: Optional[Dict] = None) -> Dict:
    if garch_params and "omega" in garch_params:
        sigmas_in = garch_in_sample_vol(window, garch_params["omega"], garch_params["a"], garch_params["b"])
    else:
        sigmas_in, _ = ewma_sigma_path(window, CONFIG["ewma_lambda"])
        
    z = window / np.where(sigmas_in > 0, sigmas_in, np.nan); z = z[np.isfinite(z)]
    if len(z) < 200: return {"ok": False}
    Lz = -z; u = float(np.quantile(Lz, CONFIG["evt_u_quantile"])); excess = Lz[Lz > u] - u
    if len(excess) < CONFIG["evt_min_excess"]: return {"ok": False}
    xi, beta = gpd_fit_pwm(excess)
    if not np.isfinite(xi) or not np.isfinite(beta) or beta <= 0: return {"ok": False}
    p_exceed = float(len(excess) / len(Lz))
    return {"ok": True, "u": u, "xi": xi, "beta": beta, "p_exceed": p_exceed}

def fhs_with_garch(window: np.ndarray, alpha: float, refit: Dict, horizon: int, rng: np.random.Generator, use_evt: bool, evt_params: Optional[Dict]) -> Tuple[float, float]:
    if "omega" not in refit:
        omega, a, b = garch_fit_opt(window, CONFIG["garch_min_omega"])
        refit.update({"omega": omega, "a": a, "b": b})
    else: omega, a, b = refit["omega"], refit["a"], refit["b"]
    
    sigmas_in_sample = garch_in_sample_vol(window, omega, a, b)
    last_sig2 = sigmas_in_sample[-1]**2
    z = window / np.where(sigmas_in_sample > 0, sigmas_in_sample, np.nan)
    z = z[np.isfinite(z)]; Lz = -z 
    if len(z) < 80: return np.nan, np.nan

    sims = CONFIG["mc_sims_10d"] if horizon > 1 else CONFIG["mc_sims_1d"]
    
    if use_evt and evt_params and evt_params.get("ok", False):
        boot_idx = rng.integers(0, len(Lz), size=(sims, horizon))
        z_innovations = Lz[boot_idx]
        evt_mask = rng.random(size=(sims, horizon)) < CONFIG["evt_tail_mix"]
        if np.any(evt_mask):
            xi, beta, u = evt_params["xi"], evt_params["beta"], evt_params["u"]
            U_vals = rng.random(size=np.count_nonzero(evt_mask))
            U_vals = np.clip(U_vals, 1e-12, 1-1e-12)
            if xi != 0: y_evt = (beta / xi) * (U_vals**(-xi) - 1.0)
            else: y_evt = -beta * np.log(U_vals)
            z_innovations[evt_mask] = u + y_evt
    else:
        z_innovations = rng.choice(Lz, size=(sims, horizon))
        
    z_innovations = -z_innovations
    sim_losses = _numba_simulate_paths_garch(horizon, sims, last_sig2, window[-1], omega, a, b, z_innovations)
    return hist_var_es(sim_losses, alpha)

def param_t_method(window: np.ndarray, alpha: float, lam: float, df_grid: List[float], sims: int, rng: np.random.Generator, horizon: int) -> Tuple[float, float]:
    sigmas, sigma_next = ewma_sigma_path(window, lam)
    z = window / np.where(sigmas > 0, sigmas, np.nan)
    z = z[np.isfinite(z)]
    
    if len(z) < 50 or not np.isfinite(sigma_next): 
        return np.nan, np.nan
    
    res = minimize(
        _numba_student_t_mle, 
        x0=[8.0],             
        args=(z,), 
        method='L-BFGS-B',    
        bounds=[(2.1, 30.0)], 
        tol=1e-4
    )
    best_df = res.x[0] if res.success else 8.0
    
    innovations = simulate_student_t_unit_var(best_df, size=(sims, horizon), rng=rng)
    if horizon == 1: 
        losses = -innovations.flatten() * sigma_next
    else: 
        losses = _numba_simulate_paths_ewma(horizon, sims, sigma_next**2, lam, innovations)
        
    return hist_var_es(losses, alpha)

def kupiec_pof(exceptions: np.ndarray, alpha: float) -> Dict[str, float]:
    mask = np.isfinite(exceptions); e, n = exceptions[mask], int(np.sum(mask)); x = int(np.nansum(e))
    if n == 0: return {"x": np.nan, "n": 0, "LR": np.nan, "pval": np.nan}
    p = 1 - alpha; phat = np.clip(x / n, 1e-12, 1 - 1e-12)
    LR = -2 * ((n - x) * math.log(1 - p) + x * math.log(p) - ((n - x) * math.log(1 - phat) + x * math.log(phat)))
    pval = 1 - chi2.cdf(LR, df=1)
    return {"x": x, "n": n, "LR": float(LR), "pval": float(pval)}

def christoffersen_independence(exceptions: np.ndarray) -> Dict[str, float]:
    e = exceptions[np.isfinite(exceptions)].astype(int);
    if len(e) < 5: return {"LR": np.nan, "pval": np.nan}
    e_prev = e[:-1]; e_curr = e[1:]
    n00 = ((e_prev==0) & (e_curr==0)).sum()
    n01 = ((e_prev==0) & (e_curr==1)).sum()
    n10 = ((e_prev==1) & (e_curr==0)).sum()
    n11 = ((e_prev==1) & (e_curr==1)).sum()
    p01=n01/(n00+n01+1e-12); p11=n11/(n10+n11+1e-12); pi=(n01+n11)/(n00+n01+n10+n11+1e-12)
    L0=((1-pi)**(n00+n10))*(pi**(n01+n11)); L1=((1-p01)**n00)*(p01**n01)*((1-p11)**n10)*(p11**n11)
    if L1==0 or L0==0: return {"LR": np.nan, "pval": np.nan}
    LR=-2*math.log(L0/L1); pval=1-chi2.cdf(LR,df=1); return {"LR":float(LR),"pval":float(pval)}

def christoffersen_cc(exceptions: np.ndarray, alpha: float) -> Dict[str, float]:
    pof_test = kupiec_pof(exceptions, alpha); ind_test = christoffersen_independence(exceptions)
    if not np.isfinite(pof_test["LR"]) or not np.isfinite(ind_test["LR"]): return {"LR": np.nan, "pval": np.nan}
    LRcc = pof_test["LR"] + ind_test["LR"]; pval = 1 - chi2.cdf(LRcc, df=2); return {"LR": float(LRcc), "pval": float(pval)}

def traffic_light_zone(n_exceptions: int, n_obs: int, alpha: float) -> str:
    if n_obs == 0: return "N/A"
    p = 1.0 - alpha
    cum_prob = binom.cdf(n_exceptions, n_obs, p)
    
    if cum_prob < 0.95:
        return "Green"
    elif cum_prob < 0.9999:
        return "Yellow"
    else:
        return "Red"


def duration_based_test(exceptions: np.ndarray, alpha: float) -> Dict[str, float]:
    e = exceptions[np.isfinite(exceptions)].astype(int)
    if e.sum() == 0 or len(e) < 5: return {"LR": np.nan, "pval": np.nan, "mean_D": np.nan}
    idx = np.where(e == 1)[0]; D = np.diff(np.concatenate(([-1], idx))); D = D[D > 0]
    if len(D) < 2: return {"LR": np.nan, "pval": np.nan, "mean_D": float(D.mean() if len(D) else np.nan)}
    p = 1 - alpha; phat = 1.0 / max(np.mean(D), 1e-12)
    l0 = np.sum((D - 1) * np.log(1 - p + 1e-12) + np.log(p + 1e-12))
    l1 = np.sum((D - 1) * np.log(1 - phat + 1e-12) + np.log(phat + 1e-12))
    LR = -2 * (l0 - l1); pval = 1 - chi2.cdf(LR, df=1)
    return {"LR": float(LR), "pval": float(pval), "mean_D": float(np.mean(D))}

def quantile_score(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    mask = np.isfinite(y) & np.isfinite(q);
    if not mask.any(): return np.nan
    u = y[mask] - q[mask]; return float(((alpha - (u < 0).astype(float)) * u).mean())

def fz_score(losses: np.ndarray, VaR: np.ndarray, ES: np.ndarray, alpha: float) -> float:
    mask = np.isfinite(losses) & np.isfinite(VaR) & np.isfinite(ES)
    if not mask.any(): return np.nan
    L, Q, E = losses[mask], VaR[mask], ES[mask]; I = (L > Q).astype(float)
    score = (I - (1 - alpha)) * (Q - L) + I * (E - Q) / (1 - alpha)
    return float(np.mean(score))

def acerbi_szekely_test(losses: np.ndarray, VaR: np.ndarray, ES: np.ndarray, alpha: float, B: int=1000, rng: Optional[np.random.Generator]=None) -> Dict[str, float]:
    if rng is None: rng = np.random.default_rng(12345)
    mask = np.isfinite(losses) & np.isfinite(VaR) & np.isfinite(ES)
    L, Q, E = losses[mask], VaR[mask], ES[mask]
    exceedances_idx = L > Q
    tail_losses = L[exceedances_idx]
    tail_es = E[exceedances_idx]
    n_ex = len(tail_losses)
    
    if n_ex < 5:
        return {"stat": np.nan, "pval": np.nan, "n_ex": int(n_ex)}
    
    ratios = tail_losses / tail_es
    z2_stat = np.mean(ratios) - 1.0
    
    ratios_centered = ratios - z2_stat
    
    bs_stats = np.zeros(B)
    for i in range(B):
        sample_ratios = rng.choice(ratios_centered, size=n_ex, replace=True)
        bs_stats[i] = np.mean(sample_ratios) - 1.0
    pval = np.mean(np.abs(bs_stats) >= np.abs(z2_stat))
    
    return {"stat": float(z2_stat), "pval": float(pval), "n_ex": int(n_ex)}



class ForecastResult:
    def __init__(self, name, horizon, VaR, ES, realized_loss, exceptions):
        self.name, self.horizon, self.VaR, self.ES, self.realized_loss, self.exceptions = name, horizon, VaR, ES, realized_loss, exceptions

def rolling_forecasts(ret: pd.Series, method: str, alpha: float, window_size: int, horizon: int, cfg: Dict, progress_label: str = "") -> ForecastResult:
    r_values, dates = ret.dropna().values, ret.dropna().index; n_obs = len(r_values)
    if n_obs < window_size + horizon + 5: raise ValueError("Not enough data.")
    
    VaR, ES, exceptions, realized = np.full(n_obs,np.nan), np.full(n_obs,np.nan), np.full(n_obs,np.nan), np.full(n_obs,np.nan)
    rng = np.random.default_rng(cfg.get("mc_seed", 1234))
    refit_cache = {}; tuned_lambda = None; evt_params = None
    total_steps = (n_obs - horizon) - window_size + 1
    progress_every = max(int(cfg.get("progress_every", 200)), 1)

    for t in range(window_size, n_obs - horizon + 1):
        idx = t + horizon - 1
        window = r_values[t - window_size:t]
        realized[idx] = -float(np.sum(r_values[t:t + horizon]))
        if (t == window_size) or ((t - window_size) % cfg["REFIT_FREQ"] == 0):
            omega, a, b = garch_fit_opt(window, cfg["garch_min_omega"])
            refit_cache = {"omega": omega, "a": a, "b": b} 
            if cfg["use_evt"]: 
                evt_params = build_evt_params(window, refit_cache)   
            tuned_lambda = tune_lambda_weighted(-window, alpha, cfg["hist_lambda_grid"], cfg["hist_val_frac"])
        var_forecast, es_forecast = np.nan, np.nan
        if method == "param_ewma":
            _, sigma_next = ewma_sigma_path(window, cfg["ewma_lambda"])
            if horizon == 1: 
                var_forecast, es_forecast = normal_var_es(sigma_next, alpha)
            else: 
                innovations = rng.normal(size=(cfg["mc_sims_10d"], horizon))
                losses = _numba_simulate_paths_ewma(horizon, cfg["mc_sims_10d"], sigma_next**2, cfg["ewma_lambda"], innovations)
                var_forecast, es_forecast = hist_var_es(losses, alpha)

        elif method == "param_t":
            var_forecast, es_forecast = param_t_method(window, alpha, cfg["ewma_lambda"], cfg.get("t_df_grid", []), cfg["param_t_sims"], rng, horizon)

        elif method == "hist_plain":
            if horizon == 1: 
                var_forecast, es_forecast = hist_var_es(-window, alpha)
            else: 
                blocks = np.array([window[i:i+horizon].sum() for i in range(len(window) - horizon + 1)])
                var_forecast, es_forecast = hist_var_es(-blocks, alpha)

        elif method == "hist_weighted":
            lam = tuned_lambda if tuned_lambda is not None else 0.97
            if horizon == 1:
                var_forecast, es_forecast = weighted_hist_var_es(-window, alpha, lam)
            else:
                blocks = np.array([window[i:i+horizon].sum() for i in range(len(window) - horizon + 1)])
                var_forecast, es_forecast = weighted_hist_var_es(-blocks, alpha, lam)            

        elif method == "hist_fhs_garch":
            var_forecast, es_forecast = fhs_with_garch(window, alpha, refit_cache, horizon, rng, cfg["use_evt"], evt_params)

        elif method == "evt_only":
            if horizon == 1 and evt_params and evt_params["ok"]:
                _, sigma_next = ewma_sigma_path(window, cfg["ewma_lambda"])
                q, es = gpd_var_es(evt_params["u"], evt_params["xi"], evt_params["beta"], evt_params["p_exceed"], alpha)
                var_forecast, es_forecast = q * sigma_next, es * sigma_next
            else: 
                var_forecast, es_forecast = fhs_with_garch(window, alpha, refit_cache, horizon, rng, True, evt_params)
        
        VaR[idx], ES[idx] = var_forecast, es_forecast
        if np.isfinite(realized[idx]) and np.isfinite(VaR[idx]): 
            exceptions[idx] = 1.0 if realized[idx] > VaR[idx] else 0.0
        
        if cfg["n_jobs"] == 1 and (t - window_size) % progress_every == 0: 
            print(f"{progress_label} {method} h={horizon}: step {t - window_size + 1}/{total_steps}")
            
    return ForecastResult(method, horizon, pd.Series(VaR, index=dates), pd.Series(ES, index=dates), pd.Series(realized, index=dates), pd.Series(exceptions, index=dates))

STRESS_PERIODS = [
    ("2008-09-01", "2009-06-30", "#d62728", "GFC"),
    ("2011-07-01", "2012-06-30", "#8c564b", "Eurozone\nCrisis"),
    ("2020-02-01", "2020-06-30", "#ff7f0e", "COVID-19"),
    ("2022-01-01", "2022-12-31", "#9467bd", "Inflation\nShock"),
]

PLOT_COLORS = {
    "loss":      "#4878cf",
    "var":       "#e85d3a",
    "es":        "#2ca02c",
    "exception": "#d62728",
    "bg":        "#f9f9f9",
    "grid":      "#e0e0e0",
    "spine":     "#cccccc",
}

TRAFFIC_COLORS = {"Green": "#27ae60", "Yellow": "#f39c12", "Red": "#e74c3c", "N/A": "#95a5a6"}

METHOD_LABELS = {
    "param_ewma":     "Parametric EWMA (Normal)",
    "param_t":        "Parametric EWMA (Student-t)",
    "hist_plain":     "Historical Simulation",
    "hist_weighted":  "Weighted Historical (BRW)",
    "hist_fhs_garch": "Filtered Historical Simulation (FHS)",
    "evt_only":       "Extreme Value Theory (EVT)",
}

def plot_var_es_professional(fr, asset_name: str, out_dir: str, backtest_row: dict) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif":  ["Georgia", "Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 12, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "figure.dpi": 300,
    })

    dates      = fr.realized_loss.index
    loss       = fr.realized_loss.values
    var        = fr.VaR.values
    es         = fr.ES.values
    exceptions = fr.exceptions.values

    exc_mask   = np.isfinite(exceptions) & (exceptions == 1)
    exc_dates  = dates[exc_mask]
    exc_losses = loss[exc_mask]

    es_roll_std = fr.ES.dropna().rolling(60, min_periods=20).std().reindex(dates).values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.6], "hspace": 0.08})
    fig.patch.set_facecolor(PLOT_COLORS["bg"])

    def _shade_stress(ax, y_frac=0.97):
        ymin, ymax = ax.get_ylim()
        for s, e, color, label in STRESS_PERIODS:
            try:
                sd, ed = np.datetime64(s), np.datetime64(e)
                if sd > dates[-1] or ed < dates[0]: continue
                ax.axvspan(sd, ed, alpha=0.10, color=color, linewidth=0)
                mid = sd + (ed - sd) // 2
                ax.text(mid, ymin + (ymax - ymin) * y_frac, label,
                        ha="center", va="top", fontsize=7, color=color,
                        fontstyle="italic", alpha=0.85)
            except Exception:
                pass

    # ── Pannello 1: VaR ──────────────────────────────────────────────────
    ax1.set_facecolor(PLOT_COLORS["bg"])
    ax1.plot(dates, loss, color=PLOT_COLORS["loss"], lw=0.8, alpha=0.75, label="Realized loss", zorder=2)
    ax1.plot(dates, var,  color=PLOT_COLORS["var"],  lw=1.4, linestyle="--", label="VaR Forecast (99%)", zorder=3)
    if len(exc_dates):
        ax1.scatter(exc_dates, exc_losses, color=PLOT_COLORS["exception"],
                    s=22, zorder=5, label=f"Exceptions ({len(exc_dates)})", linewidths=0)
    ax1.axhline(0, color=PLOT_COLORS["spine"], lw=0.6, linestyle=":")
    ax1.set_ylabel("Loss (log-return)", labelpad=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax1.grid(True, color=PLOT_COLORS["grid"], lw=0.5); ax1.set_axisbelow(True)
    ax1.autoscale(axis="y")
    _shade_stress(ax1)

    # box statistiche
    n_ex     = backtest_row.get("Exceptions", len(exc_dates))
    tl       = backtest_row.get("Traffic_Light", "N/A")
    tl_color = TRAFFIC_COLORS.get(tl, "#95a5a6")
    def _p(v): return "n/a" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.3f}"
    stats = (f"Horizon : {fr.horizon}d\n"
             f"Exceptions : {n_ex}\n"
             f"Kupiec p   : {_p(backtest_row.get('Kupiec_p'))}\n"
             f"CC p       : {_p(backtest_row.get('Christof_CC_p'))}\n"
             f"AS Z\u2082 p   : {_p(backtest_row.get('Acerbi_Z2_p'))}\n"
             f"Traffic    : {tl}")
    ax1.text(0.012, 0.97, stats, transform=ax1.transAxes, fontsize=7.5, va="top",
             fontfamily="monospace", zorder=6,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor=tl_color, linewidth=1.5, alpha=0.92))

    ax1.set_title(f"{asset_name}  ·  {METHOD_LABELS.get(fr.name, fr.name)}",
                  fontsize=12, fontweight="bold", pad=10, loc="left")

    legend_h = [
        plt.Line2D([0],[0], color=PLOT_COLORS["loss"], lw=1.2, label="Realized loss"),
        plt.Line2D([0],[0], color=PLOT_COLORS["var"],  lw=1.4, linestyle="--", label="VaR 99%"),
        plt.Line2D([0],[0], color=PLOT_COLORS["exception"], lw=0, marker="o",
                   markersize=5, label=f"Exceptions ({n_ex})"),
        Patch(facecolor=tl_color, alpha=0.85, label=f"Traffic light: {tl}"),
    ] + [Patch(facecolor=c, alpha=0.25, label=lbl) for _, _, c, lbl in STRESS_PERIODS]
    ax1.legend(handles=legend_h, loc="upper right", framealpha=0.92,
               edgecolor=PLOT_COLORS["spine"], ncol=2, fontsize=8)
    for sp in ["top","right"]: ax1.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax1.spines[sp].set_color(PLOT_COLORS["spine"])

    # ── Pannello 2: ES ───────────────────────────────────────────────────
    ax2.set_facecolor(PLOT_COLORS["bg"])
    ax2.plot(dates, es, color=PLOT_COLORS["es"], lw=1.2, label="ES Forecast (99%)", zorder=3)
    valid = np.isfinite(es_roll_std) & np.isfinite(es)
    if valid.any():
        ax2.fill_between(dates,
                         np.where(valid, es - es_roll_std, np.nan),
                         np.where(valid, es + es_roll_std, np.nan),
                         alpha=0.15, color=PLOT_COLORS["es"], label="ES ±1σ (60d rolling)")
    ax2.axhline(0, color=PLOT_COLORS["spine"], lw=0.6, linestyle=":")
    ax2.set_ylabel("Expected Shortfall", labelpad=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax2.grid(True, color=PLOT_COLORS["grid"], lw=0.5); ax2.set_axisbelow(True)
    ax2.autoscale(axis="y")
    _shade_stress(ax2, y_frac=0.95)
    ax2.legend(loc="upper right", framealpha=0.92, edgecolor=PLOT_COLORS["spine"], fontsize=8)
    for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax2.spines[sp].set_color(PLOT_COLORS["spine"])

    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")
    ax2.set_xlabel("Date", labelpad=8)

    fname = os.path.join(out_dir, f"{sanitize_name(fr.name)}_h{fr.horizon}_VaR_ES.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight",
                facecolor=PLOT_COLORS["bg"], edgecolor="none")
    plt.close(fig)
    print(f"  [plot] saved → {fname}")


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
    
    all_prices = {
        sh: read_sheet_series(cfg["excel_path"], sh, cfg["date_col_letter"], cfg["price_col_letter"], cfg["start_row"])
        for sh in cfg["sheet_names"]
    }

    tasks = [(a, all_prices[a]) for a in cfg["target_assets"]]

    Parallel(n_jobs=cfg["n_jobs"], backend="threading")(
        delayed(run_for_asset)(name, price, cfg) for name, price in tasks
    )

if __name__ == "__main__":
    main()
