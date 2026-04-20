import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from numba import jit
from typing import Dict, List, Tuple, Optional



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


# Volatility functions
def ewma_sigma_path(ret_window: np.ndarray, lam: float) -> Tuple[np.ndarray, float]:
    if len(ret_window) == 0: return np.array([]), np.nan
    var_start = np.nanvar(ret_window[:50], ddof=1) if len(ret_window) >= 50 else np.nanvar(ret_window, ddof=1)
    var_start = max(var_start, 1e-12)
    sigmas = _numba_ewma_loop(ret_window, lam, var_start)
    last_var = sigmas[-1]**2
    next_var = lam * last_var + (1 - lam) * ret_window[-1]**2
    return sigmas, math.sqrt(next_var)

def garch_fit_opt(ret: np.ndarray, min_omega: float, cfg: Dict) -> Tuple[float, float, float]:
    if len(ret) < 50: 
        return max(min_omega, 0.01 * np.nanvar(ret)), 0.05, 0.90
    start_var = np.nanvar(ret, ddof=1)
    x0 = [max(min_omega, 0.05*start_var), 0.1, 0.85]
    cons = ({'type': 'ineq', 'fun': lambda x:  0.999 - x[1] - x[2]})
    bounds = cfg["garch_bounds"]
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


# EVT
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

def build_evt_params(window: np.ndarray, garch_params: Optional[Dict], cfg: Dict) -> Dict:
    if garch_params and "omega" in garch_params:
        sigmas_in = garch_in_sample_vol(window, garch_params["omega"], garch_params["a"], garch_params["b"])
    else:
        sigmas_in, _ = ewma_sigma_path(window, cfg["ewma_lambda"])
        
    z = window / np.where(sigmas_in > 0, sigmas_in, np.nan); z = z[np.isfinite(z)]
    if len(z) < 200: return {"ok": False}
    Lz = -z; u = float(np.quantile(Lz, cfg["evt_u_quantile"])); excess = Lz[Lz > u] - u
    if len(excess) < cfg["evt_min_excess"]: return {"ok": False}
    xi, beta = gpd_fit_pwm(excess)
    if not np.isfinite(xi) or not np.isfinite(beta) or beta <= 0: return {"ok": False}
    p_exceed = float(len(excess) / len(Lz))
    return {"ok": True, "u": u, "xi": xi, "beta": beta, "p_exceed": p_exceed}


#Distributions
def normal_var_es(sigma: float, alpha: float) -> Tuple[float, float]:
    z_alpha = norm.ppf(alpha); var = z_alpha * sigma; es = sigma * norm.pdf(z_alpha) / (1 - alpha)
    return float(var), float(es)

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



# Simulations
def simulate_student_t_unit_var(df: float, size: tuple, rng: np.random.Generator) -> np.ndarray:
    Z = rng.normal(0.0, 1.0, size=size)
    G = rng.gamma(shape=df/2.0, scale=2.0, size=size)
    T = Z / np.sqrt(G / df) 
    if df > 2: T = T / np.sqrt(df / (df - 2.0))
    return T

def fhs_with_garch(window: np.ndarray, alpha: float, refit: Dict, horizon: int, rng: np.random.Generator, use_evt: bool, evt_params: Optional[Dict], cfg: Dict) -> Tuple[float, float]:
    if "omega" not in refit:
        omega, a, b = garch_fit_opt(window, cfg["garch_min_omega"], cfg)
        refit.update({"omega": omega, "a": a, "b": b})
    else: omega, a, b = refit["omega"], refit["a"], refit["b"]
    
    sigmas_in_sample = garch_in_sample_vol(window, omega, a, b)
    last_sig2 = sigmas_in_sample[-1]**2
    z = window / np.where(sigmas_in_sample > 0, sigmas_in_sample, np.nan)
    z = z[np.isfinite(z)]; Lz = -z 
    if len(z) < 80: return np.nan, np.nan

    sims = cfg["mc_sims_10d"] if horizon > 1 else cfg["mc_sims_1d"]
    
    if use_evt and evt_params and evt_params.get("ok", False):
        boot_idx = rng.integers(0, len(Lz), size=(sims, horizon))
        z_innovations = Lz[boot_idx]
        evt_mask = rng.random(size=(sims, horizon)) < cfg["evt_tail_mix"]
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
            omega, a, b = garch_fit_opt(window, cfg["garch_min_omega"], cfg)
            refit_cache = {"omega": omega, "a": a, "b": b} 
            if cfg["use_evt"]: 
                evt_params = build_evt_params(window, refit_cache, cfg)   
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
            var_forecast, es_forecast = fhs_with_garch(window, alpha, refit_cache, horizon, rng, cfg["use_evt"], evt_params, cfg)

        elif method == "evt_only":
            if horizon == 1 and evt_params and evt_params["ok"]:
                _, sigma_next = ewma_sigma_path(window, cfg["ewma_lambda"])
                q, es = gpd_var_es(evt_params["u"], evt_params["xi"], evt_params["beta"], evt_params["p_exceed"], alpha)
                var_forecast, es_forecast = q * sigma_next, es * sigma_next
            else: 
                var_forecast, es_forecast = fhs_with_garch(window, alpha, refit_cache, horizon, rng, True, evt_params, cfg)
        
        VaR[idx], ES[idx] = var_forecast, es_forecast
        if np.isfinite(realized[idx]) and np.isfinite(VaR[idx]): 
            exceptions[idx] = 1.0 if realized[idx] > VaR[idx] else 0.0
        
        if cfg["n_jobs"] == 1 and (t - window_size) % progress_every == 0: 
            print(f"{progress_label} {method} h={horizon}: step {t - window_size + 1}/{total_steps}")
            
    return ForecastResult(method, horizon, pd.Series(VaR, index=dates), pd.Series(ES, index=dates), pd.Series(realized, index=dates), pd.Series(exceptions, index=dates))
