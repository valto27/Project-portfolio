import math
from typing import Dict, Optional
import numpy as np
from scipy.stats import chi2, binom

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
