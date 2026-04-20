import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import re

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

def sanitize_name(name: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]", "_", name)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s

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

    # ── Panel 1: VaR ──────────────────────────────────────────────────
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

    # Box with stats e traffic light
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

    # ── Panel 2: ES ───────────────────────────────────────────────────
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
