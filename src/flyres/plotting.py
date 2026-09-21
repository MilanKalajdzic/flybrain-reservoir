"""Result figures (matplotlib, light theme). The numbers behind every figure are in the CSVs next to it.

Color follows the entity: each wiring keeps the same color in every figure. When the story is
"connectome vs the rest", the connectome is blue and everything else is muted gray.

Styling is applied per figure instead of through rcParams/rc_context: IPython turns on matplotlib's
interactive mode when the first figure is created, and an rc_context exit would silently undo that,
so later figures in a notebook would never show up.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .controls import WIRINGS

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
WIRING_COLORS = dict(zip(WIRINGS, SERIES))
BASELINES = ("linear_features", "ar", "buy_hold")
LABELS = {
    "connectome": "connectome", "degree_preserving": "degree-preserving", "weight_shuffle": "weight shuffle",
    "sign_shuffle": "sign shuffle", "erdos_renyi": "Erdős–Rényi", "linear_features": "linear (features)",
    "ar": "AR (lags)", "buy_hold": "buy & hold",
}
METRIC_LABELS = {"ic": "Information coefficient", "hit_rate": "Hit rate", "sharpe_net": "Sharpe (net of costs)",
                 "sharpe_gross": "Sharpe (gross)", "ann_return_net": "Annual return (net)"}
LINE = dict(solid_capstyle="round", solid_joinstyle="round")


def _figure(ncols: int = 1, figsize=(6.4, 3.6), **kw):
    fig, axes = plt.subplots(1, ncols, figsize=figsize, facecolor=SURFACE, **kw)
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8.5)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_color(INK_2)
            label.set_fontsize(9)
    return fig, axes


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=11, fontweight="bold", color=INK)


def _legend(ax, **kw) -> None:
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, **kw)


def _save(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    return fig


def plot_metric_strip(metrics: pd.DataFrame, ticker: str | None = None,
                      metric_names=("ic", "hit_rate", "sharpe_net"), path=None):
    """One row per model, one panel per metric. Small dots = seeds, big dot = mean."""
    df = metrics if ticker is None else metrics[metrics["ticker"] == ticker]
    order = [w for w in WIRINGS if w in set(df["model"])] + [b for b in BASELINES if b in set(df["model"])]
    y = {m: i for i, m in enumerate(order)}
    fig, axes = _figure(len(metric_names), figsize=(3.3 * len(metric_names), 0.42 * len(order) + 1.3), sharey=True)
    axes = np.atleast_1d(axes)
    n_wirings = len([m for m in order if m not in BASELINES])
    for ax, metric in zip(axes, metric_names):
        ax.grid(axis="y", visible=False)
        ax.axvline(0.5 if metric == "hit_rate" else 0.0, color=AXIS, linewidth=0.8, zorder=0)
        for model in order:
            vals = df.loc[df["model"] == model, metric].dropna().to_numpy()
            if len(vals) == 0:
                continue
            color = WIRING_COLORS["connectome"] if model == "connectome" else (INK_2 if model in BASELINES else MUTED)
            ax.scatter(vals, np.full(len(vals), y[model]), s=14, color=color, linewidths=0, zorder=2)
            ax.scatter([vals.mean()], [y[model]], s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        if n_wirings < len(order):
            ax.axhline(n_wirings - 0.5, color=GRID, linewidth=0.8)
        _title(ax, METRIC_LABELS.get(metric, metric))
    axes[0].set_yticks(range(len(order)), [LABELS.get(m, m) for m in order])
    axes[0].invert_yaxis()
    prefix = f"{ticker}: " if ticker else ""
    fig.suptitle(f"{prefix}out-of-sample results, small dots = seeds, large dot = mean", x=0.01, ha="left",
                 fontsize=9, color=INK_2, y=1.0)
    fig.tight_layout()
    return _save(fig, path)


def plot_memory_curves(memory: pd.DataFrame, path=None):
    """Memory capacity per delay, mean over seeds, one line per wiring."""
    mean = memory.groupby(["wiring", "delay"])["mc"].mean().unstack("wiring")
    totals = memory.groupby(["wiring", "seed"])["mc"].sum().groupby("wiring").mean()
    fig, ax = _figure()
    for w in [w for w in WIRINGS if w in mean.columns]:
        main = w == "connectome"
        ax.plot(mean.index, mean[w], color=WIRING_COLORS[w], label=f"{LABELS[w]}  (total {totals[w]:.1f})",
                linewidth=2.4 if main else 1.6, zorder=3 if main else 2, **LINE)
    ax.set_xlim(1, mean.index.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("delay k (steps back)")
    ax.set_ylabel("MC_k = R² of reconstructing u(t−k)")
    _title(ax, "Memory capacity by wiring")
    _legend(ax, loc="upper right")
    fig.tight_layout()
    return _save(fig, path)


def plot_equity(returns: dict, dates, path=None, title: str = "Seed-ensemble strategies, growth of 1 (net of costs)"):
    """Cumulative net returns. Connectome blue, buy & hold orange, controls as a gray pack."""
    fig, ax = _figure(figsize=(7.2, 3.8))
    drawn_control = False
    for name, r in returns.items():
        r = np.asarray(r, dtype=np.float64)
        ok = np.isfinite(r)
        if not ok.any():
            continue
        wealth = np.cumprod(1 + np.where(ok, r, 0.0))
        wealth[: np.argmax(ok)] = np.nan
        if name == "connectome":
            kw = dict(color=WIRING_COLORS["connectome"], linewidth=2.2, zorder=4, label=LABELS[name])
        elif name == "buy_hold":
            kw = dict(color=SERIES[1], linewidth=1.8, zorder=3, label=LABELS[name])
        elif name in WIRINGS:
            kw = dict(color=AXIS, linewidth=1.2, zorder=2, label=None if drawn_control else "controls")
            drawn_control = True
        else:
            continue
        ax.plot(dates, wealth, **kw, **LINE)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    _title(ax, title)
    _legend(ax, loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def plot_spectra(eigenvalues: dict, radius: float | None = None, path=None):
    """Eigenvalues in the complex plane, one small panel per wiring (same scale everywhere)."""
    names = list(eigenvalues)
    lim = 1.05 * max(np.abs(v).max() for v in eigenvalues.values())
    fig, axes = _figure(len(names), figsize=(3.0 * len(names), 3.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        ev = eigenvalues[name]
        color = WIRING_COLORS.get(name, SERIES[0])
        ax.scatter(ev.real, ev.imag, s=4, color=color, linewidths=0)
        lead = ev[np.argmax(np.abs(ev))]  # the eigenvalue the spectral-radius rescaling pins to the circle
        ax.scatter([lead.real], [lead.imag], s=48, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        if radius:
            t = np.linspace(0, 2 * np.pi, 400)
            ax.plot(radius * np.cos(t), radius * np.sin(t), color=AXIS, linewidth=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        _title(ax, LABELS.get(name, name))
        ax.set_xlabel("Re λ")
    axes[0].set_ylabel("Im λ")
    fig.tight_layout()
    return _save(fig, path)


def plot_degree_ccdf(neurons: pd.DataFrame, path=None):
    """Complementary CDF of in- and out-degree on log-log axes: heavy tails show up as long straight-ish tails."""
    fig, ax = _figure(figsize=(5.6, 3.6))
    for col, color, label in (("in_degree", SERIES[0], "in-degree"), ("out_degree", SERIES[1], "out-degree")):
        d = np.sort(neurons[col].to_numpy())
        d = d[d > 0]
        ax.plot(d, 1.0 - np.arange(len(d)) / len(d), color=color, label=label, linewidth=2.0, **LINE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degree (partners with ≥ min_weight synapses)")
    ax.set_ylabel("fraction of neurons ≥ degree")
    _title(ax, "Degree distribution")
    _legend(ax, loc="lower left")
    fig.tight_layout()
    return _save(fig, path)
