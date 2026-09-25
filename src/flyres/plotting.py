"""Result figures (matplotlib, light theme). The numbers behind every figure are in the CSVs next to it.

Color follows the entity: each wiring keeps the same color in every figure. When the story is
"connectome vs the rest", the connectome is blue and everything else is muted gray.

Styling is applied per figure instead of through rcParams/rc_context: IPython turns on matplotlib's
interactive mode when the first figure is created, and an rc_context exit would silently undo that,
so later figures in a notebook would never show up.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
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


SEQ_BLUE = ["#eef4fb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]  # light theme: darker = more
# dark theme counterparts (brighter = more), for the animation
D_SURFACE, D_INK, D_INK_2, D_GRID, D_AXIS = "#1a1a19", "#ffffff", "#c3c2b7", "#2c2c2a", "#383835"
GLOW = ["#22354d", "#256abf", "#3987e5", "#86b6ef", "#e6f0fd"]


def plot_activity_heatmap(dev: pd.DataFrame, close: pd.Series, episodes: dict | None = None, vmin: float = 0.5,
                          vmax: float = 1.6, ticker: str = "SPY", path=None):
    """Distance-from-normal per pathway stage over time (rows = groups), under the SPY price for context."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    weeks = dev.index
    step = weeks[-1] - weeks[-2] if len(weeks) > 1 else pd.Timedelta(days=7)
    edges = weeks.append(pd.DatetimeIndex([weeks[-1] + step]))
    price = close.loc[edges[0]:edges[-1]]

    fig = plt.figure(figsize=(11, 0.34 * dev.shape[1] + 2.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.34 * dev.shape[1] / 1.2], width_ratios=[1, 0.018],
                          hspace=0.08, wspace=0.02)
    ax_p = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[1, 0], sharex=ax_p)
    ax_c = fig.add_subplot(gs[1, 1])
    for ax in (ax_p, ax_h):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8.5)

    ax_p.plot(price.index, price.values, color=INK_2, linewidth=1.4, **LINE)
    ax_p.set_yscale("log")
    ax_p.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax_p.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_p.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_p.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax_p.set_ylabel(ticker, color=INK_2, fontsize=9)
    ax_p.tick_params(labelbottom=False)
    for name, (a, b) in (episodes or {}).items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if b < weeks[0] or a > weeks[-1]:
            continue
        ax_p.axvspan(a, b, color=GRID, alpha=0.6, zorder=0, linewidth=0)
        ax_p.text(a, 1.02, name, transform=ax_p.get_xaxis_transform(), fontsize=8.5, color=INK_2, va="bottom")

    mesh = ax_h.pcolormesh(edges, np.arange(dev.shape[1] + 1), dev.to_numpy().T, cmap=cmap,
                           norm=Normalize(vmin, vmax), shading="flat", rasterized=True)
    ax_h.set_xlim(edges[0], edges[-1])
    ax_h.hlines(np.arange(1, dev.shape[1]), edges[0], edges[-1], color=SURFACE, linewidth=2)  # row gaps
    ax_h.set_yticks(np.arange(dev.shape[1]) + 0.5, dev.columns)
    ax_h.invert_yaxis()
    ax_h.tick_params(axis="y", length=0)
    cb = fig.colorbar(mesh, cax=ax_c)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
    cb.set_label("distance from normal (mean |z|)", color=INK_2, fontsize=8.5)
    fig.text(0.125, 0.985, "How stirred up each part of the fly circuit is", fontsize=11, fontweight="bold",
             color=INK, va="top")
    fig.text(0.125, 0.955, "Average distance of each group's neurons from their own normal activity. "
             "Darker = more stirred up. Shaded = big drawdowns.", fontsize=8.5, color=INK_2, va="top")
    return _save(fig, path)


def animate_brain(glow: pd.DataFrame, xy: np.ndarray, input_idx, close: pd.Series, background_xy: np.ndarray,
                  window: tuple, title: str, path, ticker: str = "SPY", fps: int = 6, vmin: float = 0.7,
                  vmax: float = 1.9, colors: int = 96):
    """Animated frontal view of the circuit, one frame per week: brighter = further from normal activity.

    glow: weekly mean |z|, rows = weeks, columns = neurons. xy: (n, 2) soma positions (NaN = unknown).
    Input neurons have their cell bodies outside the brain, so they get their own grid on the side.
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from PIL import Image

    cmap = LinearSegmentedColormap.from_list("glow", GLOW)
    norm = Normalize(vmin, vmax)  # a normal week (|z| around 0.8) stays dim, crash weeks glow
    frames = glow.loc[window[0]:window[1]]
    price = close.loc[pd.Timestamp(window[0]) - pd.Timedelta(days=7):window[1]]
    peak = close.loc[:window[1]].cummax()

    inputs = np.asarray(input_idx)
    placed = np.flatnonzero(np.isfinite(xy[:, 0]) & ~np.isin(np.arange(len(xy)), inputs))

    fig = plt.figure(figsize=(9.6, 6.0), dpi=100, facecolor=D_SURFACE)
    ax_b = fig.add_axes([0.01, 0.02, 0.62, 0.84])
    ax_i = fig.add_axes([0.68, 0.45, 0.28, 0.30])
    ax_p = fig.add_axes([0.68, 0.10, 0.28, 0.24])
    for ax in (ax_b, ax_i, ax_p):
        ax.set_facecolor(D_SURFACE)

    # brain outline: density of all ~140k somata, drawn once
    H, xe, ye = np.histogram2d(background_xy[:, 0], background_xy[:, 1], bins=(260, 190))
    outline = LinearSegmentedColormap.from_list("outline", [D_SURFACE, "#4a4a45"])
    ax_b.imshow(np.log1p(H.T), extent=(xe[0], xe[-1], ye[-1], ye[0]), cmap=outline, interpolation="bilinear",
                aspect="equal")
    first = frames.iloc[0].to_numpy()
    sc = ax_b.scatter(xy[placed, 0], xy[placed, 1], s=7, c=first[placed], cmap=cmap, norm=norm, linewidths=0)
    ax_b.set_xlim(xe[0], xe[-1])
    ax_b.set_ylim(ye[-1], ye[0])
    ax_b.axis("off")

    k = len(inputs)
    side = int(np.ceil(np.sqrt(k)))
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    gx, gy = gx.ravel()[:k], gy.ravel()[:k]
    sci = ax_i.scatter(gx, gy, s=34, c=first[inputs], cmap=cmap, norm=norm, linewidths=0)
    ax_i.set_xlim(-1, side)
    ax_i.set_ylim(side, -1)
    ax_i.axis("off")
    ax_i.set_title("input neurons (sense organs, outside the brain)", loc="left", fontsize=8.5, color=D_INK_2)

    ax_p.plot(price.index, price.values, color=D_INK_2, linewidth=1.3, **LINE)
    marker, = ax_p.plot([], [], "o", color=D_INK, markersize=6, markeredgecolor=D_SURFACE, markeredgewidth=2)
    vline = ax_p.axvline(price.index[0], color=D_AXIS, linewidth=0.8)
    for s in ("top", "right"):
        ax_p.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax_p.spines[s].set_color(D_AXIS)
    ax_p.tick_params(colors="#898781", labelcolor=D_INK_2, labelsize=7.5)
    ax_p.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax_p.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax_p.set_title(ticker, loc="left", fontsize=8.5, color=D_INK_2)

    fig.text(0.02, 0.955, title, fontsize=13, fontweight="bold", color=D_INK)
    fig.text(0.02, 0.915, f"{glow.shape[1]:,} neurons of the male CNS connectome driven by {ticker} returns and "
             "volatility. Brighter = further from the neuron's normal activity.", fontsize=8.5, color=D_INK_2)
    date_txt = fig.text(0.68, 0.84, "", fontsize=15, fontweight="bold", color=D_INK)
    dd_txt = fig.text(0.68, 0.80, "", fontsize=9.5, color=D_INK_2)

    cax = fig.add_axes([0.03, 0.085, 0.2, 0.016])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors="#898781", labelcolor=D_INK_2, labelsize=7)
    cb.set_label("distance from normal (weekly mean |z|)", color=D_INK_2, fontsize=7.5)

    def update(i):
        week = frames.index[i]
        vals = frames.iloc[i].to_numpy()
        sc.set_array(vals[placed])
        sci.set_array(vals[inputs])
        p = close.loc[:week].iloc[-1]
        marker.set_data([week], [p])
        vline.set_xdata([week, week])
        date_txt.set_text(f"{week:%d %b %Y}")
        dd_txt.set_text(f"{ticker} {p / peak.loc[:week].iloc[-1] - 1:+.0%} from its peak")
        return sc, sci, marker, vline, date_txt, dd_txt

    images = []
    for i in range(len(frames)):  # draw each frame and grab the pixels (works with any Agg-based backend)
        update(i)
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    plt.close(fig)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # one shared palette for all frames keeps the file small (~1.5 MB for a year of weeks)
    palette = images[len(images) // 2].quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    frames_p = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    frames_p[0].save(path, save_all=True, append_images=frames_p[1:], duration=int(1000 / fps), loop=0,
                     optimize=True)
    return path


def plot_gain_sweep(summary: pd.DataFrame, gain_label: str = "spectral radius", noise: float | None = None,
                    path=None):
    """Memory capacity against gain, one line per wiring (mean over seeds). With readout-noise results,
    two panels on the same y scale: with noise (left) and noise-free (right).
    Filled markers = valid reservoir, hollow = some readouts latch or go chaotic."""
    panels = [("memory", "Noise-free (standard benchmark)")]
    if "memory_noisy" in summary.columns:
        label = f"With readout noise ({noise:.1%} of range)" if noise else "With readout noise"
        panels = [("memory_noisy", label)] + panels
    fig, axes = _figure(len(panels), figsize=(5.2 * len(panels) + 1.6, 4.1), sharey=True)
    axes = np.atleast_1d(axes)
    lo, hi = summary["gain"].min(), summary["gain"].max()
    labelled = [g for g in (0.25, 0.5, 1, 2, 3, 6, 12, 20, 30, 50, 100) if lo <= g <= hi]
    for ax, (col, title) in zip(axes, panels):
        for w in [w for w in WIRINGS if w in set(summary["wiring"])]:
            g = summary[summary["wiring"] == w].sort_values("gain")
            color = WIRING_COLORS[w]
            main = w == "connectome"
            ax.plot(g["gain"], g[col], color=color, linewidth=2.4 if main else 1.6, zorder=3 if main else 2,
                    label=LABELS[w], **LINE)
            ok = g["valid"].to_numpy()
            ax.scatter(g["gain"][ok], g[col][ok], s=46, color=color, edgecolors=SURFACE, linewidths=2, zorder=4)
            ax.scatter(g["gain"][~ok], g[col][~ok], s=40, facecolors=SURFACE, edgecolors=color, linewidths=1.6,
                       zorder=4)
        ax.set_xscale("log")
        ax.set_xticks(labelled)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel(gain_label)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK_2)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("memory capacity")
    fig.suptitle("Memory capacity at each gain", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.04)
    fig.text(0.01, 0.975, "Mean over seeds. Hollow = invalid reservoir (some neurons latch or go chaotic).",
             ha="left", fontsize=8.5, color=INK_2)
    _legend(axes[-1], loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return _save(fig, path)


# diverging: turned down (blue) <- unchanged (gray) -> turned up (red), equal steps per arm
DIVERGING = ["#184f95", "#5598e7", "#b7d3f6", "#f0efec", "#f5b9b8", "#e66767", "#a82e2d"]


def plot_region_gains(results: pd.DataFrame, region_table: pd.DataFrame, noise: float | None = None, path=None,
                      after: str = "regions", after_label: str = "per-region gains", title: str = "Per-region gains",
                      heat_title: str = "Factor on each region's incoming synapses (neurons)",
                      subtitle: str = "Hollow = each wiring's best single gain. Filled = after the search turns each "
                                      "region up (red) or down (blue). Mean over seeds, small dots = seeds."):
    """Left: memory on a fresh input at each wiring's best single gain (hollow) and after (filled; results
    columns `<metric>_<after>`), small dots = seeds. Right: the factor each region got, per wiring (geometric
    mean over seeds, log color scale, red = turned up). Also draws the homeostatic-gains figure."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    noisy = "memory_noisy_single" in results.columns
    col = "memory_noisy" if noisy else "memory"
    wirings = [w for w in WIRINGS if w in set(results["wiring"])]
    regions = list(dict.fromkeys(region_table["region"]))
    sizes = region_table.drop_duplicates("region").set_index("region")["neurons"]

    fig = plt.figure(figsize=(12.5, 0.5 * len(wirings) + 2.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.55, 0.03], wspace=0.08)
    ax, ax_h, ax_c = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for a in (ax, ax_h):
        a.set_facecolor(SURFACE)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(AXIS)
            a.spines[side].set_linewidth(0.8)
        a.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8.5)

    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    xmax = 1.15 * float(np.nanmax(results[[f"{col}_single", f"{col}_{after}"]].to_numpy()))
    ax.set_xlim(0, xmax)
    any_invalid = False
    for i, w in enumerate(wirings):
        r = results[results["wiring"] == w]
        a, b = r[f"{col}_single"], r[f"{col}_{after}"]
        ok = r[f"valid_{after}"].astype(bool).to_numpy() if f"valid_{after}" in r else np.ones(len(r), bool)
        color = WIRING_COLORS[w]
        ax.plot([a.mean(), b.mean()], [i, i], color=color, linewidth=2.0, zorder=2,
                linestyle="-" if ok.any() else ":", **LINE)
        ax.scatter(a, np.full(len(a), i + 0.2), s=10, color=color, alpha=0.45, linewidths=0, zorder=2)
        ax.scatter(b[ok], np.full(ok.sum(), i + 0.2), s=10, color=color, alpha=0.45, linewidths=0, zorder=2)
        ax.scatter(b[~ok], np.full((~ok).sum(), i + 0.2), s=14, color=color, alpha=0.6, marker="x", linewidths=1,
                   zorder=2)
        ax.scatter([a.mean()], [i], s=70, facecolors=SURFACE, edgecolors=color, linewidths=2.0, zorder=3)
        if ok.all():
            ax.scatter([b.mean()], [i], s=80, color=color, edgecolors=SURFACE, linewidths=2.0, zorder=4)
            note = ""
        else:  # an invalid reservoir (latches or goes chaotic) doesn't count, so don't draw it like one that does
            any_invalid = True
            ax.scatter([b.mean()], [i], s=70, color=color, marker="X", edgecolors=SURFACE, linewidths=1.0, zorder=4)
            note = "  (latches)" if not ok.any() else f"  ({ok.sum()}/{len(ok)} valid)"
        ax.text(max(a.mean(), b.mean()) + 0.03 * xmax, i, f"{b.mean():.1f}{note}", ha="left", va="center",
                fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(wirings)), [LABELS[w] for w in wirings])
    ax.set_ylim(len(wirings) - 0.5, -0.5)  # same row centres as the heatmap
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("memory capacity" + (f", readout noise {noise:.1%} of range" if noisy and noise else ""),
                  color=INK_2, fontsize=9)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=SURFACE,
                          markeredgecolor=INK_2, markeredgewidth=1.6, label="best single gain"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=INK_2, label=after_label)]
    if any_invalid:
        handles.append(plt.Line2D([], [], marker="X", linestyle="", markersize=7, color=INK_2,
                                  label="invalid on fresh input"))
    ax.legend(handles=handles, frameon=False, fontsize=8 if len(handles) > 2 else 8.5, labelcolor=INK_2,
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(handles), borderaxespad=0.2,
              handletextpad=0.2, columnspacing=0.8)

    t = region_table.assign(log2=np.log2(region_table["factor"]))
    lf = t.pivot_table(index="wiring", columns="region", values="log2", aggfunc="mean").reindex(
        index=wirings, columns=regions)
    lim = max(1.0, float(np.nanmax(np.abs(lf.to_numpy()))))
    cmap = LinearSegmentedColormap.from_list("diverging", DIVERGING)
    mesh = ax_h.pcolormesh(np.arange(len(regions) + 1), np.arange(len(wirings) + 1), lf.to_numpy(), cmap=cmap,
                           norm=TwoSlopeNorm(0.0, -lim, lim), shading="flat")
    ax_h.hlines(np.arange(1, len(wirings)), 0, len(regions), color=SURFACE, linewidth=2)
    ax_h.vlines(np.arange(1, len(regions)), 0, len(wirings), color=SURFACE, linewidth=2)
    for i in range(len(wirings)):
        for j in range(len(regions)):
            v = lf.iat[i, j]
            if np.isfinite(v):
                f = 2.0 ** v
                ax_h.text(j + 0.5, i + 0.5, f"×{f:.2g}" if abs(v) > 0.05 else "–", ha="center", va="center",
                          fontsize=7.5, color=SURFACE if abs(v) > 0.72 * lim else INK_2)

    def short(n):
        return f"{n / 1000:.0f}k" if n >= 10_000 else (f"{n / 1000:.1f}k" if n >= 1000 else f"{n}")

    ax_h.set_xticks(np.arange(len(regions)) + 0.5, [f"{r} ({short(int(sizes[r]))})" for r in regions],
                    rotation=35, ha="right", rotation_mode="anchor")
    ax_h.set_yticks(np.arange(len(wirings)) + 0.5, [])
    ax_h.set_ylim(len(wirings), 0)
    ax_h.tick_params(length=0)
    for side in ("left", "bottom"):
        ax_h.spines[side].set_visible(False)
    cb = fig.colorbar(mesh, cax=ax_c)
    cb.outline.set_visible(False)
    ticks = [v for v in range(-6, 7, 2 if lim > 2 else 1) if -lim <= v <= lim]
    cb.set_ticks(ticks, labels=[f"×{2 ** v}" if v >= 0 else f"×1/{2 ** -v}" for v in ticks])
    cb.ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)

    ax.set_title("Memory on a fresh input", loc="left", fontsize=9.5, color=INK_2, pad=18)
    ax_h.set_title(heat_title, loc="left", fontsize=9.5, color=INK_2, pad=18)
    fig.suptitle(title, x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.06)
    fig.text(0.125, 0.995, subtitle, ha="left", fontsize=8.5, color=INK_2)
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


VOL_BASELINE_LABELS = {"ewma": "EWMA", "har_levels": "HAR (levels)", "har": "HAR",
                       "har_inputs": "HAR + inputs (linear)"}


def plot_vol_models(metrics: pd.DataFrame, ticker: str, path=None):
    """Log MSE relative to HAR (%), one panel per horizon. Small dots = seeds, big dot = mean.
    Left of the line = better than HAR."""
    df = metrics[metrics["ticker"] == ticker]
    horizons = sorted(df["horizon"].unique())
    models = [w for w in WIRINGS if w in set(df["model"])]
    baselines = [b for b in ("har_inputs", "har", "ewma") if b in set(df["model"])]  # har_levels: fitted for QLIKE
    order = models + baselines
    y = {m: i for i, m in enumerate(order)}
    fig, axes = _figure(len(horizons), figsize=(3.9 * len(horizons) + 1.2, 0.4 * len(order) + 1.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, horizons):
        g = df[df["horizon"] == h]
        har = g.loc[g["model"] == "har", "mse_log"].mean()
        ax.grid(axis="y", visible=False)
        ax.axvline(0.0, color=AXIS, linewidth=0.8, zorder=0)
        for model in order:
            vals = 100.0 * (g.loc[g["model"] == model, "mse_log"].to_numpy() / har - 1.0)
            if len(vals) == 0:
                continue
            color = WIRING_COLORS.get(model, INK_2)
            ax.scatter(vals, np.full(len(vals), y[model]), s=14, color=color, alpha=0.5, linewidths=0, zorder=2)
            ax.scatter([vals.mean()], [y[model]], s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.axhline(len(models) - 0.5, color=GRID, linewidth=0.8)
        ax.set_xlabel("log MSE vs HAR (%)")
        ax.set_title(f"next {h} days", loc="left", fontsize=9.5, color=INK_2)
    axes[0].set_yticks(range(len(order)), [LABELS.get(m, VOL_BASELINE_LABELS.get(m, m)) for m in order])
    axes[0].invert_yaxis()
    fig.suptitle(f"{ticker}: forecasting realized volatility", x=0.01, ha="left", fontsize=11, fontweight="bold",
                 color=INK, y=1.07)
    fig.text(0.01, 1.0, "Out-of-sample error on log realized variance, relative to HAR (left = better). "
             "Small dots = seeds.", ha="left", fontsize=8.5, color=INK_2)
    fig.tight_layout()
    return _save(fig, path)


def plot_vol_memory(link: pd.DataFrame, ticker: str, path=None):
    """Each reservoir's memory capacity against what its states add to HAR + inputs (log MSE, %)."""
    df = link[link["ticker"] == ticker]
    horizons = sorted(df["horizon"].unique())
    fig, axes = _figure(len(horizons), figsize=(3.9 * len(horizons) + 1.6, 3.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, horizons):
        g = df[df["horizon"] == h]
        ax.axhline(0.0, color=AXIS, linewidth=0.8, zorder=0)
        for w in [w for w in WIRINGS if w in set(g["model"])]:
            s = g[g["model"] == w]
            ax.scatter(s["memory"], s["mse_vs_har_inputs_pct"], s=40, color=WIRING_COLORS[w], edgecolors=SURFACE,
                       linewidths=1.5, zorder=3, label=LABELS[w])
        ax.set_xlabel(f"memory capacity ({g['memory_measure'].iloc[0]})" if "memory_measure" in g else
                      "memory capacity")
        ax.set_title(f"next {h} days", loc="left", fontsize=9.5, color=INK_2)
    axes[0].set_ylabel("log MSE vs HAR + inputs (%)")
    _legend(axes[-1], loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f"{ticker}: does memory help forecast volatility?", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", color=INK, y=1.1)
    fig.text(0.01, 1.01, "One dot per reservoir (wiring x seed). Below the line = the states improve on the "
             "linear model with the same inputs.", ha="left", fontsize=8.5, color=INK_2)
    fig.tight_layout()
    return _save(fig, path)


def plot_vol_forecast(predictions: pd.DataFrame, ticker: str, horizon: int, windows: dict,
                      models=("har", "connectome", "degree_preserving"), path=None):
    """Realized volatility over the next `horizon` days against the forecasts made the day before it starts,
    annualized, for a few crisis windows. Reservoir lines are seed ensembles (mean variance forecast)."""
    df = predictions[(predictions["ticker"] == ticker) & (predictions["horizon"] == horizon)]
    fig, axes = _figure(len(windows), figsize=(5.0 * len(windows) + 1.6, 3.4), sharey=False)
    axes = np.atleast_1d(axes)
    colors = {"har": INK_2, **WIRING_COLORS}
    for ax, (name, (a, b)) in zip(axes, windows.items()):
        w = df[(df["date"] >= pd.Timestamp(a)) & (df["date"] <= pd.Timestamp(b))]
        if w.empty:
            ax.set_visible(False)
            continue
        real = w.groupby("date")["realized_var"].first()
        ax.plot(real.index, 100 * np.sqrt(252 * real), color=MUTED, linewidth=1.2, label="realized", **LINE)
        for m in models:
            s = w[w["model"] == m].groupby("date")["var_pred"].mean()
            if len(s):
                ax.plot(s.index, 100 * np.sqrt(252 * s), color=colors.get(m, INK_2),
                        linewidth=2.2 if m == "connectome" else 1.6, linestyle="--" if m == "har" else "-",
                        label=LABELS.get(m, VOL_BASELINE_LABELS.get(m, m)), **LINE)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_title(name, loc="left", fontsize=9.5, color=INK_2)
    axes[0].set_ylabel(f"volatility, next {horizon} days (annualized %)")
    _legend(axes[-1], loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f"{ticker}: realized volatility and the forecasts made for it", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", color=INK, y=1.1)
    fig.text(0.01, 1.01, "Each forecast is plotted at the day it was made. Reservoirs: mean over seeds.",
             ha="left", fontsize=8.5, color=INK_2)
    fig.tight_layout()
    return _save(fig, path)
