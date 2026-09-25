"""Run the whole comparison: every wiring x every seed, plus baselines, then summarize.

For a given seed, every wiring gets the *same* input weights, bias and readout neurons, so the
only thing that differs between wirings is the recurrent matrix. That makes the per-seed
differences a paired comparison.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from .benchmarks import memory_capacities, readout_stability
from .config import ExperimentConfig, save_config
from .connectome import SOURCES, load_connectome
from .controls import WIRINGS, make_wiring
from .market import Dataset, download_prices, load_prices_csv, make_dataset
from .metrics import block_bootstrap_sharpe_diff, evaluate, strategy_returns
from .readout import walk_forward
from .diagnostics import top_mode
from .reservoir import Reservoir, normalize_inputs, scale_weights, signed_weights
from .subgraph import Subgraph, graph_stats, select_subgraph
from .synthetic import synthetic_connectome, synthetic_prices

BASELINES = ("linear_features", "ar", "buy_hold")
KEY_METRICS = ("ic", "hit_rate", "sharpe_net")


@dataclass
class ExperimentResult:
    metrics: pd.DataFrame       # one row per (ticker, model, seed)
    memory: pd.DataFrame        # one row per (wiring, seed, delay)
    graph: pd.DataFrame         # one row per (wiring, seed)
    comparisons: pd.DataFrame   # connectome vs every control and baseline
    predictions: pd.DataFrame   # long format: date, ticker, model, seed, pred
    ensemble_returns: dict      # ticker -> DataFrame of daily net returns of seed-ensemble strategies
    summary: str
    out_dir: Path | None = None


# --------------------------------------------------------------------------- inputs

def subgraph_cache_path(cfg: ExperimentConfig) -> Path:
    sc, cc = cfg.subgraph, cfg.connectome
    spec = {"method": sc.method, "n": sc.n_neurons, "inputs": sc.n_inputs, "filter": sc.input_filter,
            "direction": sc.direction, "min_weight": cc.min_weight, "autapses": cc.drop_autapses,
            "inhibitory": sorted(cc.inhibitory)}
    if sc.method == "grow":  # growth now breaks ties the same way on every machine; don't reuse older circuits
        spec["growth"] = "stable"
    key = json.dumps(spec, sort_keys=True)
    digest = hashlib.md5(key.encode()).hexdigest()[:8]
    prefix = "" if cc.source == "malecns" else f"{cc.source}_"  # male CNS names stay as they were
    name = f"{prefix}{sc.method}_n{sc.n_neurons}_w{cc.min_weight}_{digest}"
    return Path(cfg.data.cache_dir) / "subgraphs" / name


def prepare_subgraph(cfg: ExperimentConfig, verbose: bool = True) -> Subgraph:
    sc, cc = cfg.subgraph, cfg.connectome
    args = dict(n_neurons=sc.n_neurons, n_inputs=sc.n_inputs, input_filter=sc.input_filter, method=sc.method,
                direction=sc.direction, inhibitory=cc.inhibitory, verbose=verbose)
    if cc.source == "synthetic":
        return select_subgraph(synthetic_connectome(n=cc.synthetic_n, seed=0), **args)
    if cc.source not in SOURCES:
        raise ValueError(f"connectome.source must be one of {SOURCES} or 'synthetic'")
    path = subgraph_cache_path(cfg)
    if Subgraph.exists(path):
        if verbose:
            print(f"subgraph: using cached {path.name}")
        return Subgraph.load(path)
    conn = load_connectome(cfg.data.raw_dir, cfg.data.cache_dir, cc.min_weight, cc.drop_autapses, verbose,
                           source=cc.source)
    sub = select_subgraph(conn, **args)
    sub.save(path)
    return sub


def load_closes(cfg: ExperimentConfig) -> dict:
    """Close prices per ticker, from the configured source (yfinance results are cached)."""
    mc = cfg.market
    if mc.source == "synthetic":
        closes = {"SYNTH": synthetic_prices(mc.synthetic_days, mc.synthetic_predictability, seed=0)}
    elif mc.source == "csv":
        df = load_prices_csv(mc.csv_path).loc[mc.start:mc.end]
        closes = {t: df[t] for t in (mc.tickers or list(df.columns)) if t in df.columns}
    elif mc.source == "yfinance":
        df = download_prices(mc.tickers, mc.start, mc.end, cfg.data.cache_dir)
        closes = {t: df[t] for t in df.columns}
    else:
        raise ValueError("market.source must be 'yfinance', 'csv' or 'synthetic'")
    if not closes:
        raise ValueError(f"no price data for tickers {mc.tickers}")
    return {t: c.dropna() for t, c in closes.items()}


def prepare_datasets(cfg: ExperimentConfig) -> dict[str, Dataset]:
    mc = cfg.market
    return {t: make_dataset(c, mc.features, mc.horizon, mc.z_window, mc.ar_lags, ticker=t)
            for t, c in load_closes(cfg).items()}


# --------------------------------------------------------------------------- jobs

def build_matrix(sub: Subgraph, cfg: ExperimentConfig, wiring: str, seed: int):
    """Recurrent matrix for one wiring and seed: (scaled signed W, synapse counts, signs, raw gain).
    Seeded exactly like the experiment, so anything built from it reproduces the experiment's reservoir."""
    rc = cfg.reservoir
    W_counts, sign = make_wiring(wiring, sub.W, sub.sign, np.random.default_rng([seed, 1, WIRINGS.index(wiring)]),
                                 cfg.swaps_per_edge)
    W = normalize_inputs(signed_weights(W_counts, sign, rc.weight_transform), rc.input_normalization)
    W, raw_gain = scale_weights(W, rc.spectral_radius, rc.normalize)
    return W, W_counts, sign, raw_gain


def memory_noise_levels(cfg: ExperimentConfig) -> tuple:
    """Noise-free memory always; memory with readout noise too unless memory.readout_noise is 0."""
    return (0.0, cfg.memory.readout_noise) if cfg.memory.readout_noise > 0 else (0.0,)


def readout_neurons(sub: Subgraph, cfg: ExperimentConfig, seed: int) -> np.ndarray:
    """The neurons the readout listens to for a given seed (same for every wiring of that seed)."""
    pool = sub.readout_pool
    return np.sort(np.random.default_rng([seed, 2]).choice(pool, min(cfg.subgraph.n_readout, len(pool)),
                                                           replace=False))


def reservoir_kwargs(cfg: ExperimentConfig) -> dict:
    rc = cfg.reservoir
    return dict(leak_rate=rc.leak_rate, input_scaling=rc.input_scaling, bias_scaling=rc.bias_scaling,
                input_mode=rc.input_mode, backend=rc.backend, device=rc.device)


def market_reservoir(sub: Subgraph, cfg: ExperimentConfig, W, n_features: int, seed: int) -> Reservoir:
    """The reservoir the market forecasts use (same input weights and bias for every wiring of a seed)."""
    return Reservoir(W, sub.input_idx, n_features, rng=np.random.default_rng([seed, 3]), **reservoir_kwargs(cfg))


def run_job(sub: Subgraph, datasets: dict, cfg: ExperimentConfig, wiring: str, seed: int) -> dict:
    """One wiring, one seed: build the reservoir, forecast every ticker, measure memory capacity."""
    t0 = time.time()
    rc, ev = cfg.reservoir, cfg.eval
    W, W_counts, sign, raw_gain = build_matrix(sub, cfg, wiring, seed)
    readout_idx = readout_neurons(sub, cfg, seed)
    common = reservoir_kwargs(cfg)

    rows, preds = [], {}
    for ticker, ds in datasets.items():
        res = market_reservoir(sub, cfg, W, ds.X.shape[1], seed)
        states = res.run(ds.X, record_idx=readout_idx, washout=rc.washout)
        post = ds.slice(rc.washout)
        if rc.readout_include_input:
            features = np.hstack([post.X, states])
            penalty = np.r_[np.full(post.X.shape[1], rc.input_penalty), np.ones(states.shape[1])]
        else:
            features, penalty = states, None
        pred, _ = walk_forward(features, post.target, post.horizon, ev.train_min, ev.refit_every, ev.window,
                               val_frac=ev.val_frac, penalty_factor=penalty)
        rows.append({"ticker": ticker, "model": wiring, "seed": seed,
                     **evaluate(pred, post.fwd_ret, post.next_ret, ev.position, ev.cost_bps)})
        preds[ticker] = pred

    _, spread, _ = top_mode(W, seed=seed)
    graph = {"wiring": wiring, "seed": seed, **graph_stats(W_counts, sign), "raw_gain": raw_gain,
             "top_mode_spread": spread}
    mc_curve = None
    if cfg.memory.enabled:
        res1 = Reservoir(W, sub.input_idx, 1, rng=np.random.default_rng([seed, 4]), **common)
        caps = memory_capacities(res1, readout_idx, memory_noise_levels(cfg), cfg.memory.n_steps,
                                 cfg.memory.max_delay, cfg.memory.washout, rng=np.random.default_rng([seed, 5]))
        graph["memory_capacity"], mc_curve = caps[0.0]
        if cfg.memory.readout_noise > 0:
            graph["memory_capacity_noisy"] = caps[cfg.memory.readout_noise][0]
        graph.update(readout_stability(res1, readout_idx, rng=np.random.default_rng([seed, 6]),
                                       n_tests=cfg.memory.stability_tests))
    return {"wiring": wiring, "seed": seed, "rows": rows, "preds": preds, "graph": graph, "mc": mc_curve,
            "seconds": time.time() - t0}


def run_baselines(datasets: dict, cfg: ExperimentConfig):
    """Same test period and fitting procedure as the reservoirs, without a reservoir."""
    ev, rc = cfg.eval, cfg.reservoir
    rows, preds = [], {}
    for ticker, ds in datasets.items():
        post = ds.slice(rc.washout)
        for name, X in (("linear_features", post.X), ("ar", post.X_ar)):
            pred, _ = walk_forward(X, post.target, post.horizon, ev.train_min, ev.refit_every, ev.window,
                                   val_frac=ev.val_frac)
            preds[(name, ticker)] = pred
        bh = np.full(len(post), np.nan)
        bh[ev.train_min:] = 1.0
        preds[("buy_hold", ticker)] = bh
        for name in BASELINES:
            p = preds[(name, ticker)]
            mode = "sign" if name == "buy_hold" else ev.position
            rows.append({"ticker": ticker, "model": name, "seed": -1,
                         **evaluate(p, post.fwd_ret, post.next_ret, mode, ev.cost_bps)})
    return rows, preds


# --------------------------------------------------------------------------- analysis

def _paired(a: pd.Series, b: pd.Series):
    """Mean of a - b over common seeds and the paired t-test p-value (NaN when undefined)."""
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).dropna()
    if len(d) == 0:
        return np.nan, np.nan
    if len(d) < 2 or np.allclose(d, d.iloc[0]):
        return float(d.mean()), np.nan
    return float(d.mean()), float(stats.ttest_1samp(d, 0.0).pvalue)


def compare(metrics: pd.DataFrame, graph: pd.DataFrame, preds: dict, datasets: dict, cfg: ExperimentConfig):
    """Connectome vs each control (paired over seeds) and vs baselines, plus a seed-ensemble bootstrap."""
    ev = cfg.eval
    rows, ensembles = [], {}
    wirings = [w for w in cfg.wirings if w in WIRINGS]
    for ticker, ds in datasets.items():
        post = ds.slice(cfg.reservoir.washout)
        ens_ret = {}
        for w in wirings:  # seed ensemble: average the predictions, then trade the average
            p = np.mean(np.stack([preds[(w, s, ticker)] for s in cfg.seeds]), axis=0)
            ens_ret[w] = strategy_returns(p, post.next_ret, ev.position, ev.cost_bps)[0]
        for b in BASELINES:
            mode = "sign" if b == "buy_hold" else ev.position
            ens_ret[b] = strategy_returns(preds[(b, ticker)], post.next_ret, mode, ev.cost_bps)[0]
        ensembles[ticker] = pd.DataFrame(ens_ret, index=post.dates)
        if "connectome" not in wirings:
            continue
        m = metrics[metrics["ticker"] == ticker]
        conn = m[m["model"] == "connectome"].set_index("seed")
        for other in [w for w in wirings if w != "connectome"] + list(BASELINES):
            row = {"ticker": ticker, "comparison": f"connectome vs {other}"}
            ref = m[m["model"] == other].set_index("seed")
            for metric in KEY_METRICS:
                if other in BASELINES:  # deterministic baseline: difference of means, no seed test
                    row[f"d_{metric}"], row[f"p_{metric}"] = conn[metric].mean() - ref[metric].mean(), np.nan
                else:
                    row[f"d_{metric}"], row[f"p_{metric}"] = _paired(conn[metric], ref[metric])
            boot = block_bootstrap_sharpe_diff(ens_ret["connectome"], ens_ret[other], ev.bootstrap_block, ev.n_boot)
            row.update({f"ens_sharpe_{k}": v for k, v in boot.items()})
            rows.append(row)
    if "memory_capacity" in graph.columns and "connectome" in wirings:
        mc = graph.set_index("seed")
        cols = [c for c in ("memory_capacity", "memory_capacity_noisy") if c in graph.columns]
        for other in [w for w in wirings if w != "connectome"]:
            row = {"ticker": "-", "comparison": f"connectome vs {other}"}
            for col in cols:
                row[f"d_{col}"], row[f"p_{col}"] = _paired(mc[mc["wiring"] == "connectome"][col],
                                                           mc[mc["wiring"] == other][col])
            rows.append(row)
    return pd.DataFrame(rows), ensembles


def _pm(values: pd.Series, digits: int = 3) -> str:
    values = values.dropna()
    if len(values) == 0:
        return "n/a"
    if len(values) == 1 or values.std() == 0:
        return f"{values.mean():.{digits}f}"
    return f"{values.mean():.{digits}f} ± {values.std():.{digits}f}"


def _dp(d, p, digits: int = 3) -> str:
    if d is None or not np.isfinite(d):
        return "n/a"
    return f"{d:+.{digits}f}" + (f" (p={p:.2f})" if p is not None and np.isfinite(p) else "")


def make_summary(cfg: ExperimentConfig, sub: Subgraph, datasets: dict, metrics: pd.DataFrame, graph: pd.DataFrame,
                 comparisons: pd.DataFrame) -> str:
    rc, sc, ev = cfg.reservoir, cfg.subgraph, cfg.eval
    source = "synthetic graph (NOT the real connectome)" if cfg.connectome.source == "synthetic" else \
        f"male CNS v1.0, >= {cfg.connectome.min_weight} synapses per edge"
    lines = [
        f"# flybrain-reservoir: {cfg.name}",
        "",
        f"Run {datetime.now():%Y-%m-%d %H:%M}. Connectome: {source}. Market data: {cfg.market.source}.",
        "",
        f"- Reservoir: {sub.n:,} neurons ({sc.method}), {len(sub.input_idx)} input neurons, readout from "
        f"{min(sc.n_readout, sub.n - len(sub.input_idx))} neurons"
        + (f" + the raw inputs (relative penalty {rc.input_penalty:g})" if rc.readout_include_input else "")
        + f"; spectral radius {rc.spectral_radius} ({rc.normalize}), leak {rc.leak_rate}, weights {rc.weight_transform}",
        f"- Wirings: {', '.join(cfg.wirings)}. Seeds: {len(cfg.seeds)}. Horizon: {cfg.market.horizon} day(s). "
        f"Costs: {ev.cost_bps} bps per unit turnover. Positions: {ev.position}.",
    ]
    for ticker, ds in datasets.items():
        post = ds.slice(rc.washout)
        test = post.dates[ev.train_min:]
        years = len(test) / 252
        fwd = post.fwd_ret[ev.train_min:]
        moved = fwd[np.isfinite(fwd) & (fwd != 0)]  # like metrics.hit_rate, flat days don't count
        up = float((moved > 0).mean())
        m = metrics[metrics["ticker"] == ticker]
        lines += ["", f"## {ticker}: out-of-sample, {test[0]:%Y-%m-%d} to {test[-1]:%Y-%m-%d} ({years:.1f} years)", "",
                  f"Up days in the test period: {up:.1%} (an always-long model gets this hit rate).", "",
                  "| model | IC | hit rate | Sharpe (net) | Sharpe (gross) | ann. return | max drawdown | turnover |",
                  "|---|---|---|---|---|---|---|---|"]
        for model in [w for w in cfg.wirings] + list(BASELINES):
            g = m[m["model"] == model]
            if len(g):
                lines.append(f"| {model} | {_pm(g['ic'])} | {_pm(g['hit_rate'])} | {_pm(g['sharpe_net'], 2)} | "
                             f"{_pm(g['sharpe_gross'], 2)} | {_pm(g['ann_return_net'])} | "
                             f"{_pm(g['max_drawdown_net'], 2)} | {_pm(g['turnover'], 2)} |")
        c = comparisons[comparisons["ticker"] == ticker] if len(comparisons) else comparisons
        if len(c):
            lines += ["", "### Connectome minus each alternative", "",
                      "Per-seed paired differences (t-test over seeds) and the Sharpe difference of the "
                      "seed-averaged strategies (moving-block bootstrap, 95% CI).", "",
                      "| comparison | Δ IC | Δ hit rate | Δ Sharpe (net) | ensemble Δ Sharpe [95% CI] |",
                      "|---|---|---|---|---|"]
            for _, r in c.iterrows():
                if np.isfinite(r.get("ens_sharpe_ci_low", np.nan)):
                    ci = (f"{r['ens_sharpe_diff']:+.2f} [{r['ens_sharpe_ci_low']:+.2f}, "
                          f"{r['ens_sharpe_ci_high']:+.2f}] (p={r['ens_sharpe_p_value']:.2f})")
                else:
                    ci = _dp(r.get("ens_sharpe_diff"), None, 2)
                lines.append(f"| {r['comparison']} | {_dp(r['d_ic'], r['p_ic'])} | "
                             f"{_dp(r['d_hit_rate'], r['p_hit_rate'])} | {_dp(r['d_sharpe_net'], r['p_sharpe_net'], 2)} "
                             f"| {ci} |")
        lines += ["", f"Rule of thumb: the standard error of an annualized Sharpe over {years:.0f} years is about "
                      f"{1 / np.sqrt(max(years, 1e-9)):.2f}, so differences much smaller than that are noise."]

    if "memory_capacity" in graph.columns:
        mc_cmp = comparisons[comparisons["ticker"] == "-"].set_index("comparison") if len(comparisons) else None
        noisy = "memory_capacity_noisy" in graph.columns
        noise = cfg.memory.readout_noise

        def diff(key, col):
            ok = mc_cmp is not None and key in mc_cmp.index and f"d_{col}" in mc_cmp.columns
            return _dp(mc_cmp.loc[key, f"d_{col}"], mc_cmp.loc[key, f"p_{col}"], 2) if ok else "-"

        lines += ["", "## Memory capacity (market-free benchmark)", "",
                  f"Sum over delays 1..{cfg.memory.max_delay} of R² for reconstructing past i.i.d. inputs. "
                  + (f"*With readout noise* adds noise of std {noise:g} to every readout neuron ({noise:.1%} "
                     "of its maximum activity) before fitting, so only memory that survives a little noise counts; "
                     "the noise-free number is the standard benchmark but can come from fluctuations of a millionth."
                     if noisy else ""),
                  "",
                  "| wiring | memory (noise-free) | connectome minus this |"
                  + (" memory with readout noise | connectome minus this |" if noisy else ""),
                  "|---|---|---|" + ("---|---|" if noisy else "")]
        for w in cfg.wirings:
            g = graph[graph["wiring"] == w]
            key = f"connectome vs {w}"
            line = f"| {w} | {_pm(g['memory_capacity'], 2)} | {diff(key, 'memory_capacity')} |"
            if noisy:
                line += f" {_pm(g['memory_capacity_noisy'], 2)} | {diff(key, 'memory_capacity_noisy')} |"
            lines.append(line)

    gain_label = "raw spectral radius" if rc.normalize == "spectral" else "raw bulk scale"
    has_stab = "unstable_readouts" in graph.columns
    lines += ["", "## Wiring structure and dynamics", "",
              f"| wiring | edges | reciprocity | largest SCC | {gain_label} | top mode spread | active readouts "
              "| unstable readouts |", "|---|---|---|---|---|---|---|---|"]
    for w in cfg.wirings:
        g = graph[graph["wiring"] == w]
        spread = f"~{g['top_mode_spread'].mean():,.0f} of {int(g['n'].iloc[0]):,}" if "top_mode_spread" in g else "-"
        active = f"{g['active_readouts'].mean():.0%}" if has_stab else "-"
        unstable = f"{g['unstable_readouts'].max():.0%}" if has_stab else "-"
        lines.append(f"| {w} | {int(g['edges'].mean()):,} | {_pm(g['reciprocity'])} | "
                     f"{_pm(g['largest_scc_frac'], 2)} | {_pm(g['raw_gain'], 2)} | {spread} | {active} | {unstable} |")
    lines += ["", "- Raw spectral radius: the largest eigenvalue before rescaling; every weight is divided by it "
                  f"(times {rc.spectral_radius}).",
              "- Top mode spread: roughly how many of the network's neurons carry that eigenvalue. A tiny share (say "
              "a few hundred of 166,700) means a small dense hot spot sets the gain for the whole network "
              "(`scripts/hot_spots.py` shows where it is).",
              "- Active readouts: share of readout neurons whose state varies by more than 0.001 under white-noise "
              "input (mean over seeds).",
              "- Unstable readouts: share whose state still depends on inputs from hundreds of steps ago (worst seed). "
              "Should be 0% for a valid reservoir."]
    if has_stab:
        bad = sorted(graph.loc[graph["unstable_readouts"] > 0.01, "wiring"].unique())
        if bad:
            lines += ["", f"**Warning:** {', '.join(bad)} did not forget its past (some neurons latch or go chaotic), so "
                          "the gain is too high for a valid reservoir and its results above shouldn't be compared."]
    lines += ["", "## Reading this", "",
              "- p-values come from paired tests over seeds (seed = input weights, readout neurons, control "
              "randomness). They capture seed-to-seed variation, not luck in the market history; the ensemble "
              "bootstrap covers that part.",
              "- A model that learns nothing predicts the average (positive) return and turns into buy & hold, so "
              "compare Sharpe against buy & hold and hit rate against the up-day rate. IC is the cleanest skill measure.",
              "- Several comparisons are run at once, so expect the odd p < 0.05 by chance.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- driver

def run_experiment(cfg: ExperimentConfig, verbose: bool = True, save: bool = True) -> ExperimentResult:
    t_start = time.time()
    unknown = [w for w in cfg.wirings if w not in WIRINGS]
    if unknown:
        raise ValueError(f"unknown wirings {unknown}; choose from {WIRINGS}")
    if not cfg.seeds or not cfg.wirings:
        raise ValueError("need at least one seed and one wiring")
    sub = prepare_subgraph(cfg, verbose)
    datasets = prepare_datasets(cfg)
    if verbose:
        for t, ds in datasets.items():
            print(f"data: {t} {ds.dates[0]:%Y-%m-%d} to {ds.dates[-1]:%Y-%m-%d} ({len(ds):,} rows)")

    jobs = [(w, s) for w in cfg.wirings for s in cfg.seeds]
    if verbose:
        print(f"running {len(jobs)} reservoir jobs ({len(cfg.wirings)} wirings x {len(cfg.seeds)} seeds)")
    if cfg.n_jobs == 1:
        outs = []
        for i, (w, s) in enumerate(jobs, 1):
            outs.append(run_job(sub, datasets, cfg, w, s))
            if verbose:
                print(f"  [{i}/{len(jobs)}] {w:<18} seed {s}: {outs[-1]['seconds']:.1f}s")
    else:
        outs = Parallel(n_jobs=cfg.n_jobs, verbose=10 if verbose else 0)(
            delayed(run_job)(sub, datasets, cfg, w, s) for w, s in jobs)

    base_rows, base_preds = run_baselines(datasets, cfg)
    metrics = pd.DataFrame([r for o in outs for r in o["rows"]] + base_rows)
    graph = pd.DataFrame([o["graph"] for o in outs])
    memory = pd.DataFrame([{"wiring": o["wiring"], "seed": o["seed"], "delay": k + 1, "mc": v}
                           for o in outs if o["mc"] is not None for k, v in enumerate(o["mc"])])
    preds = {(o["wiring"], o["seed"], t): p for o in outs for t, p in o["preds"].items()}
    preds.update(base_preds)

    washout = cfg.reservoir.washout
    pred_frames = []
    for key, p in preds.items():
        model, seed, ticker = (key[0], -1, key[1]) if len(key) == 2 else key
        pred_frames.append(pd.DataFrame({"date": datasets[ticker].dates[washout:], "ticker": ticker, "model": model,
                                         "seed": seed, "pred": p.astype(np.float32)}))
    predictions = pd.concat(pred_frames, ignore_index=True).dropna(subset=["pred"])

    comparisons, ensembles = compare(metrics, graph, preds, datasets, cfg)
    summary = make_summary(cfg, sub, datasets, metrics, graph, comparisons)
    result = ExperimentResult(metrics, memory, graph, comparisons, predictions, ensembles, summary)
    if save:
        result.out_dir = save_results(result, cfg)
    if verbose:
        print(f"done in {time.time() - t_start:.0f}s" + (f", results in {result.out_dir}" if save else ""))
    return result


def save_results(result: ExperimentResult, cfg: ExperimentConfig) -> Path:
    from . import plotting

    out = Path(cfg.output_dir) / cfg.name
    (out / "figures").mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(out / "metrics.csv", index=False)
    result.graph.to_csv(out / "graph_stats.csv", index=False)
    result.comparisons.to_csv(out / "comparisons.csv", index=False)
    result.memory.to_csv(out / "memory_capacity.csv", index=False)
    result.predictions.to_parquet(out / "predictions.parquet", index=False)
    (out / "summary.md").write_text(result.summary, encoding="utf-8")
    save_config(cfg, out / "config_used.yaml")

    import matplotlib.pyplot as plt

    for ticker, ens in result.ensemble_returns.items():
        ens.to_csv(out / f"ensemble_returns_{ticker}.csv")
        plt.close(plotting.plot_metric_strip(result.metrics, ticker, path=out / "figures" / f"metrics_{ticker}.png"))
        test = ens.dropna(how="all")
        plt.close(plotting.plot_equity({c: test[c].to_numpy() for c in test.columns}, test.index,
                                       path=out / "figures" / f"equity_{ticker}.png",
                                       title=f"{ticker}: seed-ensemble strategies, growth of 1 (net of costs)"))
    if len(result.memory):
        plt.close(plotting.plot_memory_curves(result.memory, path=out / "figures" / "memory_capacity.png"))
    return out
