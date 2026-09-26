"""Volatility forecasting: the same reservoirs, asked a question that has an answer.

Tomorrow's return is close to unpredictable, so the returns experiment can't show whether a wiring's
memory is worth anything in finance. Volatility is different: it clusters and decays slowly, so
remembering the past pays. Same reservoirs (same wiring, input weights, bias, readout neurons and
market inputs as the returns experiment), a second readout, a different target:

    y_t = log( (1/h) * sum_{k=1..h} r_{t+k}^2 )      realized variance over the next h days (log)

Benchmarks, all fitted walk-forward with the same ridge code and test period, and with the same (practically
nil) penalty that every reservoir readout puts on these columns:
    ewma        RiskMetrics EWMA variance (lambda 0.94), in logs, with a fitted intercept and slope
    har         log-HAR (Corsi 2009): y on log realized variance over the last 1, 5 and 22 days, practically
                OLS as in Corsi. The standard benchmark, and hard to beat
    har_inputs  HAR plus the reservoir's 5 input features, linear: everything the reservoir sees today,
                without memory or nonlinearity
    har_levels  the classic HAR-RV, fitted on variance instead of log variance (reference for QLIKE)

Every reservoir's readout sees [HAR + inputs (practically unpenalized), states]. har_inputs is fitted the
same way on the same columns, so it is exactly a reservoir readout with its states removed, and reservoir
minus har_inputs is what the states contribute. (Fitting the benchmarks with a tuned ridge penalty instead
would let the penalty, not the states, account for part of the difference: on overlapping multi-day
targets the validation block is short, and it sometimes picks a penalty that flattens the forecast.)

Diebold-Mariano tests use the Harvey-Leybourne-Newbold small-sample version, as in R's forecast::dm.test.

Losses:
    log MSE  squared error on log variance, reported as R² against the expanding historical mean. The main
             loss: it is what every log model (HAR, HAR + inputs, all reservoirs) is fitted for, so they
             are compared like for like.
    QLIKE    RV / f - log(RV / f) - 1 on a variance forecast f, the usual loss in the volatility literature
             (robust to noise in the realized-variance proxy, Patton 2011). A log forecast becomes a
             variance forecast with the lognormal correction exp(ŷ + s²/2), s² = the readout's validation
             error at fit time. Log models target E[log RV], not E[RV], which costs them on QLIKE: on
             simulated GARCH data, HAR fitted on variance beats HAR fitted on logs on QLIKE (a few percent at
             5 days, 13-24% at 22 days). So QLIKE is a second view with har_levels as its reference, not the headline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from .config import ExperimentConfig
from .controls import WIRINGS
from .experiment import (build_matrix, load_closes, market_reservoir, prepare_subgraph, readout_neurons)
from .market import Dataset, make_dataset
from .readout import walk_forward

HAR_LAGS = (1, 5, 22)
DAY_FLOOR = 1e-8       # (1 bp)²: floor on one day's squared return before taking logs
BASELINES = ("ewma", "har_levels", "har", "har_inputs")
LABELS = {"ewma": "EWMA", "har": "HAR", "har_inputs": "HAR + inputs (linear)", "har_levels": "HAR (levels)"}


@dataclass
class VolData:
    """Aligned with a market Dataset: row t = close of day t."""

    ticker: str
    dates: pd.DatetimeIndex
    X: np.ndarray           # the reservoir's (standardized) inputs, same as the market Dataset
    har: np.ndarray         # (T, 3) log realized variance over the last 1, 5, 22 days
    ewma: np.ndarray        # (T, 1) log EWMA variance forecast for the next day
    y: dict                 # horizon -> (T,) log realized variance over the next h days (NaN at the end)
    rv: dict                # horizon -> (T,) the same in variance units (for QLIKE)

    def slice(self, start: int) -> "VolData":
        return replace(self, dates=self.dates[start:], X=self.X[start:], har=self.har[start:],
                       ewma=self.ewma[start:], y={h: v[start:] for h, v in self.y.items()},
                       rv={h: v[start:] for h, v in self.rv.items()})


def make_vol_data(close: pd.Series, ds: Dataset, horizons=(5, 22), ewma_lambda: float = 0.94) -> VolData:
    """Realized-variance features and targets for the rows of `ds` (built from the same close series).
    Features at row t use returns up to and including day t; targets use days t+1 .. t+h only."""
    close = close.dropna().astype(float)
    r = np.log(close).diff()
    r2 = r ** 2
    har = pd.DataFrame({f"log_rv{k}": np.log(r2.rolling(k).mean().clip(lower=DAY_FLOOR)) for k in HAR_LAGS})
    ewma = np.log(r2.ewm(alpha=1.0 - ewma_lambda, adjust=False).mean().clip(lower=DAY_FLOOR))
    rv = {h: r2.rolling(h).mean().shift(-h).clip(lower=DAY_FLOOR / h) for h in horizons}
    idx = pd.DatetimeIndex(ds.dates)
    har, ewma = har.reindex(idx), ewma.reindex(idx)
    if har.isna().any().any() or ewma.isna().any():
        raise ValueError(f"{ds.ticker}: missing realized-variance features inside the dataset's rows")
    rv = {h: s.reindex(idx).to_numpy(np.float64) for h, s in rv.items()}
    return VolData(ticker=ds.ticker, dates=idx, X=ds.X, har=har.to_numpy(np.float64),
                   ewma=ewma.to_numpy(np.float64)[:, None], y={h: np.log(v) for h, v in rv.items()}, rv=rv)


# --------------------------------------------------------------------------- forecasts and losses

def forecast(X, y, horizon: int, cfg: ExperimentConfig, penalty=None):
    """Walk-forward log-variance forecasts and the matching variance forecasts (lognormal correction)."""
    ev = cfg.eval
    logp, splits = walk_forward(X, y, horizon, ev.train_min, ev.refit_every, ev.window, val_frac=ev.val_frac,
                                penalty_factor=penalty)
    s2 = np.full(len(y), np.nan)
    for _, s in splits.iterrows():
        s2[int(s["t0"]):int(s["t1"])] = s["val_mse"]
    return logp, np.exp(logp + 0.5 * s2)


def qlike(rv, f) -> np.ndarray:
    x = np.asarray(rv, dtype=np.float64) / np.asarray(f, dtype=np.float64)
    return x - np.log(x) - 1.0


def historical_mean(y: np.ndarray, horizon: int) -> np.ndarray:
    """Expanding mean of the targets already known at each row (row s is known at s + horizon)."""
    return pd.Series(y).shift(horizon).expanding(min_periods=1).mean().to_numpy()


def vol_metrics(logp, varp, y, rv, hist) -> dict:
    ok = np.isfinite(logp) & np.isfinite(varp) & np.isfinite(y) & np.isfinite(hist)
    e = y[ok] - logp[ok]
    mse = float((e ** 2).mean())
    c = np.corrcoef(logp[ok], y[ok])[0, 1] if ok.sum() > 2 and np.std(logp[ok]) > 0 else np.nan
    return {"n_test": int(ok.sum()), "qlike": float(qlike(rv[ok], varp[ok]).mean()), "mse_log": mse,
            "r2_log": 1.0 - mse / float(((y[ok] - hist[ok]) ** 2).mean()), "mz_r2": float(c ** 2),
            "bias_log": float(e.mean())}


def diebold_mariano(loss_a, loss_b, horizon: int) -> tuple[float, float]:
    """Diebold-Mariano test of equal expected loss for `horizon`-step forecasts, in the small-sample version
    of Harvey, Leybourne & Newbold (1997), as in R's forecast::dm.test: the variance of the mean loss
    difference uses its autocovariances up to lag h-1 with equal weights (h-day targets overlap by h-1 days;
    Bartlett weights if that estimate isn't positive), the statistic is scaled by the HLN correction and
    compared with a t distribution with n-1 degrees of freedom.
    Returns (t, two-sided p); t < 0 means `a` has the lower loss."""
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    d = d[np.isfinite(d)]
    n, h = len(d), int(horizon)
    if n < 10 or h < 1 or np.allclose(d, 0):
        return float("nan"), float("nan")
    dc = d - d.mean()
    gamma = np.array([dc[k:] @ dc[:n - k] / n for k in range(min(h, n - 1))])  # autocovariances, lags 0..h-1
    var = gamma[0] + 2.0 * gamma[1:].sum()
    if var <= 0:  # the equal-weight estimate can go negative; Bartlett weights can't
        var = gamma[0] + 2.0 * ((1.0 - np.arange(1, len(gamma)) / h) * gamma[1:]).sum()
    if var <= 0:
        return float("nan"), float("nan")
    t = d.mean() / np.sqrt(var / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(t), float(2.0 * stats.t.sf(abs(t), df=n - 1))


# --------------------------------------------------------------------------- jobs

def prepare_vol_data(cfg: ExperimentConfig) -> dict[str, VolData]:
    """Per ticker, cut at the reservoir washout so rows line up with the recorded states."""
    mc, vc, w = cfg.market, cfg.volatility, cfg.reservoir.washout
    out = {}
    for ticker, close in load_closes(cfg).items():
        ds = make_dataset(close, mc.features, mc.horizon, mc.z_window, mc.ar_lags, ticker=ticker)
        out[ticker] = (ds, make_vol_data(close, ds, vc.horizons, vc.ewma_lambda).slice(w))
    return out


def vol_job(sub, data: dict, cfg: ExperimentConfig, wiring: str, seed: int, gain: float | None = None) -> dict:
    """One wiring and seed: the returns experiment's reservoir (optionally at another gain), a readout per
    horizon."""
    t0 = time.time()
    rc = cfg.reservoir
    W, *_ = build_matrix(sub, cfg, wiring, seed)
    if gain is not None:
        W = (W * np.float32(gain / rc.spectral_radius)).tocsr()
    readout_idx = readout_neurons(sub, cfg, seed)
    rows, preds = [], {}
    for ticker, (ds, v) in data.items():
        states = market_reservoir(sub, cfg, W, ds.X.shape[1], seed).run(ds.X, record_idx=readout_idx,
                                                                       washout=rc.washout)
        lin = np.hstack([v.har, v.X])
        X = np.hstack([lin, states])
        penalty = np.r_[linear_penalty(lin, cfg), np.ones(states.shape[1])]
        for h in v.y:
            logp, varp = forecast(X, v.y[h], h, cfg, penalty)
            rows.append({"ticker": ticker, "horizon": h, "model": wiring, "seed": seed,
                         **vol_metrics(logp, varp, v.y[h], v.rv[h], historical_mean(v.y[h], h))})
            preds[(ticker, h)] = (logp, varp)
    return {"wiring": wiring, "seed": seed, "rows": rows, "preds": preds, "seconds": time.time() - t0}


def linear_penalty(X: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    """The penalty every reservoir readout puts on its HAR and input columns (practically none), for fitting
    the linear benchmarks exactly the same way."""
    return np.full(X.shape[1], cfg.reservoir.input_penalty)


def vol_baselines(data: dict, cfg: ExperimentConfig):
    ev = cfg.eval
    rows, preds = [], {}
    for ticker, (_, v) in data.items():
        inputs = {"ewma": v.ewma, "har": v.har, "har_inputs": np.hstack([v.har, v.X])}
        for h in v.y:
            out = {name: forecast(X, v.y[h], h, cfg, linear_penalty(X, cfg)) for name, X in inputs.items()}
            level, _ = walk_forward(np.exp(v.har), v.rv[h], h, ev.train_min, ev.refit_every, ev.window,
                                    val_frac=ev.val_frac, penalty_factor=linear_penalty(v.har, cfg))
            level = np.where(np.isfinite(level), np.maximum(level, DAY_FLOOR), np.nan)  # OLS can go negative
            out["har_levels"] = (np.log(level), level)
            for name, (logp, varp) in out.items():
                rows.append({"ticker": ticker, "horizon": h, "model": name, "seed": -1,
                             **vol_metrics(logp, varp, v.y[h], v.rv[h], historical_mean(v.y[h], h))})
                preds[(name, ticker, h)] = (logp, varp)
    return rows, preds


def compare_vol(data: dict, preds: dict, metrics: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Seed-ensemble forecasts (mean over seeds) against HAR and HAR + inputs, and connectome vs each control.
    Diebold-Mariano test on squared log errors (HLN version, see diebold_mariano), the QLIKE change as a
    second view, and for connectome vs a control the paired t-test over seeds."""
    wirings = [w for w in cfg.wirings if w in WIRINGS]
    rows = []
    for ticker, (_, v) in data.items():
        for h in v.y:
            y, rv = v.y[h], v.rv[h]
            ens = {w: tuple(np.mean([preds[(w, s, ticker, h)][i] for s in cfg.seeds], axis=0) for i in (0, 1))
                   for w in wirings}
            ens.update({b: preds[(b, ticker, h)] for b in BASELINES})
            m = metrics[(metrics["ticker"] == ticker) & (metrics["horizon"] == h)]
            pairs = [(w, b) for w in wirings for b in ("har", "har_inputs")]
            pairs += [("connectome", w) for w in wirings if w != "connectome"] if "connectome" in wirings else []
            for a, b in pairs:
                sa, sb = (y - ens[a][0]) ** 2, (y - ens[b][0]) ** 2
                qa, qb = qlike(rv, ens[a][1]), qlike(rv, ens[b][1])
                ok = np.isfinite(sa) & np.isfinite(sb) & np.isfinite(qa) & np.isfinite(qb)
                t, p = diebold_mariano(sa[ok], sb[ok], horizon=h)
                row = {"ticker": ticker, "horizon": h, "comparison": f"{a} vs {b}",
                       "ens_mse_change_pct": 100.0 * (sa[ok].mean() / sb[ok].mean() - 1.0), "dm_t": t, "dm_p": p,
                       "ens_qlike_change_pct": 100.0 * (qa[ok].mean() / qb[ok].mean() - 1.0)}
                if a in wirings and b in wirings:
                    ma = m[m["model"] == a].set_index("seed")["mse_log"]
                    mb = m[m["model"] == b].set_index("seed")["mse_log"]
                    d = (ma - mb.reindex(ma.index)).dropna()
                    row["seed_mse_diff"] = float(d.mean())
                    row["seed_p"] = float(stats.ttest_1samp(d, 0.0).pvalue) if len(d) > 1 and d.std() > 0 else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def memory_link(metrics: pd.DataFrame, graph: pd.DataFrame | None) -> pd.DataFrame | None:
    """Per reservoir (wiring, seed): memory from the main experiment (or the gain sweep) next to how much its
    states improve on HAR + inputs (log MSE, %; negative = better). None without memory numbers."""
    if graph is None or not len(graph):
        return None
    col = "memory_capacity_noisy" if "memory_capacity_noisy" in graph.columns else "memory_capacity"
    if col not in graph.columns:
        return None
    base = metrics[metrics["model"] == "har_inputs"].set_index(["ticker", "horizon"])["mse_log"]
    res = metrics[metrics["model"].isin(WIRINGS)].copy()
    res["mse_vs_har_inputs_pct"] = 100.0 * (res["mse_log"] / base.reindex(
        pd.MultiIndex.from_frame(res[["ticker", "horizon"]])).to_numpy() - 1.0)
    out = res.merge(graph[["wiring", "seed", col]].rename(columns={"wiring": "model", col: "memory"}),
                    on=["model", "seed"], how="inner")
    out["memory_measure"] = "with readout noise" if col == "memory_capacity_noisy" else "noise-free"
    return out if len(out) else None


def within_wiring_spearman(memory, error, wiring, n_perm: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Rank correlation between memory and error inside each wiring (ranks taken and centered within the
    wiring, then pooled), with a permutation p-value that shuffles memory only within wirings.

    Reservoirs of one wiring share its structure, so a correlation over all reservoirs mostly compares
    wirings, and its textbook p-value treats them as independent (50 reservoirs are really 5 wirings).
    This asks the question that can be tested: within a wiring, do seeds with more memory forecast better?"""
    frame = pd.DataFrame({"m": memory, "e": error, "w": wiring}).dropna()
    groups = [g for _, g in frame.groupby("w") if len(g) > 1]
    if not groups:
        return float("nan"), float("nan")
    rm = [stats.rankdata(g["m"]) - (len(g) + 1) / 2 for g in groups]
    re = np.concatenate([stats.rankdata(g["e"]) - (len(g) + 1) / 2 for g in groups])

    def corr(parts):
        x = np.concatenate(parts)
        den = np.sqrt((x ** 2).sum() * (re ** 2).sum())
        return float((x * re).sum() / den) if den > 0 else float("nan")

    obs = corr(rm)
    if not np.isfinite(obs):
        return obs, float("nan")
    rng = np.random.default_rng(seed)
    perm = np.array([corr([rng.permutation(r) for r in rm]) for _ in range(n_perm)])
    return obs, float((1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (n_perm + 1))


# --------------------------------------------------------------------------- driver

@dataclass
class VolResult:
    metrics: pd.DataFrame
    comparisons: pd.DataFrame
    predictions: pd.DataFrame
    link: pd.DataFrame | None
    data: dict
    gains: dict | None = None   # wiring -> gain, when not the config's spectral radius


def run_vol_experiment(cfg: ExperimentConfig, graph: pd.DataFrame | None = None, verbose: bool = True,
                       gains: dict | None = None) -> VolResult:
    """`graph`: per (wiring, seed) memory capacity, to relate memory to forecasting skill. `gains`: run each
    wiring at its own gain (e.g. its best valid gain from the gain sweep) instead of the config's."""
    unknown = [w for w in cfg.wirings if w not in WIRINGS]
    if unknown:
        raise ValueError(f"unknown wirings {unknown}; choose from {WIRINGS}")
    sub = prepare_subgraph(cfg, verbose)
    data = prepare_vol_data(cfg)
    jobs = [(w, s) for w in cfg.wirings for s in cfg.seeds]
    if verbose:
        for t, (_, v) in data.items():
            print(f"data: {t} {v.dates[0]:%Y-%m-%d} to {v.dates[-1]:%Y-%m-%d}, horizons {list(v.y)}")
        print(f"volatility: {len(jobs)} reservoir jobs ({len(cfg.wirings)} wirings x {len(cfg.seeds)} seeds)")
    if cfg.n_jobs == 1:
        outs = []
        for i, (w, s) in enumerate(jobs, 1):
            outs.append(vol_job(sub, data, cfg, w, s, (gains or {}).get(w)))
            if verbose:
                print(f"  [{i}/{len(jobs)}] {w:<18} seed {s}: {outs[-1]['seconds']:.1f}s", flush=True)
    else:
        outs = Parallel(n_jobs=cfg.n_jobs, verbose=10 if verbose else 0)(
            delayed(vol_job)(sub, data, cfg, w, s, (gains or {}).get(w)) for w, s in jobs)
    base_rows, preds = vol_baselines(data, cfg)
    metrics = pd.DataFrame([r for o in outs for r in o["rows"]] + base_rows)
    preds.update({(o["wiring"], o["seed"], t, h): p for o in outs for (t, h), p in o["preds"].items()})
    n_test = metrics.groupby(["ticker", "horizon"])["n_test"].nunique()
    if (n_test > 1).any():
        raise RuntimeError("models were evaluated on different test days")
    comparisons = compare_vol(data, preds, metrics, cfg)

    frames = []
    for key, (logp, varp) in preds.items():
        model, seed, ticker, h = (key[0], -1, key[1], key[2]) if len(key) == 3 else key
        v = data[ticker][1]
        frames.append(pd.DataFrame({"date": v.dates, "ticker": ticker, "horizon": h, "model": model, "seed": seed,
                                    "log_var_pred": logp.astype(np.float32), "var_pred": varp.astype(np.float32),
                                    "realized_var": v.rv[h].astype(np.float32)}))
    predictions = pd.concat(frames, ignore_index=True).dropna(subset=["log_var_pred"])
    return VolResult(metrics, comparisons, predictions, memory_link(metrics, graph), data, gains)


# --------------------------------------------------------------------------- summary

def _order(models) -> list:
    models = set(models)
    return [w for w in WIRINGS if w in models] + [b for b in reversed(BASELINES) if b in models]


def _pm(s: pd.Series, digits: int = 3) -> str:
    s = s.dropna()
    if s.empty:
        return "-"
    return f"{s.mean():.{digits}f}" + (f" ± {s.std():.{digits}f}" if len(s) > 1 and s.std() > 0 else "")


def vol_markdown(cfg: ExperimentConfig, res: VolResult) -> str:
    m, c = res.metrics, res.comparisons
    lines = [f"# Volatility forecasts: {cfg.name}", "",
             "Target: log realized variance over the next h days (mean of squared daily log returns). Every model "
             "is fitted walk-forward on the same test days as the returns experiment. Reservoir readouts see "
             "HAR + the 5 market inputs (practically unpenalized) + the reservoir states; the linear benchmarks are "
             "fitted with the same penalty on the same columns, so HAR + inputs is a reservoir readout without its "
             "states and *reservoir minus HAR + inputs* is what the states add.", "",
             "- R² (log): out-of-sample, against the expanding historical mean; *log MSE vs HAR* is the change in "
             "squared error on log variance (negative = better than HAR). The main measure: every log model is "
             "fitted for exactly this.",
             "- QLIKE: the usual robust loss on the variance itself (lower is better). Log models aren't fitted for "
             "it; HAR (levels) is, so it is the reference there.", ""]
    if res.gains:
        lines += ["Reservoir gains: each wiring at its best valid gain from the gain sweep ("
                  + ", ".join(f"{w} {g:g}" for w, g in res.gains.items()) + ").", ""]
    else:
        lines += [f"Reservoir gain: spectral radius {cfg.reservoir.spectral_radius} for every wiring, as in the "
                  "returns experiment.", ""]
    for (ticker, h), g in m.groupby(["ticker", "horizon"], sort=True):
        v = res.data[ticker][1]
        har = g.loc[g["model"] == "har", "mse_log"].iloc[0]
        har_q = g.loc[g["model"] == "har_levels", "qlike"].iloc[0]
        test = v.dates[cfg.eval.train_min:]
        lines += [f"## {ticker}, next {h} days: {test[0]:%Y-%m-%d} to {test[-1]:%Y-%m-%d} "
                  f"({g['n_test'].iloc[0]:,} forecasts)", "",
                  "| model | R² (log) | log MSE vs HAR | MZ R² | QLIKE | QLIKE vs HAR (levels) |",
                  "|---|---|---|---|---|---|"]
        for model in _order(g["model"]):
            r = g[g["model"] == model]
            lines.append(f"| {LABELS.get(model, model)} | {_pm(r['r2_log'])} | "
                         f"{100 * (r['mse_log'].mean() / har - 1):+.1f}% | {_pm(r['mz_r2'])} | {_pm(r['qlike'])} | "
                         f"{100 * (r['qlike'].mean() / har_q - 1):+.1f}% |")
        cc = c[(c["ticker"] == ticker) & (c["horizon"] == h)]
        if len(cc):
            lines += ["", "Seed-ensemble forecasts (mean over seeds). Diebold-Mariano test on squared log errors "
                          "(Harvey-Leybourne-Newbold version, autocovariances up to lag "
                          f"{h - 1}); for connectome vs a control also the paired t-test over seeds.", "",
                      "| comparison | log MSE change | DM t | DM p | QLIKE change | seeds: Δ log MSE (p) |",
                      "|---|---|---|---|---|---|"]
            for _, r in cc.iterrows():
                d = r.get("seed_mse_diff", np.nan)
                seeds = f"{d:+.4f} (p={r['seed_p']:.2f})" if np.isfinite(d) else ""
                name = r["comparison"].replace("har_inputs", "HAR + inputs").replace(" har", " HAR")
                lines.append(f"| {name} | {r['ens_mse_change_pct']:+.1f}% | {r['dm_t']:+.2f} | {r['dm_p']:.3f} | "
                             f"{r['ens_qlike_change_pct']:+.1f}% | {seeds} |")
        lines.append("")
    if res.link is not None:
        measure = res.link["memory_measure"].iloc[0]
        source = "the gain sweep, at the same gains" if res.gains else "the main experiment"
        lines += ["## Does memory help?", "",
                  f"Each reservoir's memory capacity ({measure}, from {source}) against how much its states change "
                  "the log MSE of HAR + inputs; negative = more memory, lower error. *All reservoirs*: Spearman "
                  "correlation over every wiring and seed, which mostly compares wirings (descriptive; the reservoirs "
                  "of one wiring aren't independent). *Within wirings*: the same with ranks taken inside each wiring, "
                  "and a p-value from shuffling memory within wirings (10,000 permutations).", "",
                  "| ticker | horizon | reservoirs | ρ, all reservoirs | ρ, within wirings | p (within) |",
                  "|---|---|---|---|---|---|"]
        for (ticker, h), g in res.link.groupby(["ticker", "horizon"]):
            rho = stats.spearmanr(g["memory"], g["mse_vs_har_inputs_pct"])[0]
            rho_w, p_w = within_wiring_spearman(g["memory"], g["mse_vs_har_inputs_pct"], g["model"])
            lines.append(f"| {ticker} | {h} | {len(g)} | {rho:+.2f} | {rho_w:+.2f} | {p_w:.3f} |")
        lines.append("")
    return "\n".join(lines)
