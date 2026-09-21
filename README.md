# flybrain-reservoir

**Does a real fruit fly brain make a better forecasting machine than random wiring?**

This project uses the wiring diagram of the adult male *Drosophila* central nervous system
(166,700 neurons, released in 2026 by FlyEM/HHMI Janelia with Google Research) as the recurrent
network of an echo state network, feeds it market data, and compares it against control
networks that keep some of its statistics and scramble the rest.

The goal isn't to beat the market with a fly (don't expect that). The goal is a clean answer to a
narrower question: *is biological wiring special as a reservoir, compared to random wiring with
matching statistics?*

## How it works

Reservoir computing: a fixed recurrent network turns an input time series into a
high-dimensional state that carries memory of the past plus nonlinear mixes of it. Only a
linear readout on top is trained.

```
x[t+1] = (1 - a) * x[t] + a * tanh(W x[t] + W_in u[t] + b)      # reservoir, never trained
ŷ[t]   = w_out · [u[t], x_R[t]]                                  # readout, ridge regression
```

- **W** = the connectome: synapse counts (log1p) between the selected neurons, signed by
  neurotransmitter under Dale's law (GABA, glutamate, histamine inhibitory, everything else
  excitatory, the rule from Shiu et al. 2024), rescaled to spectral radius 0.9.
- **u[t]** = 5 market features (1/5/20-day returns, 20-day volatility, vol ratio), z-scored
  against a trailing window only. They enter through **sensory neurons** only.
- **x_R[t]** = states of 300 randomly chosen non-sensory neurons.
- **Target** = next-day return divided by trailing volatility. Position = sign of the forecast,
  1 bp cost per unit of turnover.
- **Evaluation** = walk-forward: 3 years of history before the first forecast, readout refit
  every 6 months on an expanding window, ridge penalty picked on the last 20% of each training
  window. A model fitted on day t only trains on targets already known on day t (tested).

### Controls (same neurons, inputs and readouts; only W changes)

| wiring | keeps | scrambles |
|---|---|---|
| `connectome` | everything | nothing |
| `degree_preserving` | each neuron's in/out degree and outgoing weights | who connects to whom (reciprocity, motifs, modules) |
| `weight_shuffle` | exact topology | which connection has which synapse count |
| `sign_shuffle` | topology, weights, E/I ratio | which neurons are inhibitory |
| `erdos_renyi` | number of edges, weight distribution | all structure |

All controls are rescaled to the same spectral radius. For a given seed, every wiring gets the
same input weights, bias and readout neurons, so per-seed differences are paired comparisons.

### Baselines
Linear ridge on the same 5 features, AR ridge on 10 lagged returns, buy & hold. Same test days,
same fitting code.

### Market-free benchmark
**Memory capacity** (Jaeger 2001): feed white noise, train readouts to reconstruct the input from
k steps ago, sum the R² over k. It shows how much input history each wiring keeps, which is a
cleaner test of "is the wiring special" than noisy market returns.

## Setup (Windows)

```bat
cd Desktop\Code\flybrain-reservoir
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
python scripts\run_experiment.py --config configs\demo_synthetic.yaml
```

`pytest` runs on synthetic data in a few seconds. The demo runs the whole pipeline offline on a
fake connectome and fake prices with a planted signal (~15 s), just to prove everything works.

## Running it on the real brain

```bat
python scripts\download_data.py          :: ~1.2 GB from Janelia's public bucket, resumable
python scripts\build_connectome.py       :: parse + cache the graph (a few minutes, once)
python scripts\run_experiment.py --config configs\small.yaml
```

Results go to `results/small/`. Start with `summary.md`; the CSVs and `figures/` have the rest.
`notebooks/01_connectome_tour.ipynb` explores the graph and its eigenvalue spectrum,
`notebooks/02_results.ipynb` runs and plots an experiment.

Override anything from the command line:

```bat
python scripts\run_experiment.py --config configs\small.yaml --set subgraph.n_neurons=10000 --set "seeds=[0,1,2,3,4,5,6,7,8,9]"
```

### Scaling up
- Subgraph size is just `subgraph.n_neurons`; everything is sparse, so 3k to 50k is a config change.
- `configs/full.yaml` runs the whole CNS (`method: all`) with 4 parallel jobs. Meant for a
  desktop (32 GB RAM is plenty); expect roughly an hour.
- GPU: `pip install torch` (a ROCm build for AMD cards) and `--set reservoir.backend=torch`.
  ROCm support on Windows has been patchy, so check the current PyTorch install page first; CPU
  works fine, just slower.

## Design decisions (and why)

1. **Readout sees the raw inputs too.** A readout neuron only hears today's input one step later
   (it has to travel through W), so a states-only readout can't use the freshest information.
   On synthetic data with a planted signal, the states-only readout found almost nothing.
2. **Raw inputs are (almost) unpenalized in the ridge.** Then the reservoir model contains the
   linear model as a special case and the reservoir states act as a regularized correction.
   With one shared penalty, adding even 25 states made it *worse* than plain linear (slow,
   persistent regressors overfit noisy targets).
3. **Leak rate 1.0.** No per-neuron smoothing, so all memory has to come from the wiring, which is
   the thing being tested. Leaky neurons add memory that has nothing to do with W.
4. **Equal spectral radius for every wiring.** Same overall gain. Caveat: a heavy-tailed graph's
   biggest eigenvalue can be an outlier driven by a few hubs, which leaves the rest of its spectrum
   small after rescaling. `normalize: frobenius` matches the bulk instead, as a robustness check.
5. **Edges need ≥ 5 synapses.** Standard threshold to drop noisy connections (`connectome.min_weight`).
6. **Inputs = the most-connected sensory neurons**, and the subgraph is grown from them by
   repeatedly adding the neurons most strongly connected to it, so the signal can actually
   propagate. `subgraph.input_filter` can pick one modality, e.g. `{class: olfactory}`.

## What to expect (honestly)

- Daily index returns are close to unpredictable. Expect ICs near 0 to 0.03 and hit rates close to
  the share of up days for everything, fly included.
- A model that learns nothing predicts the average return, which is positive, so it turns into
  buy & hold. Compare Sharpe against buy & hold and hit rate against the up-day rate; IC is the
  cleanest skill measure.
- The standard error of an annualized Sharpe over ~18 years is about 0.24. Small differences are noise.
- The interesting result is relative: connectome vs controls in forecasting and in memory capacity.
  "Real wiring is no better than a degree-matched random graph" is a real, reportable finding.

### Limitations
- Rate model, not spiking. A leaky integrate-and-fire version is the obvious next step.
- The sign rule is an approximation (glutamate isn't inhibitory everywhere, neuromodulators are
  treated as excitatory, "unclear" transmitters default to excitatory).
- One subgraph-growing rule; results could depend on which part of the brain you take.

## Repo layout

```
src/flyres/
  connectome.py   download, parse and cache the male CNS  (W[post, pre] = synapse count)
  subgraph.py     choose the reservoir neurons (grow from sensory neurons, top degree, or all)
  controls.py     null-model wirings
  reservoir.py    echo state network (numpy/scipy or torch)
  market.py       prices -> causal features and targets
  readout.py      ridge regression and walk-forward refits
  metrics.py      IC, hit rate, Sharpe, drawdown, block bootstrap
  benchmarks.py   memory capacity
  experiment.py   runs everything and writes results
  plotting.py     figures
  synthetic.py    fake data for tests and the offline demo
scripts/          download_data.py, build_connectome.py, run_experiment.py
configs/          small.yaml (laptop), full.yaml (whole CNS), demo_synthetic.yaml (offline)
notebooks/        01_connectome_tour.ipynb, 02_results.ipynb
tests/            pytest suite (no downloads needed)
```

## Data and credits

- Male CNS v1.0 connectome: FlyEM (HHMI Janelia), University of Cambridge, MRC LMB and Google
  Research, <https://male-cns.janelia.org>, licensed CC-BY 4.0. Cite the dataset paper listed on
  the project site if you use it.
- Sign rule: Shiu et al. 2024, *A Drosophila computational brain model reveals sensorimotor
  processing*, Nature.
- Memory capacity: Jaeger 2001, *Short term memory in echo state networks*.
- Degree-preserving rewiring: Maslov & Sneppen 2002, *Specificity and stability in topology of protein networks*, Science.
- File schema cross-checked against [sstamou03/fly_brain](https://github.com/sstamou03/fly_brain).
