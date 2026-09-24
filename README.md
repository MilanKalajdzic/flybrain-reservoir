# flybrain-reservoir

**Does a real fruit fly brain make a better forecasting machine than random wiring?**

<p align="center">
  <img src="docs/img/brain_2008.gif" width="820" alt="Animated frontal view of the fly brain: 3,000 connectome neurons glow brighter as the 2008 crash deepens">
</p>

*3,000 neurons of the fly's sensory circuitry (mostly the smell pathway), wired exactly as in the
male CNS connectome and driven by SPY returns and volatility through its sensory neurons. Brighter = further
from that neuron's normal activity. It reacts to the crash; it doesn't see it coming.
([COVID version](docs/img/brain_COVID.gif))*

This project uses the wiring diagram of the adult male *Drosophila* central nervous system
(166,700 neurons, released in 2026 by FlyEM/HHMI Janelia with Google Research) as the recurrent
network of an echo state network, feeds it market data, and compares it against control
networks that keep some of its statistics and scramble the rest.

The goal isn't to beat the market with a fly. It's a clean answer to a narrower question: *is
biological wiring special as a reservoir, compared to random wiring with matching statistics?*

## Results

SPY daily, out of sample Sep 2007 to Sep 2026. Two scales: a 3,000-neuron circuit grown from the
sensory neurons (`configs/small.yaml`, 5 seeds) and the whole CNS, 166,700 neurons (`configs/full.yaml`,
10 seeds). The gain sweeps use 3 seeds. `scripts/report.py` prints every number below from the
result folders.

**Short version:** the fly brain is no better at markets than random wiring, and as a memory it's
worse. Its wiring is a set of dense modules, and a reservoir with one global gain can't drive all of
them at once. In a small circuit, tuning the gain brings it level with the controls; across the
whole brain it stays 8–11× behind.

**Markets: no edge at either scale, and the wiring doesn't matter.** Every reservoir has an IC around
0.015 (t ≈ 1), a hit rate around 53.5%, below the 55.1% you get by always being long, and a Sharpe
of 0.46 to 0.54 against 0.62 for buy & hold. None of the connectome-vs-control differences is
larger than the noise (the standard error of a 19-year Sharpe is about 0.23). The equity curve's
early lead over buy & hold comes entirely from sidestepping 2008.

**Memory at the standard gain: the fly remembers the least.** Memory capacity counts how many past
steps of a random input a linear readout can recover ([details](#market-free-benchmarks)). The main
number adds a little readout noise, so it only counts memory a real readout could use:

| memory capacity | connectome | degree-preserving | weight shuffle | sign shuffle | Erdős–Rényi |
|---|---|---|---|---|---|
| 3,000 neurons, readout noise | **2.8** | 5.9 | 3.0 | 4.6 | 6.9 |
| whole CNS, readout noise | **2.2** | 15.5 | 2.4 | 2.9 | 7.4 |
| *3,000 neurons, noise-free* | *9.2* | *10.5* | *9.3* | *10.0* | *11.0* |
| *whole CNS, noise-free* | *14.1* | *27.7* | *15.7* | *14.0* | *16.7* |

Noise-free, the fly's whole-brain memory looks respectable, but 84% of it lives in fluctuations
smaller than a thousandth of a neuron's range. With noise, the degree-preserving shuffle remembers
7× more.

**Why: hot spots.** Every wiring is rescaled so its largest eigenvalue is 0.9. In the fly, that
eigenvalue (251, against ~34 for the degree-preserving shuffle) lives on ~200 neurons in the
antennal lobe, the smell center: local interneurons and projection neurons wired into a dense knot
(38% of all possible connections present, 41 synapses per connection against 14 brain-wide, mostly
labeled excitatory). Dividing every weight by ~280 to tame that knot silences almost everything
else: at the standard gain only 3% of the whole-brain readouts move at all, against 90% in the
degree-preserving shuffle, whose top mode is spread over ~15,500 neurons instead. The weight and
sign shuffles keep the fly's connections, so they keep its knots too (3–5% of readouts move).

It's not one knot, either. Remove it and the next hot spot sets the gain (`scripts/hot_spots.py`):

| step | largest eigenvalue | where the hot spot is |
|---|---|---|
| 1 | 251 | antennal lobe (213 neurons) |
| 2 | 206 | optic lobe |
| 3 | 178 | optic lobe |
| 4 | 173 | central brain and descending neurons |
| 5 | 148 | mushroom body (Kenyon cells, output neurons) |
| 6 | 116 | central complex |

Every brain region has its own dense core, and one global volume knob can only suit the loudest.
Normalizing each neuron's total input first (`reservoir.input_normalization: l1`) doesn't fix it:
tiny two-neuron loops then set the gain instead and 95% of the brain stays silent.

**Memory with each wiring at its own best gain (`scripts/gain_sweep.py`).** Forcing one gain on
every wiring is arbitrary, so the sweep tries 13 gains from 0.5 to 20 and keeps each wiring's best
*valid* one: at most 1% of readouts latch onto the distant past or go chaotic, for every seed.

<p align="center">
  <img src="docs/img/gain_sweep_full.png" width="820" alt="Memory capacity against gain for the connectome and four control wirings across the whole CNS, with and without readout noise">
</p>

| best valid memory, readout noise (gain) | connectome | degree-preserving | weight shuffle | sign shuffle | Erdős–Rényi |
|---|---|---|---|---|---|
| 3,000 neurons | 6.7 (3) | 6.8 (12) | 7.4 (3) | 6.7 (4) | 7.2 (1) |
| whole CNS | **2.9** (1.25) | 22.8 (1) | 3.4 (1.25) | 3.7 (1.25) | 30.8 (8) |

- **3,000 neurons: a tie.** Turning the gain up to 3 more than doubles the fly's memory, and every
  wiring lands between 6.7 and 7.4, within the seed-to-seed spread. In this circuit the next hot
  spot is much weaker than the antennal lobe (82 vs 251), so the gain can go up about 3× and wake
  most of the circuit (60% of readouts move) before anything latches. Noise-free, the wirings with
  the fly's connections even come out ahead (12.8 to 14.7 vs 11.4 to 11.9), but that lead lives in
  tiny fluctuations and disappears with noise.
- **Whole brain: the fly stays far behind.** Its best is 2.9 at gain 1.25, and past that its cores
  latch: the next hot spots are close behind the first (206, 178, 173…), so there's no headroom.
  The degree-preserving shuffle reaches 22.8 and Erdős–Rényi 30.8, 8–11× more. Noise-free the
  order is the same (15.5 vs 36.0 and 69.1).
- **It's the topology.** The three wirings that keep the fly's connections (connectome, weight
  shuffle, sign shuffle) all end up near 3; the two that scramble who connects to whom reach 23 to
  31. Moving synapse counts around or changing which neurons are inhibitory doesn't help.
- Caveats on the controls' side: the degree-preserving shuffle is only valid up to gain 1 (it turns
  unstable past that), so its best sits right at the edge. Erdős–Rényi's best, at gain 8, has 90% of
  its readouts barely moving, with the memory carried by the 10% that do, and it turns invalid at
  12. At the standard gain it already has 7.4, 2.5× the fly's best.

<p align="center">
  <img src="docs/img/gain_sweep_small.png" width="820" alt="Memory capacity against gain for the connectome and four control wirings in the 3,000-neuron circuit, with and without readout noise">
</p>

So the honest answer: biological wiring isn't a better reservoir, and at brain scale it's a clearly
worse one. The fly brain is **a set of dense modules, and a reservoir with one global gain can't
drive all of them at once.** In a small circuit with a single dominant module, tuning the gain is
enough to match random wiring; across the whole brain it isn't.

### What the fly does with the market

<p align="center">
  <img src="docs/img/activity_heatmap.png" width="900" alt="Heatmap of how far each neuron group is from its normal activity, month by month, under the SPY price">
</p>

Every crash shows up as a dark band: 2008, the 2010 flash crash, August 2011, 2015, February 2018,
COVID, 2022. In crash months the average neuron sits roughly twice as far from its normal
activity as in calm months, in every part of the circuit. Crashes raise variability rather than
pushing groups up or down, which is why the figures show distance from normal instead of raw
activity. The circuit is mostly the fly's smell pathway: olfactory receptor neurons, the
antennal lobe, the mushroom body (its learning center) and about 400 brain-to-body command neurons.

It also habituates. In the animation, October 2008 glows far brighter than the actual bottom in
March 2009. The inputs are measured against the trailing year: in October, volatility was 4.6
standard deviations above normal, while by March it was still 40% annualized but normal for a year
that had been all crisis. That adaptation comes from the input scaling, not from the fly's wiring.

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

### Market-free benchmarks
- **Memory capacity** (Jaeger 2001): feed white noise, train readouts to reconstruct the input
  from k steps ago, sum the R² over k. It shows how much input history each wiring keeps, which is
  a cleaner test of "is the wiring special" than noisy market returns. Reported twice: noise-free
  (the standard benchmark) and **with readout noise** (0.1% of a neuron's range added before
  fitting, `memory.readout_noise`). The noise-free readout rescales every neuron and can decode
  fluctuations of a millionth, which no physical system could carry; on the whole-brain fly that
  is 84% of its measured memory. The noisy number counts only memory that is actually usable.
- **Readout stability**: run the reservoir twice with inputs that differ only in the distant past.
  *Unstable readouts* end up in different states (they latched or went chaotic); a valid reservoir
  has none. *Active readouts* are the ones that move at all. Both are reported per wiring in every
  summary, together with *top mode spread*: roughly how many neurons carry the largest eigenvalue.

## Setup

Linux (or macOS):

```bash
cd flybrain-reservoir
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python scripts/run_experiment.py --config configs/demo_synthetic.yaml
```

If `python3 -m venv` fails on Debian or Ubuntu, install the venv module first:
`sudo apt install python3-venv`. You can also `source .venv/bin/activate` once per terminal and
then just type `python`.

Windows (PowerShell):

```powershell
cd flybrain-reservoir
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

Calling the venv's Python directly skips `activate`, which PowerShell often blocks.

The tests run on synthetic data in about 20 seconds. The demo runs the whole pipeline offline on a
fake connectome with a planted signal, just to prove everything works.

Notebooks in VS Code: open the `flybrain-reservoir` folder itself and pick the `.venv` kernel. If a
notebook still says `No module named 'flyres'`, run `%pip install -e "<full path to the repo>"` in a
cell and restart the kernel.

## Running it on the real brain

With the venv activated (`source .venv/bin/activate`), or with `.venv/bin/python` in place of
`python`:

```bash
python scripts/download_data.py        # ~1.2 GB from Janelia's public bucket, resumable
python scripts/build_connectome.py     # parse + cache the graph (a few minutes, once)
python scripts/run_experiment.py --config configs/small.yaml
python scripts/brain_activity.py --config configs/small.yaml
python scripts/gain_sweep.py --config configs/small.yaml     # each wiring at its own best gain
python scripts/hot_spots.py --config configs/small.yaml      # where the dominant eigenvalue lives
python scripts/region_gains.py --config configs/small.yaml   # each brain region its own gain
python scripts/report.py                                      # headline numbers from all finished runs
```

Results go to `results/small/`. Start with `summary.md`; the CSVs and `figures/` have the rest.
`brain_activity.py` writes the heatmap and the animation to `results/small/brain/` (pick another
crash with `--window 2020-01-01 2020-08-31 --name COVID`). `gain_sweep.py` writes
`results/small/gain_sweep/` (summary, CSVs, figure); widen the grid with `--gains 0.5 1 2 3 6 12 20`.
`region_gains.py` writes `results/small/region_gains/`; add `--set n_jobs=4` to use more cores. `notebooks/01_connectome_tour.ipynb`
explores the graph and its eigenvalue spectrum, `notebooks/02_results.ipynb` runs and plots an
experiment.

Override anything from the command line:

```bash
python scripts/run_experiment.py --config configs/small.yaml --set reservoir.spectral_radius=0.5 --set name=rho05
python scripts/run_experiment.py --config configs/small.yaml --set reservoir.normalize=frobenius --set reservoir.spectral_radius=0.5 --set name=frob
```

### Scaling up
- Subgraph size is just `subgraph.n_neurons`; everything is sparse, so 3k to 50k is a config change.
- `configs/full.yaml` runs the whole CNS (`method: all`) with 4 parallel jobs. Meant for a
  desktop (32 GB RAM is plenty). With the GPU backend on an RX 7900 XTX the experiment takes about
  11 minutes and the gain sweep about 4; CPU-only is slower.

### GPU (AMD on Linux)

The reservoir runs on the GPU through PyTorch (`reservoir.backend: torch`). AMD cards need
PyTorch's ROCm build, which only exists for Linux. Plain `pip install torch` gets the NVIDIA build
instead, so use the ROCm index:

```bash
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
```

(`rocm7.2` is current as of September 2026; pytorch.org's install selector shows the latest.)
These wheels bundle the ROCm libraries, so you only need the `amdgpu` kernel driver, which
mainstream distros ship, and access to the GPU device files:

```bash
sudo usermod -aG render,video $USER    # then log out and back in
```

Check it works (the second test runs the reservoir on the GPU and compares it with the CPU version):

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/bin/python -m pytest tests/test_reservoir.py -k "torch or gpu" -v
```

ROCm GPUs show up as `cuda` in PyTorch, so nothing else changes. Then:

```bash
python scripts/run_experiment.py --config configs/full.yaml --set reservoir.backend=torch
```

The GPU only pays off for big reservoirs: at 3,000 neurons the CPU is already fast, while at the
full 166k each time step is a sparse multiply with ~6M connections, which is where the GPU wins.

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
4. **Equal spectral radius for every wiring, plus a gain sweep.** The main experiment uses the
   standard echo-state recipe: every wiring rescaled to spectral radius 0.9. That's fair for
   random graphs, but the fly's largest eigenvalue sits on a small hot spot, so it gets turned down
   much harder than the controls. `gain_sweep.py` removes that bias by comparing each wiring at its
   own best valid gain. Two alternatives were tried and rejected: matching total synaptic strength
   (`normalize: frobenius`) pushes the fly's cores past 1 so they latch, and per-neuron input
   normalization hands the gain to tiny two-neuron loops.
5. **Edges need ≥ 5 synapses.** Standard threshold to drop noisy connections (`connectome.min_weight`).
6. **Inputs = the most-connected sensory neurons**, and the subgraph is grown from them by
   repeatedly adding the neurons most strongly connected to it, so the signal can actually
   propagate. `subgraph.input_filter` can pick one modality, e.g. `{class: olfactory}`.

## Reading the numbers

- A model that learns nothing predicts the average return, which is positive, so it turns into
  buy & hold. Compare Sharpe against buy & hold and hit rate against the up-day rate; IC is the
  cleanest skill measure.
- The standard error of an annualized Sharpe over ~19 years is about 0.23. Small differences are noise.
- p-values come from paired tests over seeds. Several comparisons run at once, so expect the odd
  p < 0.05 by chance; the memory gaps (with readout noise) between the fly and the two scrambled
  wirings (p < 0.001) survive that. The fly vs weight and sign shuffle differences are small.

### Limitations
- Rate model, not spiking. A leaky integrate-and-fire version is the obvious next step.
- The sign rule is an approximation (glutamate isn't inhibitory everywhere, neuromodulators are
  treated as excitatory, "unclear" transmitters default to excitatory).
- The hot spots depend on the neurotransmitter labels, which are mostly *predictions* from the
  dataset. The antennal-lobe core is labeled mostly excitatory; if more of it is actually
  inhibitory, it is less explosive than modeled here.
- One global gain is the model's choice, not the fly's: real neurons have their own thresholds,
  adaptation and neuromodulation. Per-region gains would be the natural next experiment.
- The readout noise level (0.1% of a neuron's range) is a judgment call. It changes the small-circuit
  verdict at best gains (fly's connections slightly ahead noise-free, a tie with noise), not the
  whole-brain one.
- Memory capacity with white-noise input is one benchmark. Other tasks (nonlinear transforms,
  chaotic time series) could rank the wirings differently.

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
  benchmarks.py   memory capacity, readout stability (active / unstable readouts)
  diagnostics.py  where the dominant eigenvalue lives, hot-spot cascade
  sweep.py        gain sweep: each wiring at its own best valid gain
  regions.py      per-region gains: anatomical regions, coordinate search for each region's factor
  activity.py     neuron groups, activity relative to normal, soma positions
  experiment.py   runs everything and writes results
  plotting.py     figures and the brain animation
  synthetic.py    fake data for tests and the offline demo
scripts/          download_data.py, build_connectome.py, run_experiment.py, brain_activity.py,
                  gain_sweep.py, hot_spots.py, region_gains.py, report.py
configs/          small.yaml (laptop), full.yaml (whole CNS), demo_synthetic.yaml (offline)
notebooks/        01_connectome_tour.ipynb, 02_results.ipynb
docs/img/         figures used in this README
tests/            pytest suite (no downloads needed)
```

## Data and credits

- Male CNS v1.0 connectome: FlyEM (HHMI Janelia), University of Cambridge, MRC LMB and Google
  Research, <https://male-cns.janelia.org>, licensed CC-BY 4.0. Cite the dataset paper listed on
  the project site if you use it. The figures in `docs/img` are derived from it.
- Sign rule: Shiu et al. 2024, *A Drosophila computational brain model reveals sensorimotor
  processing*, Nature.
- Memory capacity: Jaeger 2001, *Short term memory in echo state networks*.
- Degree-preserving rewiring: Maslov & Sneppen 2002, *Specificity and stability in topology of protein networks*, Science.
- File schema cross-checked against [sstamou03/fly_brain](https://github.com/sstamou03/fly_brain).
