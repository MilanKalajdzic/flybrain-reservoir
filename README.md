# flybrain-reservoir

[![tests](https://github.com/MilanKalajdzic/flybrain-reservoir/actions/workflows/tests.yml/badge.svg)](https://github.com/MilanKalajdzic/flybrain-reservoir/actions/workflows/tests.yml)

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
10 seeds). The gain sweeps and the per-region search use 3 seeds. `scripts/report.py` prints every
number below from the result folders.

**Short version:** the fly brain is no better at markets than random wiring, and as a memory it's
worse. Its wiring is a set of dense knots, one inside almost every brain region, and a reservoir
can't drive them all at once. With one global gain, tuning brings the fly level with the controls in
a small circuit but leaves it 8–11× behind across the whole brain. Giving every region, or every
neuron, its own gain helps random wiring far more than the fly. On volatility, where there is something to forecast, every
reservoir beats the standard HAR benchmark by about 17% at a 5-day horizon, but a linear model with
the same inputs already gets 15% of that, and the wiring moves the result by about 1%.

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
*valid* one: at most 1% of readouts latch onto the distant past or go chaotic, in each of four
latching tests with different inputs, for every seed.

<p align="center">
  <img src="docs/img/gain_sweep_full.png" width="820" alt="Memory capacity against gain for the connectome and four control wirings across the whole CNS, with and without readout noise">
</p>

| best valid memory, readout noise (gain) | connectome | degree-preserving | weight shuffle | sign shuffle | Erdős–Rényi |
|---|---|---|---|---|---|
| 3,000 neurons | 6.7 (3) | 6.8 (12) | 7.4 (3) | 4.1 (1) | 7.2 (1) |
| whole CNS | **2.9** (1.25) | 22.8 (1) | 2.9 (1.1) | 3.1 (1) | 30.8 (8) |

- **3,000 neurons: a tie.** Turning the gain up to 3 more than doubles the fly's memory, and four of
  the five wirings land between 6.7 and 7.4, within the seed-to-seed spread (the sign shuffle
  latches at any gain above 1 and stays at 4.1). In this circuit the next hot spot is much weaker
  than the antennal lobe (82 vs 251), so the gain can go up about 3× and wake most of the circuit
  (59% of readouts move) before anything latches. Noise-free, the connectome and the weight shuffle
  even come out ahead (13.9 and 14.7 vs 11.4 to 11.9), but that lead lives in tiny fluctuations and
  disappears with noise.
- **Whole brain: the fly stays far behind.** Its best is 2.9 at gain 1.25, and past that its cores
  latch: the next hot spots are close behind the first (206, 178, 173…), so there's no headroom.
  The degree-preserving shuffle reaches 22.8 and Erdős–Rényi 30.8, 8–11× more. Noise-free the
  order is the same (15.5 vs 36.0 and 69.1).
- **It's the topology.** The three wirings that keep the fly's connections (connectome, weight
  shuffle, sign shuffle) all end up near 3 (2.9 to 3.1); the two that scramble who connects to whom
  reach 23 to 31. Moving synapse counts around or changing which neurons are inhibitory doesn't help.
- Caveats on the controls' side: the degree-preserving shuffle is only valid up to gain 1 (it turns
  unstable past that), so its best sits right at the edge. Erdős–Rényi's best, at gain 8, has 90% of
  its readouts barely moving, with the memory carried by the 10% that do, and it turns invalid at
  12. At the standard gain it already has 7.4, 2.5× the fly's best.

<p align="center">
  <img src="docs/img/gain_sweep_small.png" width="820" alt="Memory capacity against gain for the connectome and four control wirings in the 3,000-neuron circuit, with and without readout noise">
</p>

**Each brain region its own gain (`scripts/region_gains.py`).** A real brain isn't stuck with one
volume knob: neuromodulators and local inhibition tune each region. So the model gets the same
freedom. The neurons are split into the anatomical regions from the activity figures (visual system,
central brain, nerve cord, mushroom body, antennal lobe, …: 9 in the small circuit, 11 in the whole
CNS), each region's incoming synapses get their own factor, and a search moves one factor at a time
(×2, then ×1.41) while memory improves and every latching test passes. It starts from the two best
peaks of each wiring's single-gain curve and picks factors on one set of white-noise inputs; every
number below comes from fresh inputs it never saw. Every wiring gets the same regions and the same
search.

<p align="center">
  <img src="docs/img/region_gains_full.png" width="900" alt="Memory with one gain versus one gain per brain region for the connectome and four control wirings across the whole CNS, and the factor the search chose for each region">
</p>

| memory with readout noise, fresh inputs | connectome | degree-preserving | weight shuffle | sign shuffle | Erdős–Rényi |
|---|---|---|---|---|---|
| 3,000 neurons, best single gain | 6.8 | 6.9 | 7.4 | 5.9 | 7.2 |
| 3,000 neurons, per-region gains | **11.5** | 42.6 | 12.4 | 12.2 | 36.0 |
| whole CNS, best single gain | 3.4 | 22.9 | 3.3 | 4.7 | 30.9 |
| whole CNS, per-region gains | **6.2** | 47.5 | 6.0 | 7.2 | 71.2 |

(Single gains here are each seed's own best, measured on the fresh inputs, so they differ a little
from the sweep table.)

- **The fly gains about 75%, random wiring 2–6×.** In the small circuit the tie turns into a 3–4×
  gap (11.5 vs 42.6 and 36.0). Across the whole brain the gap stays about where it was, 8–12×
  (6.2 vs 47.5 and 71.2).
- **Same split as before.** The three wirings with the fly's connections end up together (around 12
  in the small circuit, 6 to 7 in the whole brain), the two scrambled ones far above.
- **Why it doesn't rescue the fly: the knots sit inside the regions.** Every fly region is at least
  4 times louder on its own than the same region in the degree-preserving shuffle (largest eigenvalue
  of the region's own connections: antennal lobe 252 vs 7.5, visual system 206 vs 33, rest of the
  central brain 171 vs 15, central complex 116 vs 4). A region's factor scales its knot together
  with everything around it, so after the search the fly's dominant mode still sits on ~60 neurons,
  against ~10,000 in the degree-preserving shuffle and ~14,000 in Erdős–Rényi.
- Erdős–Rényi's per-region result comes from starting at gain 1, not from its saturated best single
  gain of 8, and wakes up the whole brain (98% of readouts move, from 10%). Every per-region setting
  passes all latching tests on the fresh inputs, except one sign-shuffle seed in the small circuit.

<p align="center">
  <img src="docs/img/region_gains_small.png" width="900" alt="Memory with one gain versus one gain per brain region in the 3,000-neuron circuit, and the factor the search chose for each region">
</p>

**Each neuron its own gain (`scripts/homeostasis.py`).** Real neurons don't wait for a search:
synaptic scaling multiplies all of a neuron's incoming synapses up when it's too quiet and down when
it's too busy. The model gets the same rule. Drive the reservoir with white noise, measure how big
each neuron's recurrent input is, move its gain halfway toward a target, and repeat for 30 rounds.
Knots get turned down and silent stretches of the brain turned up, with no knowledge of anatomy.
Four targets; each wiring keeps its best valid one, judged on fresh inputs as before.

<p align="center">
  <img src="docs/img/homeostasis_full.png" width="900" alt="Memory with the best single gain versus homeostatic per-neuron gains for the connectome and four control wirings across the whole CNS, and the mean gain the rule gave each region">
</p>

| memory with readout noise, fresh inputs | connectome | degree-preserving | weight shuffle | sign shuffle | Erdős–Rényi |
|---|---|---|---|---|---|
| 3,000 neurons, best single gain | 6.8 | 6.9 | 7.4 | 5.9 | 7.2 |
| 3,000 neurons, homeostatic | 15.3 (1 of 3 seeds valid) | 17.3 | 17.9 | 13.1 (2 of 3) | 15.5 |
| whole CNS, best single gain | 3.4 | 22.9 | 3.3 | 4.7 | 30.9 |
| whole CNS, homeostatic | **5.4** (2 of 3) | 55.2 | latches (0 of 3) | 9.6 | 39.5 |

(Mean over 3 seeds. In brackets: how many seeds end up valid on fresh inputs, when not all of them do.)

- **Per-neuron gains don't rescue the fly either.** Across the whole brain it goes from 3.4 to 5.4,
  while the degree-preserving shuffle goes from 23 to 55: the gap widens to about 10×.
- **The fly mostly latches.** Of its 12 tries (4 targets × 3 seeds), the fly ends up valid once in
  the small circuit and twice in the whole brain; the degree-preserving shuffle 9 times at both
  scales. When the fly latches, 80–90% of the circuit locks up at once (checked on the small
  circuit), so it's a global state, not one knot.
- **Topology again, at brain scale.** In the small circuit the weight shuffle (the fly's connections,
  synapse counts shuffled) does fine, which suggested the fly's actual synapse counts were the
  problem. The whole brain doesn't back that up: there the weight shuffle never ends up valid (0 of
  12 tries) and the sign shuffle gains little (4.7 to 9.6). The three wirings with the fly's
  connections struggle; the two that scramble them don't.
- **Caveats.** The rule doesn't always settle. In these mostly excitatory networks a neuron's input
  can have no level near the target: a bit more gain tips its neighborhood into a self-sustained
  active state, a bit less drops it back. So after 30 rounds usually only a minority of neurons are
  within ×2 of the target; at the highest target it does settle, and then almost everything latches
  (1 valid of 30 tries). Two other rules collapsed first (documented in `homeostasis.py`). Validity
  this close to the edge is also fragile: the same small-circuit run gave the fly 0 valid tries on
  one machine and 1 on another. Erdős–Rényi's whole-brain number varies a lot between seeds
  (39.5 ± 37.6).

So the honest answer: biological wiring isn't a better reservoir, and at brain scale it's a clearly
worse one. The fly brain is **a set of dense knots, one inside almost every region, and a reservoir
can't drive them all at once**, whether it gets one global gain, one per region or one per neuron.
The only place the fly keeps up is a small circuit with one global gain, and finer gains take that
away too.

### Volatility: a question with an answer

Returns are close to unpredictable; volatility isn't. `scripts/vol_forecast.py` puts a second readout
on the same reservoirs (same wiring, inputs and seeds as above) and forecasts the log realized
variance over the next 5 and 22 trading days, walk-forward on the same test days. The benchmark is
HAR (Corsi 2009), the standard model for this. Each reservoir's readout also sees the HAR features
and its own 5 inputs directly, so it contains the linear model *HAR + inputs* as a special case:
reservoir minus HAR + inputs is what the wiring adds.

<p align="center">
  <img src="docs/img/vol_models_full.png" width="820" alt="Out-of-sample error of each model's volatility forecasts relative to HAR, whole CNS, 5- and 22-day horizons">
</p>

| whole CNS: log MSE vs HAR (R², log) | next 5 days | next 22 days |
|---|---|---|
| connectome | −16.7% (0.533) | −5.0% (0.442) |
| degree-preserving | −17.4% (0.537) | −6.3% (0.449) |
| weight shuffle | −17.0% (0.535) | −5.2% (0.443) |
| sign shuffle | −16.7% (0.533) | −5.2% (0.443) |
| Erdős–Rényi | −16.9% (0.534) | −5.3% (0.443) |
| HAR + inputs (linear) | −15.1% (0.524) | **−7.1% (0.454)** |
| HAR | 0% (0.439) | 0% (0.412) |
| EWMA | +1.3% (0.432) | −3.7% (0.434) |

- **Volatility is forecastable, and the biggest gain is linear.** HAR explains 44% of the variation
  in next week's log variance. Adding the reservoir's five inputs (recent returns and the vol-spike
  ratio) in a plain linear model cuts the error by another 15%, the largest improvement anywhere in
  this project, and no reservoir is involved.
- **The reservoir adds a little, at short horizons only.** Every wiring improves on HAR + inputs by
  2–3% at 5 days (Diebold–Mariano p ≤ 0.013 for all five) and not at all at 22 days, where the
  linear model is best.
- **The wiring barely matters.** The wirings are within 1.5% of each other, against 17% over HAR.
  At the standard gain the fly is 0.6% behind the degree-preserving shuffle at 5 days (consistent
  across seeds, p ≈ 0.01, but tiny); with every wiring at its best gain it lands between the
  controls.
- **Memory helps only a little.** At the standard gain, reservoirs with more memory forecast slightly
  better at 5 days (Spearman ρ = −0.32 over 50 reservoirs, p = 0.02). With every wiring at its best
  gain, where memory ranges from 3 to 31, there's no relationship (15 reservoirs). The HAR features
  already carry a month of history, which leaves little for the reservoir's own memory to add.
- The 3,000-neuron circuit tells the same story (reservoirs −15.5% to −16.5% vs HAR at 5 days,
  HAR + inputs −14.4%), with no memory link. One failure worth knowing: the degree-preserving
  shuffle at gain 12 is saturated (10% of neurons move) and forecast 0.6% annualized volatility on
  23 October 2008, one bad day that dominates that seed's QLIKE.

Why log MSE and not QLIKE, the usual volatility loss: every log model here (HAR, HAR + inputs, the
reservoirs) is fitted for log MSE, so it compares them like for like. Log models target the mean of
log variance, not of variance, which can cost them on QLIKE: on simulated GARCH data, HAR fitted on
variance beats HAR fitted on logs by 10–20% QLIKE and ties the reservoirs (on SPY the two HARs tie).
The summaries report QLIKE too, with that HAR-on-variance as its reference. Before trusting any of this, the forecasts were
checked for lookahead: changing all prices after a date leaves every earlier forecast identical.

<p align="center">
  <img src="docs/img/vol_forecast_full.png" width="900" alt="Realized 5-day volatility and the forecasts made for it by HAR, the connectome and the degree-preserving shuffle, around the 2008 crash and COVID">
</p>

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
  has none. Four such tests with different inputs, and the worst one counts
  (`memory.stability_tests`): near the edge a reservoir can latch for some inputs and not others,
  and a single test passes by luck surprisingly often. *Active readouts* are the ones that move at all. Both are reported per wiring in every
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
python scripts/homeostasis.py --config configs/small.yaml    # each neuron its own gain (homeostatic rule)
python scripts/vol_forecast.py --config configs/small.yaml   # volatility forecasts, same reservoirs
python scripts/report.py                                      # headline numbers from all finished runs
```

Results go to `results/small/`. Start with `summary.md`; the CSVs and `figures/` have the rest.
`brain_activity.py` writes the heatmap and the animation to `results/small/brain/` (pick another
crash with `--window 2020-01-01 2020-08-31 --name COVID`). `gain_sweep.py` writes
`results/small/gain_sweep/` (summary, CSVs, figure); widen the grid with `--gains 0.5 1 2 3 6 12 20`.
`region_gains.py` writes `results/small/region_gains/`; add `--set n_jobs=4` to use more cores.
`vol_forecast.py` writes `results/small/volatility/` (run it after `run_experiment.py` so it can relate
each reservoir's memory to its forecasts); `--best-gains` runs every wiring at its best gain from
the gain sweep instead. `notebooks/01_connectome_tour.ipynb`
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
  12 minutes, the gain sweep about 6, the per-region search about 35 and the homeostatic rule about
  20; CPU-only is slower.

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
4. **Equal spectral radius for every wiring, plus a gain sweep and per-region gains.** The main experiment uses the
   standard echo-state recipe: every wiring rescaled to spectral radius 0.9. That's fair for
   random graphs, but the fly's largest eigenvalue sits on a small hot spot, so it gets turned down
   much harder than the controls. `gain_sweep.py` removes that bias by comparing each wiring at its
   own best valid gain. Two alternatives were tried and rejected: matching total synaptic strength
   (`normalize: frobenius`) pushes the fly's cores past 1 so they latch, and per-neuron input
   normalization hands the gain to tiny two-neuron loops. `region_gains.py` then gives every brain
   region its own gain on top, and `homeostasis.py` every neuron.
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
- Gains are tried globally, per brain region and per neuron. The region search is a local
  coordinate search and the per-neuron rule doesn't always settle, both over 3 seeds, so neither is
  a global optimum. Real brains also tune excitability with neuromodulators that depend on what the
  animal is doing, which nothing here models.
- The readout noise level (0.1% of a neuron's range) is a judgment call. It changes the small-circuit
  verdict at best gains (fly's connections slightly ahead noise-free, a tie with noise), not the
  whole-brain one.
- Realized variance comes from daily squared returns, the only thing daily closes allow. That proxy
  is noisy; with intraday (5-minute) data every volatility model would be more precise and HAR harder
  to beat.
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
  homeostasis.py  per-neuron gains from synaptic scaling toward a target input size
  volatility.py   realized-volatility forecasts: HAR benchmarks, reservoir readouts, DM tests
  activity.py     neuron groups, activity relative to normal, soma positions
  experiment.py   runs everything and writes results
  plotting.py     figures and the brain animation
  synthetic.py    fake data for tests and the offline demo
scripts/          download_data.py, build_connectome.py, run_experiment.py, brain_activity.py,
                  gain_sweep.py, hot_spots.py, region_gains.py, homeostasis.py, vol_forecast.py,
                  report.py
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
