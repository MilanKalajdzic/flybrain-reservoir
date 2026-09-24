"""Experiment configuration: dataclasses with defaults, loaded from YAML.

Unknown keys raise an error, so a typo in a config file fails loudly instead of silently
running with the default.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    cache_dir: str = "data/cache"


@dataclass
class ConnectomeConfig:
    source: str = "malecns"          # malecns | synthetic (fake graph, for offline runs)
    min_weight: int = 5              # drop connections with fewer synapses (usual noise threshold)
    drop_autapses: bool = True
    inhibitory: list = field(default_factory=lambda: ["gaba", "glutamate", "histamine"])
    synthetic_n: int = 3000


@dataclass
class SubgraphConfig:
    method: str = "grow"             # grow | top_degree | all
    n_neurons: int = 3000
    n_inputs: int = 100
    input_filter: dict = field(default_factory=lambda: {"superclass": "sensory"})
    direction: str = "both"          # both | downstream
    n_readout: int = 300             # recorded neurons the readout listens to


@dataclass
class ReservoirConfig:
    spectral_radius: float = 0.9
    normalize: str = "spectral"      # spectral | frobenius
    weight_transform: str = "log1p"  # raw | log1p | binary
    input_normalization: str = "none"  # none | l1 (each neuron's incoming weights sum to 1 in absolute value)
    leak_rate: float = 1.0           # 1 = no per-neuron smoothing: all memory has to come from the wiring
    input_scaling: float = 0.5
    bias_scaling: float = 0.1
    input_mode: str = "dense"        # dense | labeled
    readout_include_input: bool = True  # standard ESN readout on [u(t), x(t)]
    input_penalty: float = 0.001     # ridge penalty on the raw inputs relative to the states (small = the
                                     # linear model sits inside the reservoir model, states are a correction)
    washout: int = 250
    backend: str = "numpy"           # numpy | torch
    device: str | None = None


@dataclass
class MarketConfig:
    source: str = "yfinance"         # yfinance | csv | synthetic
    tickers: list = field(default_factory=lambda: ["SPY"])
    start: str = "2003-01-01"
    end: str | None = None
    csv_path: str | None = None
    features: list = field(default_factory=lambda: ["r1", "r5", "r20", "vol20", "vol_ratio"])
    horizon: int = 1
    z_window: int = 252
    ar_lags: int = 10
    synthetic_days: int = 5000
    synthetic_predictability: float = 0.0


@dataclass
class EvalConfig:
    train_min: int = 756             # ~3 years before the first prediction
    refit_every: int = 126           # ~6 months
    window: int | None = None        # None = expanding window
    position: str = "sign"           # sign | linear
    cost_bps: float = 1.0            # per unit of turnover
    val_frac: float = 0.2
    bootstrap_block: int = 20
    n_boot: int = 2000


@dataclass
class MemoryConfig:
    enabled: bool = True
    n_steps: int = 3000
    max_delay: int = 100
    washout: int = 200
    readout_noise: float = 0.001     # also report memory with this much noise on each readout (states are in
                                     # [-1, 1], so 0.001 = 0.1% of the range); 0 = noise-free only
    stability_tests: int = 4         # independent latching tests per reservoir; valid only if it passes all


@dataclass
class VolatilityConfig:
    horizons: list = field(default_factory=lambda: [5, 22])  # forecast realized variance over the next h days
    ewma_lambda: float = 0.94        # RiskMetrics decay for the EWMA baseline


@dataclass
class ExperimentConfig:
    name: str = "small"
    seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    wirings: list = field(default_factory=lambda: ["connectome", "degree_preserving", "weight_shuffle",
                                                   "sign_shuffle", "erdos_renyi"])
    swaps_per_edge: float = 10.0
    n_jobs: int = 1
    output_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    connectome: ConnectomeConfig = field(default_factory=ConnectomeConfig)
    subgraph: SubgraphConfig = field(default_factory=SubgraphConfig)
    reservoir: ReservoirConfig = field(default_factory=ReservoirConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)

    def to_dict(self) -> dict:
        return asdict(self)


_SECTIONS = {"data": DataConfig, "connectome": ConnectomeConfig, "subgraph": SubgraphConfig,
             "reservoir": ReservoirConfig, "market": MarketConfig, "eval": EvalConfig, "memory": MemoryConfig,
             "volatility": VolatilityConfig}


def _build(cls, values: dict):
    values = dict(values or {})
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown config keys in {cls.__name__}: {unknown}")
    kwargs = {}
    for name, value in values.items():
        section = _SECTIONS.get(name) if cls is ExperimentConfig else None
        kwargs[name] = _build(section, value) if section else value
    return cls(**kwargs)


def _set_dotted(d: dict, dotted: str, value) -> None:
    *path, last = dotted.split(".")
    for key in path:
        d = d.setdefault(key, {})
    d[last] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load a YAML config; `overrides` like ["subgraph.n_neurons=5000", "seeds=[0,1]"] win over the file."""
    raw = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        _set_dotted(raw, key.strip(), yaml.safe_load(value))
    return _build(ExperimentConfig, raw)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
