import numpy as np
import pytest

from flyres.benchmarks import memory_capacity
from flyres.config import load_config
from flyres.experiment import run_experiment
from flyres.reservoir import Reservoir, scale_weights, signed_weights


def test_config_overrides_and_typos(tmp_path):
    cfg = load_config(None, ["subgraph.n_neurons=500", "seeds=[1, 2]", "reservoir.leak_rate=0.3"])
    assert cfg.subgraph.n_neurons == 500 and cfg.seeds == [1, 2] and cfg.reservoir.leak_rate == 0.3
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(None, ["subgraph.n_neuron=500"])
    p = tmp_path / "c.yaml"
    p.write_text("name: x\nreservoir: {spectral_radius: 0.5}\n")
    assert load_config(p).reservoir.spectral_radius == 0.5


def test_memory_capacity_is_sane(sub):
    W, _ = scale_weights(signed_weights(sub.W, sub.sign), 0.9)
    res = Reservoir(W, sub.input_idx, 1, rng=np.random.default_rng(0))
    total, curve = memory_capacity(res, sub.readout_pool[:100], n_steps=1500, max_delay=30,
                                   rng=np.random.default_rng(1))
    assert 0 < total <= 30
    assert curve[0] > curve[-1]  # recent inputs are remembered better than old ones


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_end_to_end_synthetic(tmp_path, n_jobs):
    cfg = load_config(None, [
        "name=smoke", f"output_dir={tmp_path.as_posix()}", f"n_jobs={n_jobs}", "seeds=[0, 1]",
        "connectome.source=synthetic", "connectome.synthetic_n=600",
        "subgraph.n_neurons=200", "subgraph.n_inputs=15", "subgraph.n_readout=40",
        "market.source=synthetic", "market.synthetic_days=1800", "market.synthetic_predictability=0.3",
        "eval.train_min=400", "eval.refit_every=200", "eval.n_boot=200",
        "memory.n_steps=800", "memory.max_delay=20",
    ])
    res = run_experiment(cfg, verbose=False)
    models = set(res.metrics["model"])
    assert models == {"connectome", "degree_preserving", "weight_shuffle", "sign_shuffle", "erdos_renyi",
                      "linear_features", "ar", "buy_hold"}
    out = tmp_path / "smoke"
    for f in ["summary.md", "metrics.csv", "comparisons.csv", "memory_capacity.csv", "graph_stats.csv",
              "predictions.parquet", "config_used.yaml", "figures/metrics_SYNTH.png",
              "figures/equity_SYNTH.png", "figures/memory_capacity.png"]:
        assert (out / f).exists(), f
    assert "Connectome minus each alternative" in res.summary
    # every model is evaluated on exactly the same days
    assert res.metrics["n_test"].nunique() == 1
    # the planted signal is found
    assert res.metrics.loc[res.metrics["model"] == "connectome", "ic"].mean() > 0.05
