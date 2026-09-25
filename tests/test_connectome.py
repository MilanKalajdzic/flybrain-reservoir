import numpy as np
import pytest

from flyres.connectome import FILES, build_cache, canonical_nt, known_nt, load_connectome, signs_from_nt
from flyres.synthetic import synthetic_connectome, write_mock_flywire_files, write_mock_raw_files


@pytest.mark.parametrize("raw, expected", [
    ("acetylcholine", "acetylcholine"), ("ACh", "acetylcholine"), ("gaba", "gaba"), ("GABA ", "gaba"),
    ("glutamate", "glutamate"), ("gluatmate", "glutamate"), ("dopamine", "dopamine"), ("dopmaine", "dopamine"),
    ("5HT", "serotonin"), ("octopamine", "octopamine"), ("histamine", "histamine"),
    ("unclear", None), ("unknown", None), (None, None), (float("nan"), None),
])
def test_canonical_nt(raw, expected):
    assert canonical_nt(raw) == expected


def test_signs_follow_dale_rule():
    labels = ["gaba", "acetylcholine", "glutamate", "histamine", "dopamine", "unknown"]
    np.testing.assert_array_equal(signs_from_nt(labels), [-1, 1, -1, -1, 1, 1])


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    conn = synthetic_connectome(n=500, mean_degree=15, seed=3)
    d = tmp_path_factory.mktemp("raw")
    write_mock_raw_files(conn, d, n_fragments=100, batch_rows=500)
    return conn, d


def test_build_cache_reproduces_graph(raw_dir, tmp_path):
    conn, d = raw_dir
    built = build_cache(d, tmp_path, min_weight=1, verbose=False)
    # fragments and autapses dropped, orientation W[post, pre] preserved, NT fallbacks resolved
    assert built.n == conn.n
    assert (built.W != conn.W).nnz == 0
    assert built.W.diagonal().sum() == 0
    np.testing.assert_array_equal(built.neurons["nt"].to_numpy(), conn.neurons["nt"].to_numpy())
    np.testing.assert_array_equal(built.neurons["in_degree"], np.diff(conn.W.indptr))


def test_min_weight_threshold(raw_dir, tmp_path):
    conn, d = raw_dir
    built = build_cache(d, tmp_path, min_weight=3, verbose=False)
    ref = conn.W.copy()
    ref.data[ref.data < 3] = 0
    ref.eliminate_zeros()
    assert built.W.data.min() >= 3
    assert (built.W != ref).nnz == 0


def test_load_uses_cache(raw_dir, tmp_path):
    conn, d = raw_dir
    first = load_connectome(d, tmp_path, min_weight=1, verbose=False)
    (d / FILES["weights"]).rename(d / "moved.feather")  # a cache hit must not need the raw file
    try:
        second = load_connectome(d, tmp_path, min_weight=1, verbose=False)
    finally:
        (d / "moved.feather").rename(d / FILES["weights"])
    assert (first.W != second.W).nnz == 0


def test_missing_raw_files_explain_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        build_cache(tmp_path / "nothing", tmp_path / "cache", verbose=False)


@pytest.mark.parametrize("raw, expected", [
    ("acetylcholine", "acetylcholine"), ("histamine; acetylcholine, histamine", "histamine"),
    ("acetylcholine; sNPF; acetylcholine, sNPF", "acetylcholine"), ("gaba-negative", None),
    ("acetylcholine-negative, glutamate-negative, gaba-negative", None), ("glutamate, gaba; glutamate", "glutamate"),
    ("gaba, nitric oxide", "gaba"), ("allatostatin-a", None), (None, None), (float("nan"), None),
])
def test_known_nt(raw, expected):
    assert known_nt(raw) == expected


@pytest.fixture(scope="module")
def flywire_dir(tmp_path_factory):
    conn = synthetic_connectome(n=400, mean_degree=15, seed=5)
    d = tmp_path_factory.mktemp("flywire")
    write_mock_flywire_files(conn, d)
    return conn, d


def test_flywire_build_reproduces_graph(flywire_dir, tmp_path):
    conn, d = flywire_dir
    built = build_cache(d, tmp_path, min_weight=1, verbose=False, source="flywire")
    # unknown neurons and autapses dropped, W[post, pre] kept, known_nt beats a wrong prediction
    assert built.n == conn.n and (built.W != conn.W).nnz == 0 and built.W.diagonal().sum() == 0
    np.testing.assert_array_equal(built.neurons["nt"].to_numpy(), conn.neurons["nt"].to_numpy())
    assert list(built.neurons["bodyId"]) == sorted(built.neurons["bodyId"])
    assert {"central", "descending", "sensory"} == set(built.neurons["superclass"])
    assert (tmp_path / "flywire_783_w1.npz").exists()  # its own cache, next to the male CNS one
    again = load_connectome(d, tmp_path, min_weight=1, verbose=False, source="flywire")
    assert (again.W != built.W).nnz == 0


def test_flywire_vocabulary_maps_to_the_same_groups():
    from flyres.activity import neuron_group

    pairs = [("optic", "ME>LO", "Visual"), ("visual_projection", None, "Visual"),
             ("central", None, "Other central brain"), ("central", "ALPN", "Antennal lobe (smell processing)"),
             ("central", "Kenyon_Cell", "Mushroom body (learning)"), ("central", "CX", "Central complex (navigation)"),
             ("descending", None, "Descending (brain to body)"), ("ascending", None, "Ascending (body to brain)"),
             ("sensory", "olfactory", "Smell sensors"), ("sensory_ascending", "mechanosensory",
                                                         "Touch & body-position sensors")]
    for sup, cls, group in pairs:
        assert neuron_group(sup, cls) == group


def test_flywire_subgraph_and_soma_positions(flywire_dir, tmp_path):
    from flyres.activity import all_soma_positions, soma_positions
    from flyres.config import load_config
    from flyres.experiment import prepare_subgraph, subgraph_cache_path

    conn, d = flywire_dir
    cfg = load_config(None, [f"data.raw_dir={d.as_posix()}", f"data.cache_dir={tmp_path.as_posix()}",
                             "connectome.source=flywire", "connectome.min_weight=1", "subgraph.n_neurons=150",
                             "subgraph.n_inputs=10"])
    sub = prepare_subgraph(cfg, verbose=False)
    assert sub.n == 150 and subgraph_cache_path(cfg).name.startswith("flywire_")
    xyz = soma_positions(d, sub.neurons["bodyId"], source="flywire")
    sensory = (sub.neurons["superclass"] == "sensory").to_numpy()
    assert np.isnan(xyz[sensory]).all() and np.isfinite(xyz[~sensory]).all()
    assert len(all_soma_positions(d, source="flywire")) == (conn.neurons["superclass"] != "sensory").sum()


def test_missing_flywire_files_say_which_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="--source flywire"):
        build_cache(tmp_path / "nothing", tmp_path / "cache", verbose=False, source="flywire")
