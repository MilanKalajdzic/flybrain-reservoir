import numpy as np
import pytest

from flyres.connectome import FILES, build_cache, canonical_nt, load_connectome, signs_from_nt
from flyres.synthetic import synthetic_connectome, write_mock_raw_files


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
