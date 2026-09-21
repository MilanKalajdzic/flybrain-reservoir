import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyres.subgraph import Subgraph, graph_stats, match, select_subgraph


def test_grow_selects_requested_size_with_inputs_first(conn, sub):
    assert sub.n == 300
    assert len(np.unique(sub.neurons["global_idx"])) == sub.n
    np.testing.assert_array_equal(sub.input_idx, np.arange(20))
    assert (sub.neurons.loc[sub.input_idx, "superclass"] == "sensory").all()
    # submatrix really is the induced subgraph of the full graph
    g = sub.neurons["global_idx"].to_numpy()
    assert (sub.W != conn.W[g][:, g]).nnz == 0


def test_inputs_are_the_most_connected_candidates(conn, sub):
    out_syn = conn.neurons["out_synapses"].to_numpy()
    cand = np.flatnonzero((conn.neurons["superclass"] == "sensory").to_numpy())
    expected = cand[np.argsort(-out_syn[cand], kind="stable")[:20]]
    np.testing.assert_array_equal(sub.neurons["global_idx"].to_numpy()[:20], expected)


def test_selection_is_deterministic(conn, sub):
    again = select_subgraph(conn, n_neurons=300, n_inputs=20, input_filter={"superclass": "sensory"}, verbose=False)
    np.testing.assert_array_equal(again.neurons["global_idx"], sub.neurons["global_idx"])


@pytest.mark.parametrize("method", ["top_degree", "all"])
def test_other_methods(conn, method):
    s = select_subgraph(conn, n_neurons=200, n_inputs=10, input_filter={"superclass": "sensory"}, method=method,
                        verbose=False)
    assert s.n == (conn.n if method == "all" else 200)
    assert len(s.input_idx) > 0


def test_bad_filter_lists_available_values(conn):
    with pytest.raises(ValueError, match="superclass values"):
        select_subgraph(conn, input_filter={"superclass": "no_such_thing"}, verbose=False)


def test_match_semantics():
    df = pd.DataFrame({"superclass": ["vnc_sensory", "cb_intrinsic", None], "class": ["gustatory", "x", "y"]})
    np.testing.assert_array_equal(match(df, {"superclass": "SENSORY"}), [True, False, False])
    np.testing.assert_array_equal(match(df, {"class": ["gust", "y"]}), [True, False, True])
    np.testing.assert_array_equal(match(df, None), [True, True, True])


def test_save_load_roundtrip(sub, tmp_path):
    sub.save(tmp_path / "sg")
    assert Subgraph.exists(tmp_path / "sg")
    back = Subgraph.load(tmp_path / "sg")
    assert (back.W != sub.W).nnz == 0
    np.testing.assert_array_equal(back.sign, sub.sign)
    np.testing.assert_array_equal(back.input_idx, sub.input_idx)


def test_graph_stats_on_known_graph():
    # 0 <-> 1 reciprocal, 1 -> 2 one-way: reciprocity 2/3, SCC {0,1} = 2/3 of nodes
    W = sp.csr_matrix(([1.0, 1.0, 1.0], ([1, 0, 2], [0, 1, 1])), shape=(3, 3))
    s = graph_stats(W)
    assert s["edges"] == 3
    assert s["reciprocity"] == pytest.approx(2 / 3)
    assert s["largest_scc_frac"] == pytest.approx(2 / 3)
