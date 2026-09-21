import numpy as np
import pytest
import scipy.sparse as sp

from flyres.controls import WIRINGS, degree_preserving_rewire, erdos_renyi_like, make_wiring, shuffle_weights


def _degrees(W):
    A = sp.csr_matrix(W, copy=True)
    A.data[:] = 1
    return np.asarray(A.sum(axis=1)).ravel(), np.asarray(A.sum(axis=0)).ravel()  # in, out


def _edge_set(W):
    coo = sp.coo_matrix(W)
    return set(zip(coo.col.tolist(), coo.row.tolist()))


def test_degree_preserving_keeps_every_degree(sub, rng):
    R = degree_preserving_rewire(sub.W, rng)
    for a, b in zip(_degrees(sub.W), _degrees(R)):
        np.testing.assert_array_equal(a, b)
    assert R.nnz == sub.W.nnz  # no merged duplicates
    assert R.diagonal().sum() == 0  # no self-loops
    np.testing.assert_allclose(np.sort(R.data), np.sort(sub.W.data))


def test_degree_preserving_keeps_outgoing_weights_per_neuron(sub, rng):
    R = degree_preserving_rewire(sub.W, rng).tocsc()
    W = sub.W.tocsc()
    for j in range(0, sub.n, 17):  # column j = outgoing synapses of neuron j
        np.testing.assert_allclose(np.sort(W[:, j].data), np.sort(R[:, j].data))


def test_degree_preserving_actually_rewires(sub, rng):
    R = degree_preserving_rewire(sub.W, rng)
    kept = len(_edge_set(sub.W) & _edge_set(R)) / sub.W.nnz
    assert kept < 0.5


def test_erdos_renyi_matches_counts(sub, rng):
    R = erdos_renyi_like(sub.W, rng)
    assert R.nnz == sub.W.nnz
    assert R.diagonal().sum() == 0
    np.testing.assert_allclose(np.sort(R.data), np.sort(sub.W.data))


def test_weight_shuffle_keeps_topology(sub, rng):
    R = shuffle_weights(sub.W, rng)
    assert _edge_set(R) == _edge_set(sub.W)
    np.testing.assert_allclose(np.sort(R.data), np.sort(sub.W.data))
    assert not np.allclose(R.tocsr().data, sub.W.tocsr().data)


def test_sign_shuffle_keeps_ei_ratio(sub, rng):
    W, s = make_wiring("sign_shuffle", sub.W, sub.sign, rng)
    assert W is sub.W
    assert (s < 0).sum() == (sub.sign < 0).sum()


def test_controls_reduce_reciprocity(sub, rng):
    def recip(W):
        A = (sp.csr_matrix(W) != 0).astype(np.int8)
        return A.multiply(A.T).nnz / A.nnz
    base = recip(sub.W)
    assert recip(make_wiring("degree_preserving", sub.W, sub.sign, rng)[0]) < base / 2
    assert recip(make_wiring("erdos_renyi", sub.W, sub.sign, rng)[0]) < base / 2


def test_unknown_wiring():
    with pytest.raises(ValueError):
        make_wiring("nope", sp.eye(3, format="csr"), np.ones(3), np.random.default_rng(0))


def test_all_wirings_run(sub):
    for w in WIRINGS:
        W, s = make_wiring(w, sub.W, sub.sign, np.random.default_rng(1))
        assert W.shape == sub.W.shape and len(s) == sub.n
