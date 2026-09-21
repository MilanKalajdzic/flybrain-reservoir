import numpy as np
import pytest
import scipy.sparse as sp

from flyres.reservoir import Reservoir, scale_weights, signed_weights, spectral_radius


def test_signs_applied_per_presynaptic_neuron(sub):
    W = signed_weights(sub.W, sub.sign, "raw").tocoo()
    np.testing.assert_array_equal(np.sign(W.data), sub.sign[W.col])


@pytest.mark.parametrize("transform", ["raw", "log1p", "binary"])
def test_transforms_keep_sparsity(sub, transform):
    W = signed_weights(sub.W, sub.sign, transform)
    assert W.nnz == sub.W.nnz


@pytest.mark.parametrize("n", [200, 600])  # dense path and ARPACK path
def test_spectral_scaling_hits_target(conn, n):
    W = signed_weights(conn.W[:n][:, :n], conn.signs()[:n])
    Ws, raw = scale_weights(W, 0.9)
    exact = np.abs(np.linalg.eigvals(Ws.toarray().astype(np.float64))).max()
    assert exact == pytest.approx(0.9, rel=1e-3)
    assert raw == pytest.approx(np.abs(np.linalg.eigvals(W.toarray().astype(np.float64))).max(), rel=1e-3)


def test_feedforward_graph_is_rejected():
    W = sp.csr_matrix(np.triu(np.ones((5, 5)), k=1))  # nilpotent: every eigenvalue is 0
    assert spectral_radius(W) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match="feedforward"):
        scale_weights(W, 0.9)


def _reservoir(sub, seed=0, **kw):
    W, _ = scale_weights(signed_weights(sub.W, sub.sign), 0.9)
    return Reservoir(W, sub.input_idx, 3, rng=np.random.default_rng(seed), **kw)


def test_run_shapes_bounds_and_determinism(sub):
    U = np.random.default_rng(1).standard_normal((400, 3))
    a = _reservoir(sub).run(U, washout=50)
    b = _reservoir(sub).run(U, washout=50)
    assert a.shape == (350, sub.n)
    assert np.abs(a).max() < 1.0
    np.testing.assert_array_equal(a, b)
    batch = _reservoir(sub).run(np.stack([U, U]), washout=50)
    np.testing.assert_allclose(batch[1], a, atol=1e-6)


def test_echo_state_property(sub):
    """Inputs that differ only in the distant past must end in the same state (fading memory)."""
    rng = np.random.default_rng(2)
    U1 = rng.standard_normal((600, 3))
    U2 = U1.copy()
    U2[:100] = rng.standard_normal((100, 3))
    res = _reservoir(sub, leak_rate=1.0)
    s1, s2 = res.run(U1), res.run(U2)
    assert np.abs(s1[50] - s2[50]).max() > 1e-3  # they did differ
    assert np.abs(s1[-1] - s2[-1]).max() < 1e-5  # and forgot it


def test_input_only_enters_input_neurons(sub):
    zero = sp.csr_matrix(sub.W.shape, dtype=np.float32)
    res = Reservoir(zero, sub.input_idx, 3, bias_scaling=0.0, rng=np.random.default_rng(0))
    S = res.run(np.ones((10, 3)))
    others = np.setdiff1d(np.arange(sub.n), sub.input_idx)
    assert np.all(S[:, others] == 0)
    assert np.any(S[:, sub.input_idx] != 0)


def test_labeled_lines_give_one_feature_per_input_neuron(sub):
    res = _reservoir(sub, input_mode="labeled")
    assert ((res.W_in != 0).sum(axis=1) == 1).all()


def test_torch_backend_matches_numpy(sub):
    pytest.importorskip("torch")
    U = np.random.default_rng(3).standard_normal((200, 3))
    a = _reservoir(sub).run(U)
    b = _reservoir(sub, backend="torch", device="cpu").run(U)
    np.testing.assert_allclose(a, b, atol=1e-4)
