"""Leaky-tanh echo state network whose recurrent matrix is a (real or control) connectome.

    x[t+1] = (1 - a) * x[t] + a * tanh(W @ x[t] + W_in @ u[t] + b)

W is the signed synapse matrix rescaled to a target spectral radius, W_in only feeds the input
(sensory) neurons, and only a subset of neurons is recorded for the readout. Nothing in here is
trained: the reservoir is fixed and only the linear readout (readout.py) learns.

Backends: "numpy" (scipy.sparse, CPU) or "torch" (CPU or GPU, `pip install torch`).
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

TRANSFORMS = ("raw", "log1p", "binary")
NORMALIZATIONS = ("spectral", "frobenius")


def signed_weights(W_counts: sp.spmatrix, sign: np.ndarray, transform: str = "log1p") -> sp.csr_matrix:
    """Synapse counts -> signed weights. Column j (presynaptic neuron) gets sign[j] (Dale's law).

    raw: w = count. log1p: w = log(1 + count), tames the heavy tail (a few connections have
    hundreds of synapses). binary: w = 1 for every edge (topology only).
    """
    W = sp.csr_matrix(W_counts, dtype=np.float32, copy=True)
    if transform == "log1p":
        W.data = np.log1p(W.data)
    elif transform == "binary":
        W.data = np.ones_like(W.data)
    elif transform != "raw":
        raise ValueError(f"transform must be one of {TRANSFORMS}")
    return (W @ sp.diags(np.asarray(sign, dtype=np.float32))).tocsr()


def spectral_radius(W: sp.spmatrix, seed: int = 0) -> float:
    """Largest |eigenvalue|. Dense for small matrices, ARPACK otherwise, power iteration as fallback."""
    n = W.shape[0]
    if W.nnz == 0:
        return 0.0
    if n <= 400:
        return float(np.abs(np.linalg.eigvals(W.toarray().astype(np.float64))).max())
    A = sp.csr_matrix(W, dtype=np.float64)
    v0 = np.random.default_rng(seed).standard_normal(n)
    try:
        vals = spla.eigs(A, k=1, which="LM", return_eigenvectors=False, tol=1e-4,
                         maxiter=max(5000, 10 * n), v0=v0)
        return float(np.abs(vals).max())
    except spla.ArpackNoConvergence as e:
        if len(e.eigenvalues):
            return float(np.abs(e.eigenvalues).max())
    return _power_radius(A, v0)


def _power_radius(A: sp.spmatrix, v0: np.ndarray, iters: int = 1000) -> float:
    """Geometric-mean growth rate of ||A^k v||; converges to rho(A) even for complex leading pairs."""
    x = v0 / np.linalg.norm(v0)
    logs = []
    for _ in range(iters):
        x = A @ x
        norm = np.linalg.norm(x)
        if norm == 0:
            return 0.0
        logs.append(np.log(norm))
        x /= norm
    return float(np.exp(np.mean(logs[iters // 2:])))


def scale_weights(W: sp.csr_matrix, target: float = 0.9, method: str = "spectral") -> tuple[sp.csr_matrix, float]:
    """Rescale so every wiring runs at the same gain. Returns (scaled W, the raw value that was matched).

    spectral:  rho(W) = target. Standard echo-state recipe.
    frobenius: ||W||_F / sqrt(n) = target. For iid random matrices that's also the spectral radius
               (circular law), but for heavy-tailed graphs it matches the bulk of the spectrum
               instead of the single biggest outlier eigenvalue. Useful as a robustness check.
    """
    if method == "spectral":
        raw = spectral_radius(W)
    elif method == "frobenius":
        raw = float(np.sqrt((W.data.astype(np.float64) ** 2).sum() / W.shape[0]))
    else:
        raise ValueError(f"method must be one of {NORMALIZATIONS}")
    if raw < 1e-12:
        raise ValueError("wiring has no recurrent gain (feedforward graph?); pick a bigger/denser subgraph")
    return (W * np.float32(target / raw)).tocsr(), raw


class Reservoir:
    """Fixed random-input, fixed-wiring echo state network."""

    def __init__(self, W: sp.spmatrix, input_idx, n_features: int, *, leak_rate: float = 0.3,
                 input_scaling: float = 0.5, bias_scaling: float = 0.1, input_mode: str = "dense",
                 rng: np.random.Generator | None = None, backend: str = "numpy", device: str | None = None):
        rng = rng if rng is not None else np.random.default_rng()
        self.W = sp.csr_matrix(W, dtype=np.float32)
        self.n = self.W.shape[0]
        self.input_idx = np.asarray(input_idx, dtype=np.int64)
        k = len(self.input_idx)
        if input_mode == "dense":  # every input neuron sees a random mix of all features
            W_in = rng.uniform(-1.0, 1.0, size=(k, n_features))
        elif input_mode == "labeled":  # "labeled lines": each input neuron is tuned to one feature (ON or OFF)
            W_in = np.zeros((k, n_features))
            W_in[np.arange(k), np.arange(k) % n_features] = rng.uniform(0.5, 1.0, k) * rng.choice([-1.0, 1.0], k)
        else:
            raise ValueError("input_mode must be 'dense' or 'labeled'")
        self.W_in = (input_scaling * W_in).astype(np.float32)
        self.bias = (bias_scaling * rng.uniform(-1.0, 1.0, self.n)).astype(np.float32)
        self.leak_rate = float(leak_rate)
        if backend not in ("numpy", "torch"):
            raise ValueError("backend must be 'numpy' or 'torch'")
        self.backend = backend
        self.device = device

    def run(self, U, record_idx=None, washout: int = 0) -> np.ndarray:
        """Drive the reservoir with inputs U and return recorded states.

        U: (T, n_features), or (B, T, n_features) for B independent streams sharing the wiring.
        Returns (T - washout, n_recorded) or (B, T - washout, n_recorded). State starts at zero,
        so drop the first `washout` steps while it forgets that arbitrary start.
        """
        U = np.asarray(U, dtype=np.float32)
        single = U.ndim == 2
        if single:
            U = U[None]
        if U.ndim != 3 or U.shape[2] != self.W_in.shape[1]:
            raise ValueError(f"U must be (T, {self.W_in.shape[1]}) or (B, T, {self.W_in.shape[1]}), got {U.shape}")
        rec = np.arange(self.n) if record_idx is None else np.asarray(record_idx, dtype=np.int64)
        drive = U @ self.W_in.T  # (B, T, k): input current into each input neuron
        states = (self._run_numpy if self.backend == "numpy" else self._run_torch)(drive, rec)[:, washout:]
        return states[0] if single else states

    def _run_numpy(self, drive: np.ndarray, rec: np.ndarray) -> np.ndarray:
        B, T, _ = drive.shape
        a = np.float32(self.leak_rate)
        x = np.zeros((self.n, B), dtype=np.float32)
        bias = self.bias[:, None]
        out = np.empty((B, T, len(rec)), dtype=np.float32)
        for t in range(T):
            pre = self.W @ x
            pre += bias
            pre[self.input_idx] += drive[:, t, :].T
            x = (1 - a) * x + a * np.tanh(pre)
            out[:, t, :] = x[rec].T
        return out

    def _run_torch(self, drive: np.ndarray, rec: np.ndarray) -> np.ndarray:
        import torch

        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))  # ROCm shows up as cuda
        W = self.W if self.W.has_sorted_indices else self.W.sorted_indices()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*")
            Wt = torch.sparse_csr_tensor(torch.as_tensor(W.indptr, dtype=torch.int64),
                                         torch.as_tensor(W.indices, dtype=torch.int64),
                                         torch.as_tensor(W.data), size=W.shape, device=dev,
                                         check_invariants=True)
        d = torch.as_tensor(drive, device=dev)
        bias = torch.as_tensor(self.bias, device=dev)[:, None]
        inp = torch.as_tensor(self.input_idx, device=dev)
        rec_t = torch.as_tensor(rec, device=dev)
        B, T, _ = drive.shape
        a = self.leak_rate
        x = torch.zeros((self.n, B), dtype=torch.float32, device=dev)
        out = torch.empty((T, len(rec), B), dtype=torch.float32, device=dev)
        with torch.no_grad():
            for t in range(T):
                pre = torch.sparse.mm(Wt, x) + bias
                pre.index_add_(0, inp, d[:, t, :].T.contiguous())
                x = (1 - a) * x + a * torch.tanh(pre)
                out[t] = x[rec_t]
        return out.permute(2, 0, 1).cpu().numpy()
