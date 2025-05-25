# ----------------------------------------------
# File: src/common/kohonen/som.py
# ----------------------------------------------
"""Self‑Organizing Map (Kohonen) implementation used in TP 4.

The map is rectangular (m rows × n cols) with a Gaussian neighbourhood
function and learning‑rate/σ exponential decay. Designed to be minimal
but fully functional for the Europe dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


class SOM:
    """Simple 2‑D Self‑Organising Map.

    Parameters
    ----------
    m, n : int
        Grid dimensions (``m`` rows × ``n`` columns).
    dim : int
        Dimension of the input vectors.
    n_iterations : int, default ``1000``
        Total number of weight‑updates (usually ``epochs × n_samples``).
    alpha : float | None, default ``None``
        Initial learning‑rate. If *None*, uses ``0.3``.
    sigma : float | None, default ``None``
        Initial neighbourhood radius. If *None*, uses ``max(m, n) / 2``.
    random_state : int | None, default ``None``
        Seed for the RNG so results are reproducible.
    """

    def __init__(
        self,
        m: int,
        n: int,
        dim: int,
        *,
        n_iterations: int = 1000,
        alpha: float | None = None,
        sigma: float | None = None,
        random_state: int | None = None,
    ) -> None:
        self.m = m
        self.n = n
        self.dim = dim
        self.n_iterations = int(n_iterations)

        # Hyper‑parameters with sensible defaults
        self.alpha0 = 0.3 if alpha is None else float(alpha)
        self.sigma0 = max(m, n) / 2.0 if sigma is None else float(sigma)

        # RNG
        self.rng = np.random.default_rng(random_state)

        # Weight matrix (units × dim)
        self.weights = self.rng.uniform(low=-1.0, high=1.0, size=(m * n, dim))

        # 2‑D coordinates for each unit to speed-up distance computations
        self.locations = np.array([(i, j) for i in range(m) for j in range(n)])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _decay(self, initial: float, t: int) -> float:
        """Exponential decay used for both α and σ."""
        return initial * np.exp(-t / self.n_iterations)

    def _gaussian_neighbourhood(self, bmu_idx: int, sigma: float) -> np.ndarray:
        """Return Gaussian neighbourhood vector *g* for every unit."""
        bmu_loc = self.locations[bmu_idx]
        dists = np.linalg.norm(self.locations - bmu_loc, axis=1)
        return np.exp(-(dists**2) / (2.0 * sigma**2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def winner(self, x: np.ndarray) -> int:  # Best Matching Unit index
        dists = np.linalg.norm(self.weights - x, axis=1)
        return int(np.argmin(dists))

    def update(self, x: np.ndarray, t: int) -> None:
        """Single online update for sample *x* at step *t*."""
        alpha_t = self._decay(self.alpha0, t)
        sigma_t = max(1e-3, self._decay(self.sigma0, t))

        bmu_idx = self.winner(x)
        g = self._gaussian_neighbourhood(bmu_idx, sigma_t)[:, np.newaxis]  # (units,1)
        self.weights += alpha_t * g * (x - self.weights)

    def fit(self, X: np.ndarray) -> "SOM":
        """Train the SOM in‑place and return *self* (for chaining)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError("Input array must be 2‑D with shape (n_samples, dim)")

        for t in range(self.n_iterations):
            x = X[t % len(X)]
            self.update(x, t)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map samples to their BMU grid coordinates (row, col)."""
        X = np.asarray(X, dtype=float)
        indices = np.apply_along_axis(self.winner, 1, X)
        return self.locations[indices]

    def save(self, path: str | Path) -> None:
        """Persist the weight matrix to ``*.npy`` file."""
        np.save(path, self.weights)

    # Convenience ------------------------------------------------------
    @property
    def codebook(self) -> np.ndarray:
        """Weights reshaped as (m, n, dim)."""
        return self.weights.reshape(self.m, self.n, self.dim)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SOM(m={self.m}, n={self.n}, dim={self.dim}, n_iter={self.n_iterations}, "
            f"alpha0={self.alpha0:.3f}, sigma0={self.sigma0:.3f})"
        )

# ----------------------------------------------------------------------
# Utility function: quick training in one call (handy for notebooks)
# ----------------------------------------------------------------------

def fit_som(
    X: np.ndarray,
    grid_size: Tuple[int, int] = (10, 10),
    *,
    n_iterations: int | None = None,
    random_state: int | None = None,
) -> SOM:
    """Train a SOM with default hyper‑params and return it."""
    m, n = grid_size
    n_iterations = n_iterations or (len(X) * 500)
    som = SOM(m, n, X.shape[1], n_iterations=n_iterations, random_state=random_state)
    som.fit(X)
    return som