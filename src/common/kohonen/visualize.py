# ----------------------------------------------
# File: src/common/kohonen/visualize.py
# ----------------------------------------------
"""Visualization helpers for Self‑Organizing Maps."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ----------------------------------------------------------------------

def plot_hits(som, X: np.ndarray, *, ax=None):
    """Heatmap: number of samples that map to each neuron."""
    hits = np.zeros((som.m, som.n), dtype=int)
    for r, c in som.transform(X):
        hits[r, c] += 1
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(hits, annot=True, fmt="d", cmap="viridis", cbar=False, ax=ax)
    ax.set_title("Hit‑map (sample density)")
    ax.invert_yaxis()
    return ax


def plot_umatrix(som, *, ax=None):
    """U‑matrix showing average distance to neighbours."""
    m, n = som.m, som.n
    codebook = som.codebook
    umat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            neigh = []
            if i > 0:
                neigh.append(codebook[i - 1, j])
            if i < m - 1:
                neigh.append(codebook[i + 1, j])
            if j > 0:
                neigh.append(codebook[i, j - 1])
            if j < n - 1:
                neigh.append(codebook[i, j + 1])
            umat[i, j] = np.mean([np.linalg.norm(codebook[i, j] - nb) for nb in neigh])
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(umat, cmap="rocket", ax=ax)
    ax.set_title("U‑matrix (cluster boundaries)")
    ax.invert_yaxis()
    return ax


def plot_scatter(som, X: np.ndarray, labels: list[str], *, ax=None):
    """Scatter plot of BMU positions with text labels (e.g. country names)."""
    pos = som.transform(X)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pos[:, 1], pos[:, 0], s=60, alpha=0.7)
    for (r, c), text in zip(pos, labels):
        ax.text(c, r, text, ha="center", va="center", fontsize=8)
    ax.set_xticks(range(som.n))
    ax.set_yticks(range(som.m))
    ax.invert_yaxis()
    ax.set_title("Samples on SOM grid")
    return ax