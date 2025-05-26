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
    """
    Scatter de posiciones BMU con nombres de países:
    – si una neurona tiene >1 país, apila los nombres con un pequeño offset
      vertical, así no se pisan.
    """
    pos = som.transform(X)           # (n_samples, 2)
    grouped: dict[tuple[int, int], list[str]] = {}
    for (r, c), lab in zip(pos, labels):
        grouped.setdefault((r, c), []).append(lab)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # dibujamos un punto por neurona ocupada
    for (r, c), labs in grouped.items():
        ax.scatter(c, r, s=60, alpha=0.7)
        if len(labs) == 1:
            ax.text(c, r + 0.15, labs[0], ha="center", va="bottom", fontsize=8)
        else:
            for i, lab in enumerate(labs):
                ax.text(
                    c,
                    r + 0.15 + i * 0.25,    # desplazamiento incremental
                    lab,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(range(som.n))
    ax.set_yticks(range(som.m))
    ax.invert_yaxis()                 # ← para alinear con el heat-map
    ax.set_title("Samples on SOM grid")
    return ax
