from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- HIT-MAP ----------------------------------------------------------- #
def plot_hits(som, X: np.ndarray, *, ax=None):
    """
    Plots the hit map (sample density) with row 0 at the bottom,
    adding extra margins for clarity.
    """
    hits = np.zeros((som.m, som.n), dtype=int)
    for r, c in som.transform(X):
        hits[r, c] += 1

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # increased overall frame
    else:
        fig = ax.get_figure()

    im = ax.imshow(
        hits,
        cmap=sns.color_palette("viridis", as_cmap=True),
        origin='lower', interpolation='nearest'
    )
    # Annotate counts at cell centers
    for (i, j), val in np.ndenumerate(hits):
        ax.text(j, i, str(val), ha='center', va='center', fontsize=8)

    ax.set_title("Hit-map (sample density)")
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    # Set ticks at each integer cell
    ax.set_xticks(np.arange(som.n))
    ax.set_xticklabels(np.arange(som.n))
    ax.set_yticks(np.arange(som.m))
    ax.set_yticklabels(np.arange(som.m))

    ax.set_aspect('equal')
    # Adjust margins
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    return ax

# --- U-MATRIX ---------------------------------------------------------- #
def plot_umatrix(som, *, ax=None):
    """
    Plots the U-matrix (cluster boundaries) with row 0 at the bottom,
    using an inset colorbar that preserves plot size.
    """
    m, n = som.m, som.n
    code = som.codebook
    umat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            neighbors: list[np.ndarray] = []
            if i > 0:
                neighbors.append(code[i - 1, j])
            if i < m - 1:
                neighbors.append(code[i + 1, j])
            if j > 0:
                neighbors.append(code[i, j - 1])
            if j < n - 1:
                neighbors.append(code[i, j + 1])
            umat[i, j] = np.mean([np.linalg.norm(code[i, j] - nb) for nb in neighbors])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    rocket_cmap = sns.color_palette("rocket", as_cmap=True)
    im = ax.imshow(
        umat,
        cmap=rocket_cmap,
        origin='lower', interpolation='nearest'
    )
    # Create an inset colorbar that does not shrink the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    ax.set_title("U-matrix (cluster boundaries)")
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    # Set ticks at each integer cell
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(np.arange(n))
    ax.set_yticks(np.arange(m))
    ax.set_yticklabels(np.arange(m))

    ax.set_aspect('equal')
    # Adjust margins
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    return ax

# --- SCATTER ----------------------------------------------------------- #
def plot_scatter(som, X: np.ndarray, labels: list[str], *, ax=None):
    """
    Plots sample positions on the SOM grid with row 0 at the bottom,
    adding extra margins.
    """
    positions = som.transform(X)
    groups: dict[tuple[int, int], list[str]] = {}
    for (r, c), lab in zip(positions, labels):
        groups.setdefault((r, c), []).append(lab)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    for (r, c), labs in groups.items():
        ax.scatter(c, r, s=60)
        if len(labs) == 1:
            ax.text(c, r + 0.2, labs[0], ha="center", va="bottom", fontsize=8)
        else:
            for i, lab in enumerate(labs):
                ax.text(c, r + 0.2 + i * 0.25, lab, ha="center", va="bottom", fontsize=7)

    ax.set_title("Samples on SOM grid")
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    # Set ticks at each integer cell
    ax.set_xticks(np.arange(som.n))
    ax.set_xticklabels(np.arange(som.n))
    ax.set_yticks(np.arange(som.m))
    ax.set_yticklabels(np.arange(som.m))

    ax.set_aspect('equal')
    # Adjust margins
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    return ax