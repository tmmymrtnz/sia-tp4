# ----------------------------------------------
# File: src/tp4/ex1/runner_kohonen.py
# ----------------------------------------------
"""Command-line runner for SOM experiment (TP 4 – Ej. 1.1)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ...common.kohonen.som import SOM
from ...common.kohonen import visualize as viz

# ----------------------------------------------------------------------
# utilidades métricas ---------------------------------------------------
def quantization_error(som: SOM, X: np.ndarray) -> float:
    """QE = distancia promedio entre cada muestra y su BMU."""
    bmu_idx = np.apply_along_axis(som.winner, 1, X)          # (N,)
    dists   = np.linalg.norm(X - som.weights[bmu_idx], axis=1)
    return float(dists.mean())


def topographic_error(som: SOM, X: np.ndarray) -> float:
    """TE = % de muestras cuyo 2.º BMU no es vecino 4-conexo del 1.º."""
    err = 0
    for x in X:
        d     = np.linalg.norm(som.weights - x, axis=1)
        idx1, idx2 = np.argsort(d)[:2]                      # BMU y 2.º BMU
        r1, c1 = som.locations[idx1]
        r2, c2 = som.locations[idx2]
        if max(abs(r1 - r2), abs(c1 - c2)) > 1:             # no vecinos directos
            err += 1
    return err / len(X)

# ----------------------------------------------------------------------
def load_config(path: str | Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

# ----------------------------------------------------------------------
def run(cfg_path: str | Path):
    cfg = load_config(cfg_path)

    # ------------------------------------------------------------------
    # 1) Datos ----------------------------------------------------------
    # ------------------------------------------------------------------
    df = pd.read_csv(cfg["data_path"])
    label_col = cfg.get("label_column", df.columns[0])
    labels = df[label_col].astype(str).tolist()
    X = df.drop(columns=[label_col]).values

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # 2) Entrenamiento --------------------------------------------------
    # ------------------------------------------------------------------
    m, n = cfg.get("grid", [10, 10])
    som = SOM(
        m,
        n,
        dim=X_std.shape[1],
        n_iterations=cfg.get("n_iterations", 500 * len(X_std)),
        alpha=cfg.get("alpha"),
        sigma=cfg.get("sigma"),
        random_state=cfg.get("seed", 42),
    ).fit(X_std)

    # ------------------------------------------------------------------
    # 3) Métricas -------------------------------------------------------
    # ------------------------------------------------------------------
    qe = quantization_error(som, X_std)
    te = topographic_error(som, X_std)

    # ------------------------------------------------------------------
    # 4) Salidas --------------------------------------------------------
    # ------------------------------------------------------------------
    out_dir = Path(cfg.get("out_dir", "plots/kohonen"))
    out_dir.mkdir(parents=True, exist_ok=True)

    som.save(out_dir / "weights.npy")
    np.save(out_dir / "locations.npy", som.transform(X_std))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    viz.plot_hits(som, X_std, ax=axes[0])
    viz.plot_umatrix(som, ax=axes[1])
    viz.plot_scatter(som, X_std, labels, ax=axes[2])
    plt.tight_layout()
    fig.savefig(out_dir / "overview.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5) Log ------------------------------------------------------------
    # ------------------------------------------------------------------
    print(f"✓ SOM entrenado. Resultados en → {out_dir}")
    print(f"   Quantization Error (QE): {qe:.4f}")
    print(f"   Topographic Error  (TE): {te:.4f}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner: Kohonen SOM TP4-Ex1.1")
    parser.add_argument("config", help="Ruta al JSON de configuración")
    args = parser.parse_args()
    run(args.config)
