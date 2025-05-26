# ------------------------------------------------------------
#  TP4 – Ej. 1.1 · Análisis de hiper-parámetros de Kohonen SOM
# ------------------------------------------------------------
"""
Explora, de a un hiper-parámetro por vez, su impacto sobre
Quantization Error (QE) y Topographic Error (TE).

Ejemplo de uso
--------------
python -m src.tp4.ex1.hyperparam_analysis \
       --data  data/europe.csv \
       --label Country \
       --out   plots/kohonen_scan
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.common.kohonen.som import SOM  # import relativo

# --------------------------- métricas --------------------------------- #
def quantization_error(som: SOM, X: np.ndarray) -> float:
    """
    QE = distancia promedio entre cada muestra y el peso de su BMU.
    """
    bmu_idxs = np.apply_along_axis(som.winner, 1, X)    # (N,)
    dists = np.linalg.norm(X - som.weights[bmu_idxs], axis=1)
    return float(dists.mean())


def topographic_error(som: SOM, X: np.ndarray) -> float:
    """
    TE = % de muestras cuyo 2.º BMU no es vecino 4-conexo del 1.º.
    """
    errs = 0
    for x in X:
        d = np.linalg.norm(som.weights - x, axis=1)
        idx1, idx2 = np.argsort(d)[:2]
        r1, c1 = som.locations[idx1]
        r2, c2 = som.locations[idx2]
        if max(abs(r1 - r2), abs(c1 - c2)) > 1:   # no son vecinos directos
            errs += 1
    return errs / len(X)



# ------------------------- exploración -------------------------------- #
def single_scan(
    X: np.ndarray,
    *,
    label: str,
    values: list,
    param: str,
    base_cfg: dict,
    out_dir: Path,
):
    records = []

    for val in tqdm(values, desc=f"{param}", ncols=75):
        cfg = base_cfg.copy()
        cfg[param] = val
        som = SOM(
            cfg["m"],
            cfg["n"],
            dim=X.shape[1],
            n_iterations=cfg["n_iterations"],
            alpha=cfg["alpha"],
            sigma=cfg["sigma"],
            random_state=cfg["seed"],
        ).fit(X)

        qe = quantization_error(som, X)
        te = topographic_error(som, X)

        records.append({param: val, "QE": qe, "TE": te, "seed": cfg["seed"]})

    df = pd.DataFrame.from_records(records)
    df.to_csv(out_dir / f"{param}.csv", index=False)

    # ----------- gráficos ---------- #
    for metric in ("QE", "TE"):
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=df, x=param, y=metric, marker="o")
        plt.title(f"{metric} vs {param}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric.lower()}_{param}.png", dpi=300)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ruta CSV (Europe dataset)")
    ap.add_argument("--label", default="Country", help="Columna etiqueta")
    ap.add_argument("--out", default="plots/kohonen_scan", help="Directorio salida")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------- datos
    df = pd.read_csv(args.data)
    labels = df.pop(args.label)  # quitamos la columna país
    X = StandardScaler().fit_transform(df.values)

    # ---------------------------------------------------- base cfg
    base = dict(
        m=10,
        n=10,
        n_iterations=4500,
        alpha=0.4,
        sigma=5,
        seed=0,
    )

    # 🚦 hiper-parámetros a explorar (uno por vez)
    grid = {
        "m": [6, 8, 10, 12],
        "n": [6, 8, 10, 12],
        "n_iterations": [1500, 3000, 6000, 10000],
        "alpha": [0.1, 0.3, 0.5, 0.7],
        "sigma": [2, 4, 6, 8],
        "seed": [0, 42, 1337, 2025],
    }

    for param, values in grid.items():
        single_scan(X, label=args.label, values=values, param=param, base_cfg=base, out_dir=out_dir)

    print(f"✓ Resultados y gráficos guardados en → {out_dir}")


if __name__ == "__main__":
    main()
