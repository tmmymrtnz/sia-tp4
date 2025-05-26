# ------------------------------------------------------------
#  TP4 – Ej. 1.1 · Análisis de hiper-parámetros de Kohonen SOM
# ------------------------------------------------------------
"""
Explora, de a un hiper–parámetro por vez, su impacto sobre
Quantization Error (QE) y Topographic Error (TE).

Ahora se agrega un barrido **lado × lado** para la grilla
(6×6 → … → 12×12), evitando repetir m y n por separado.

Ejemplo de uso
--------------
python -m src.tp4.ex1.hyperparam_analysis \
       --data  data/europe.csv \
       --label Country \
       --out   plots/kohonen_scan
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.common.kohonen.som import SOM  # import absoluto desde src root

# --------------------------- métricas --------------------------------- #

def quantization_error(som: SOM, X: np.ndarray) -> float:
    """QE = distancia promedio entre cada muestra y su BMU."""
    idx = np.apply_along_axis(som.winner, 1, X)
    d   = np.linalg.norm(X - som.weights[idx], axis=1)
    return float(d.mean())


def topographic_error(som: SOM, X: np.ndarray) -> float:
    """TE = % de patrones cuyo segundo BMU no es vecino 4-conexo del primero."""
    err = 0
    for x in X:
        d     = np.linalg.norm(som.weights - x, axis=1)
        i1, i2 = np.argsort(d)[:2]
        r1, c1 = som.locations[i1]
        r2, c2 = som.locations[i2]
        if max(abs(r1 - r2), abs(c1 - c2)) > 1:
            err += 1
    return err / len(X)

# ------------------------- exploración -------------------------------- #

def single_scan(
    X: np.ndarray,
    *,
    values: list[int | float],
    param: str,
    base_cfg: dict,
    out_dir: Path,
):
    """Barre *un* hiper–parámetro manteniendo los demás en base_cfg."""
    records: list[dict] = []

    for val in tqdm(values, desc=f"{param}", ncols=70):
        cfg = base_cfg.copy()

        if param == "side":        # ← barrido conjunto m = n = side
            cfg["m"] = cfg["n"] = val
        else:
            cfg[param] = val

        som = (
            SOM(
                cfg["m"], cfg["n"], X.shape[1],
                n_iterations=cfg["n_iterations"],
                alpha=cfg["alpha"], sigma=cfg["sigma"],
                random_state=cfg["seed"],
            )
            .fit(X)
        )

        qe = quantization_error(som, X)
        te = topographic_error(som, X)
        records.append({param: val, "QE": qe, "TE": te})

    df = pd.DataFrame.from_records(records)
    df.to_csv(out_dir / f"{param}.csv", index=False)

    # ----- gráfico
    for metric in ("QE", "TE"):
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=df, x=param, y=metric, marker="o")
        plt.title(f"{metric} vs {param}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric.lower()}_{param}.png", dpi=300)
        plt.close()


# ---------------------------- main ------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ruta CSV (Europe dataset)")
    ap.add_argument("--label", default="Country", help="Columna etiqueta")
    ap.add_argument("--out", default="plots/kohonen_scan", help="Dir. salida")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- datos
    df = pd.read_csv(args.data)
    df.drop(columns=[args.label], inplace=False)
    X = StandardScaler().fit_transform(df.drop(columns=[args.label]).values)

    # ---------------- base config (valores óptimos)
    base = dict(m=10, n=10, n_iterations=6000, alpha=0.5, sigma=4, seed=42)

    # ------------- hiper-parámetros a barrer ---------------
    grid: dict[str, list] = {
        "side": [6, 7, 8, 9, 10, 11, 12],
        "n_iterations": [1500, 3000, 6000, 10000],
        "alpha": [0.1, 0.3, 0.5, 0.7],
        "sigma": [2, 4, 6, 8],
        "seed": [0, 42, 1337, 2025],
    }

    for param, values in grid.items():
        single_scan(X, values=values, param=param, base_cfg=base, out_dir=out_dir)

    print(f"✓ Resultados guardados en → {out_dir}")


if __name__ == "__main__":
    main()