# ------------------------------------------------------------
#  src/analysis/losses.py
# ------------------------------------------------------------
"""
Inciso C – Comparativa de funciones de pérdida (MSE vs BCE / CCE).

Uso:
    python -m analysis.losses <config.json>

El JSON admite los mismos campos que el resto de experimentos:
{
  "dataset"      : "parity" | "digit" | "mnist",
  "data_path"    : "data/TP3-ej3-digitos.txt",  # sólo si no es mnist
  "layer_sizes"  : [35,16,1],
  "activations"  : ["","tanh","sigmoid"],
  "optimizer"    : "adam",
  "optim_kwargs" : { "learning_rate": 0.01 },
  "batch_size"   : 32,
  "max_epochs"   : 200,
  "n_seeds"      : 5                            # repetición de seeds
}
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd

# --- librerías del stack local ------------------------------------------
sys.path.insert(0, "src")
from analysis.utils.utils_exp import (
    load_dataset, run_experiment, plot_suite
)

# ------------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------------ #
def _pick_loss_variants(n_outputs: int):
    """
    Devuelve un dict de variantes {nombre → cfg_train_overrides}
    según la dimensionalidad de la salida:
      • 1   →  binario   →  MSE vs BCE
      • >1  →  multiclase→  MSE vs Cross-Entropy (CCE)
    """
    if n_outputs == 1:
        return {
            "mse" : {"loss_name": "mse"},
            "bce" : {"loss_name": "bce"}
        }
    # multiclase
    return {
        "mse" : {"loss_name": "mse"},
        "cce" : {"loss_name": "cross_entropy"}
    }

def _decode_problem_type(y: np.ndarray) -> str:
    if y.ndim == 1 or y.shape[1] == 1:
        return "binary"
    return "multiclass"

# ------------------------------------------------------------------------ #
#  MAIN
# ------------------------------------------------------------------------ #
def main():
    if len(sys.argv) != 2:
        print("Uso: python -m analysis.losses <config.json>")
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    cfg      = json.loads(cfg_path.read_text())

    # ---------------- Dataset ----------------
    X_tr, y_tr, X_te, y_te = load_dataset(
        cfg["dataset"], cfg.get("data_path", "")
    )
    n_out = y_tr.shape[1] if y_tr.ndim == 2 else 1
    problem_type = _decode_problem_type(y_tr)
    print(f"[INFO] Problema detectado: {problem_type} "
          f"({n_out} salida{'s' if n_out>1 else ''})")

    # ---------------- Variante de pérdidas ---------------
    loss_variants = _pick_loss_variants(n_out)

    # ---------------- Config común de red ----------------
    base_net = {
        "layer_sizes" : cfg["layer_sizes"],
        "activations" : cfg["activations"]
    }

    # ---------------- Experimentos -----------------------
    variants = {}
    for name, train_over in loss_variants.items():
        variants[name] = {
            "net"       : base_net,
            "train"     : train_over
        }

    # config de entrenamiento por defecto
    cfg_train_base = dict(
        loss_name      = "mse",                     # placeholder (override)
        optimizer_name = cfg["optimizer"],
        optim_kwargs   = cfg.get("optim_kwargs", {}),
        batch_size     = cfg.get("batch_size", 32),
        max_epochs     = cfg.get("max_epochs", 400),
        log_every      = 0
    )

    # ---------------- Ejecutar ---------------------------
    df = run_experiment(
        X_tr, y_tr, X_te, y_te,
        variants       = variants,
        cfg_train_base = cfg_train_base,
        n_seeds        = cfg.get("n_seeds", 5)
    )

    # ---------------- Graficar / Tabla -------------------
    out_dir = f"plots/losses_{cfg['dataset']}"
    plot_suite(df, out_prefix=out_dir,
               title="Comparativa de pérdidas "
                     f"({problem_type.capitalize()})")

    print("\n=== Resumen ===")
    print(df.groupby("config")["acc"].agg(["mean", "std", "min", "max"]))


if __name__ == "__main__":
    main()
