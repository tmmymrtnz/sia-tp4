# ------------------------------------------------------------
#  src/analysis/activations.py
# ------------------------------------------------------------
"""
Experimento unificado de funciones de activación.
Recibe un único JSON con una lista de tareas (datasets) y corre todo.

Ejemplo:
    python -m analysis.activations configs/analysis/activations/all_tasks.json
"""
from __future__ import annotations
from pathlib import Path
import json, sys, itertools

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from analysis.utils.utils_exp import (
    load_dataset, run_experiment, plot_suite
)

# ---------------------------------------------------------------- #
def _build_variants(base_net, hidden_act_names, out_act_names,
                    loss_hidden, loss_out):
    """
    Devuelve un dict ‘variants’ listo para run_experiment().
    Se generan todas las combinaciones (hidden_act, out_act).
    """
    variants = {}
    for h_act, o_act in itertools.product(hidden_act_names, out_act_names):
        tag = f"{h_act}-{o_act}"
        net_cfg = {
            **base_net,
            "activations": [""] + [h_act]*(len(base_net["layer_sizes"])-2) + [o_act]
        }
        train_cfg = {
            "loss_name":   loss_out[o_act],
            **loss_hidden  # resto (optimizador, batch, etc.)
        }
        variants[tag] = {"net": net_cfg, "train": train_cfg}
    return variants


# ---------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) != 2:
        print("Uso: python -m analysis.activations <config.json>")
        sys.exit(1)

    cfg_all = json.loads(Path(sys.argv[1]).read_text())
    n_seeds = cfg_all.get("n_seeds", 3)

    # mappings para elegir loss según activación de SALIDA
    _loss_for_out = {"softmax": "cce", "sigmoid": "bce"}

    for task in cfg_all["tasks"]:
        tag_ds = task["tag"]            # ‘parity’, ‘mnist’, …
        print(f"\n=== DATASET :: {tag_ds.upper()} ===")

        # ------------- datos -----------------------------
        X_tr, y_tr, X_te, y_te = load_dataset(
            task["dataset"], task.get("data_path", "")
        )

        # ------------- variantes -------------------------
        base_net  = {"layer_sizes": task["layer_sizes"]}
        v = _build_variants(
            base_net              = base_net,
            hidden_act_names      = task["hidden_acts"],
            out_act_names         = task["out_acts"],
            loss_hidden           = task["train_cfg"],
            loss_out              = _loss_for_out
        )

        # ------------- correr ----------------------------
        df = run_experiment(
            X_tr, y_tr, X_te, y_te,
            variants      = v,
            cfg_train_base= task["train_cfg"],
            n_seeds       = n_seeds
        )

        # ------------- gráficas --------------------------
        out_dir = f"plots/activations_{tag_ds}"
        title   = task.get("plot_title", f"Activations – {tag_ds}")
        plot_suite(df,
                   metrics   = ("acc", "f1"),
                   out_prefix= out_dir,
                   title     = title)
        # tabla resumen
        print(df.groupby("config")[["acc", "f1"]]
                .agg(["mean", "std", "min", "max"])
                .round(4))

# ---------------------------------------------------------------- #
if __name__ == "__main__":
    main()
