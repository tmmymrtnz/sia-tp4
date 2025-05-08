# ------------------------------------------------------------
#  src/analysis/activations_exp.py
# ------------------------------------------------------------
"""
Inciso B - Comparativa de funciones de activación.

Ejemplo:
    python -m analysis.activations_exp configs/activations_digit.json
"""
from pathlib import Path
import json, sys

import pandas as pd
from analysis.utils_exp import load_dataset, run_experiment, plot_suite

# ------------------------------------------------------------ #
def main():
    if len(sys.argv) != 2:
        print("Uso: python -m analysis.activations_exp <config.json>")
        sys.exit(1)

    cfg = json.loads(Path(sys.argv[1]).read_text())

    # -------- dataset -------------
    X_tr, y_tr, X_te, y_te = load_dataset(cfg["dataset"], cfg["data_path"])

    # -------- variantes -----------
    base_net = {"layer_sizes": cfg["layer_sizes"]}
    variants = {
        "sigmoid": {"net": {**base_net, "activations": [""] + ["sigmoid"]*(len(cfg["layer_sizes"])-1)}},
        "tanh"   : {"net": {**base_net, "activations": [""] + ["tanh"]*(len(cfg["layer_sizes"])-1)}},
        "relu"   : {"net": {**base_net, "activations": [""] + ["relu"]*(len(cfg["layer_sizes"])-1)}}
    }
    # output softmax si multiclase
    if y_tr.shape[1] > 1:
        for k in variants:
            acts = variants[k]["net"]["activations"]
            acts[-1] = "softmax"

    # -------- config train --------
    cfg_train = dict(
        loss_name      = cfg["loss"],
        optimizer_name = cfg["optimizer"],
        optim_kwargs   = cfg.get("optim_kwargs", {}),
        batch_size     = cfg.get("batch_size", 32),
        max_epochs     = cfg.get("max_epochs", 500),
        log_every      = 0
    )

    # -------- correr --------------
    df = run_experiment(
        X_tr, y_tr, X_te, y_te,
        variants      = variants,
        cfg_train_base= cfg_train,
        n_seeds       = cfg.get("n_seeds", 5)
    )

    # -------- gráficas ------------
    out_dir = f"plots/activations_{cfg['dataset']}"
    plot_suite(df, out_prefix=out_dir)

    # -------- tabla resumen -------
    print(df.groupby("config")["acc"].agg(["mean","std","min","max"]))


if __name__ == "__main__":
    main()
