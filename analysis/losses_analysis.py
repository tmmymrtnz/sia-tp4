# ------------------------------------------------------------
#  src/analysis/losses_exp.py
# ------------------------------------------------------------
"""
Inciso C - MSE vs BCE para clasificación binaria.

Uso:
    python -m analysis.losses_exp configs/loss_parity.json
"""
from pathlib import Path
import json, sys

from analysis.utils_exp import load_dataset, run_experiment, plot_suite

def main():
    if len(sys.argv) != 2:
        print("Uso: python -m analysis.losses_exp <config.json>")
        sys.exit(1)

    cfg = json.loads(Path(sys.argv[1]).read_text())
    if cfg["dataset"] != "parity":
        print("⚠️  Este script asume dataset binario (parity).")
        sys.exit(1)

    X_tr, y_tr, X_te, y_te = load_dataset("parity", cfg["data_path"])

    base_net = {
        "layer_sizes": cfg["layer_sizes"],
        "activations": cfg["activations"]
    }
    variants = {
        "mse": {"net": base_net, "train": {"loss_name": "mse"}},
        "bce": {"net": base_net, "train": {"loss_name": "bce"}}
    }

    cfg_train = dict(
        optimizer_name = cfg["optimizer"],
        optim_kwargs   = cfg.get("optim_kwargs", {}),
        batch_size     = cfg.get("batch_size", 16),
        max_epochs     = cfg.get("max_epochs", 3000),
        log_every      = 0
    )

    df = run_experiment(
        X_tr, y_tr, X_te, y_te,
        variants      = variants,
        cfg_train_base= cfg_train,
        n_seeds       = cfg.get("n_seeds", 5)
    )

    out_dir = "plots/losses_parity"
    plot_suite(df, out_prefix=out_dir)
    print(df.groupby("config")["acc"].agg(["mean","std","min","max"]))


if __name__ == "__main__":
    main()
