#!/usr/bin/env python3
"""
Compara SGD, Momentum y Adam promediando sobre varias corridas.
Genera dos gráficos:
  1) Accuracy promedio ± [min,max] para cada optimizador.
  2) Curvas de pérdida promedio ± [min,max] por época para cada optimizador.

Uso:
    python src/analysis/optimizers.py configs/ex3/analysis_optimizers.json
"""
import sys, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# asegurarnos de que src/ esté en sys.path
sys.path.insert(0, "src")

from ex3.runner_parity import load_parity_dataset
from ex3.runner_digit  import load_digit_dataset
from common.perceptrons.multilayer.network import MLP
from common.perceptrons.multilayer.trainer import Trainer

def single_run(cfg, opt_cfg):
    """Entrena una vez y devuelve (accuracy, loss_history)."""
    task = cfg["task"]
    data_path = Path(cfg["data_path"])
    if task == "parity":
        X, y = load_parity_dataset(data_path)
    else:
        X, y = load_digit_dataset(data_path)

    layers = cfg["layer_sizes"]
    acts   = cfg["activations"]
    if len(acts) == len(layers) - 1:
        acts = [""] + acts

    net = MLP(layers, acts, cfg.get("dropout_rate", 0.0))
    trainer = Trainer(
        net             = net,
        loss_name       = cfg["loss"],
        optimizer_name  = opt_cfg["name"],
        optim_kwargs    = opt_cfg.get("optim_kwargs", {}),
        batch_size      = cfg["batch_size"],
        max_epochs      = cfg["max_epochs"],
        log_every       = 0,
        early_stopping  = cfg.get("early_stopping", True),
        patience        = cfg.get("patience", 10),
        min_delta       = cfg.get("min_delta", 1e-4)
    )

    loss_hist = trainer.fit(X, y)  # now returns loss_history
    y_hat = net.forward(X)

    if task == "parity":
        y_pred = (y_hat > 0.5).astype(int)
        acc    = np.mean(y_pred == y)
    else:
        preds = np.argmax(y_hat, axis=1)
        trues = np.argmax(y, axis=1)
        acc   = np.mean(preds == trues)

    return acc, loss_hist

def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    n_runs = cfg.get("n_runs", 10)

    acc_stats = {}
    loss_histories = {}

    for opt_cfg in cfg["optimizers"]:
        name = opt_cfg["name"]
        accs = []
        runs_losses = []
        print(f"→ Testing optimizer '{name}' for {n_runs} runs:")
        for i in range(n_runs):
            acc, hist = single_run(cfg, opt_cfg)
            accs.append(acc)
            runs_losses.append(hist)
            print(f"   run {i+1:2d}/{n_runs}  acc = {acc:.4f}  (epochs = {len(hist)})")

        arr = np.array(accs)
        acc_stats[name] = {
            "mean": arr.mean(),
            "min":  arr.min(),
            "max":  arr.max()
        }
        loss_histories[name] = runs_losses

    # --- Plot accuracy with min/max bars ---
    names      = list(acc_stats.keys())
    means      = [acc_stats[n]["mean"] for n in names]
    mins       = [acc_stats[n]["min"]  for n in names]
    maxs       = [acc_stats[n]["max"]  for n in names]
    lower_err  = np.array(means) - np.array(mins)
    upper_err  = np.array(maxs)  - np.array(means)

    plt.figure(figsize=(6,4))
    plt.errorbar(names, means,
                 yerr=[lower_err, upper_err],
                 fmt='o', capsize=5)
    plt.title("Optimizer Accuracy (mean ± [min,max])")
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot loss curves mean ± [min,max] ---
    # truncate to shortest run
    min_epochs = min(len(h) for runs in loss_histories.values() for h in runs)
    epochs = np.arange(1, min_epochs+1)

    plt.figure(figsize=(8,4))
    for name, runs in loss_histories.items():
        # build array of shape (n_runs, min_epochs)
        arr = np.array([h[:min_epochs] for h in runs])
        mean_loss = arr.mean(axis=0)
        min_loss  = arr.min(axis=0)
        max_loss  = arr.max(axis=0)

        plt.plot(epochs, mean_loss, label=name)
        plt.fill_between(epochs, min_loss, max_loss, alpha=0.2)

    plt.title("Training Loss Curve (mean ± [min,max])")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
