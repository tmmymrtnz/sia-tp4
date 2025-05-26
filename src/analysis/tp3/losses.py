"""
loss.py · Parity — Binary Cross-Entropy vs MSE
================================================
Ejecuta el experimento propuesto en el inciso C:

    • Se entrena la misma red en el problema PARITY
      usando dos funciones de pérdida:  BCE  y  MSE.
    • Se repite training con varias seeds.
    • Se evalúa la exactitud final, la curva de loss
      y el desempeño sobre versiones ruidosas del
      conjunto (noisy1-4) + clean.

Genera en <out_dir>/ :
    ├── acc_box.png      (accuracy final)
    ├── loss_curves.png  (curva media ± σ)
    └── noise_acc.png    (robustez con datasets ruidosos)

Uso:
    python src/analysis/loss.py configs/ex3/loss_parity.json
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# stack MLP / Trainer
sys.path.insert(0, "src")
from common.perceptrons.multilayer.network  import MLP
from common.perceptrons.multilayer.trainer  import Trainer
from ex3.runner_parity                     import load_parity_dataset


# ──────────────────────────────────────────────────────────────── #
def _train_once(X_tr, y_tr, X_te, y_te,
                cfg_net: dict, cfg_train: dict) -> Dict:
    """Entrena una vez y devuelve métricas + red entrenada."""
    net = MLP(cfg_net["layer_sizes"], cfg_net["activations"])
    trainer = Trainer(net=net, **cfg_train)

    t0 = time.time()
    loss_hist = trainer.fit(X_tr, y_tr)
    dur = time.time() - t0

    y_prob = net.forward(X_te)
    preds  = (y_prob > 0.5).astype(int).ravel()
    acc    = (preds == y_te.ravel()).mean()

    return {
        "net":       net,
        "loss_hist": loss_hist,
        "acc":       acc,
        "time":      dur
    }


# ──────────────────────────────────────────────────────────────── #
def main(cfg_path: Path):
    cfg = json.loads(cfg_path.read_text())

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # —— carga datasets ———————————————————————————————— #
    X_clean, y_clean = load_parity_dataset(Path(cfg["dataset_clean"]))
    noisy_sets = {
        Path(p).stem: load_parity_dataset(Path(p))
        for p in cfg["datasets_noisy"]
    }

    # —— especificaciones de la red ————————————————————— #
    cfg_net = {
        "layer_sizes" : cfg["layer_sizes"],
        "activations" : [cfg["hidden_act"], cfg["out_act"]]
    }

    # —— las dos variantes de loss —————————————————————— #
    loss_variants = {"mse": "mse", "bce": "bce"}

    records: List[dict] = []
    noise_accum: Dict[str, List[float]] = {
        loss_name: {ds: [] for ds in ["clean", *noisy_sets]}
        for loss_name in loss_variants
    }

    for loss_tag, loss_name in loss_variants.items():
        for seed in range(cfg["n_seeds"]):
            rng_state = cfg["random_state_base"] + seed
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_clean, y_clean, test_size=cfg["test_size"],
                stratify=y_clean.ravel(), random_state=rng_state
            )

            cfg_train = dict(
                loss_name      = loss_name,
                optimizer_name = cfg["train_cfg"]["optimizer_name"],
                optim_kwargs   = cfg["train_cfg"]["optim_kwargs"],
                batch_size     = cfg["train_cfg"]["batch_size"],
                max_epochs     = cfg["train_cfg"]["max_epochs"],
                log_every      = 0
            )

            res = _train_once(X_tr, y_tr, X_te, y_te, cfg_net, cfg_train)

            # — accuracy sobre datasets ruidosos — #
            noise_acc = {}
            for tag, (Xn, yn) in noisy_sets.items():
                yp = res["net"].forward(Xn)
                acc_n = ((yp > 0.5).astype(int).ravel() == yn.ravel()).mean()
                noise_acc[tag] = acc_n
                noise_accum[loss_tag][tag].append(acc_n)

            noise_accum[loss_tag]["clean"].append(res["acc"])

            records.append({
                "config"    : loss_tag,
                "seed"      : seed,
                "acc"       : res["acc"],
                "time"      : res["time"],
                "loss_hist" : res["loss_hist"],
                **{f"acc_{t}": a for t, a in noise_acc.items()}
            })

            print(f"[{loss_tag:3s} | seed {seed}] acc={res['acc']:.3f} "
                  f"time={res['time']:.2f}s")

    df = pd.DataFrame.from_records(records)

    # ──────────────────── G R Á F I C O S ───────────────────────── #
    sns.set_theme(style="whitegrid")

    # —— Accuracy final —— #
    plt.figure(figsize=(4,4))
    sns.boxplot(data=df, x="config", y="acc", palette="viridis")
    sns.stripplot(data=df, x="config", y="acc",
                  color="k", size=3, alpha=.6)
    plt.title("PARITY · Accuracy final")
    plt.xlabel("Loss function"); plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "acc_box.png", dpi=250); plt.close()

    # —— Curvas de pérdida —— #
    plt.figure(figsize=(5,4))
    for loss_tag, sub in df.groupby("config"):
        h = np.stack(sub["loss_hist"].values)     # (runs, epochs)
        mu, sd = h.mean(0), h.std(0)
        plt.plot(mu, label=loss_tag.upper())
        plt.fill_between(range(len(mu)), mu-sd, mu+sd, alpha=.15)
    plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("Loss (log)")
    plt.title("PARITY · Convergencia")
    plt.legend(title="loss"); plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=250); plt.close()

    # —— Robustez a ruido —— #
    labels = ["clean", *noisy_sets]
    width  = 0.35
    x = np.arange(len(labels))

    plt.figure(figsize=(6,4))
    for i, (loss_tag, scores) in enumerate(noise_accum.items()):
        mu = [np.mean(scores[lbl]) for lbl in labels]
        plt.bar(x + i*width - width/2, mu, width,
                label=loss_tag.upper(), alpha=.9)
    plt.xticks(x, labels, rotation=45)
    plt.ylim(0,1); plt.ylabel("Accuracy")
    plt.title("PARITY · Robustez con entradas ruidosas")
    plt.legend(title="loss")
    plt.tight_layout()
    plt.savefig(out_dir / "noise_acc.png", dpi=250); plt.close()

    print(f"\n✔ Gráficos guardados en «{out_dir}/»")


# ———————————————————————————————————————————————————————————— #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso:  python src/analysis/loss.py <config.json>")
        sys.exit(1)
    main(Path(sys.argv[1]))
