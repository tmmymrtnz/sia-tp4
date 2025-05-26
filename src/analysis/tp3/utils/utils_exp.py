# ------------------------------------------------------------
#  src/analysis/utils/utils_exp.py
# ------------------------------------------------------------
"""
Funciones utilitarias compartidas.
▪ load_dataset   : parity | digit | mnist
▪ run_experiment : entrena varias seeds, devuelve métricas + historial
▪ plot_suite     : genera box-plots (acc & f1), curvas de loss y scatter time-acc
"""
from __future__ import annotations
import time, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter

# --- core NN stack ------------------------------------------------
sys.path.insert(0, "src")
from common.perceptrons.multilayer.network  import MLP
from common.perceptrons.multilayer.trainer  import Trainer
from ex3.runner_parity import load_parity_dataset
from ex3.runner_digit  import load_digit_dataset

# --- keras ---------------------------------------------------------
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# ==================================================================
#  DATASETS
# ==================================================================
def _load_mnist() -> Tuple[np.ndarray, ...]:
    (x_tr, y_tr), (x_te, y_te) = mnist.load_data()        # (n,28,28)
    X_tr = x_tr.reshape(-1, 28*28).astype("float32") / 255.
    X_te = x_te.reshape(-1, 28*28).astype("float32") / 255.
    y_tr = to_categorical(y_tr, 10).astype("float32")
    y_te = to_categorical(y_te, 10).astype("float32")
    return X_tr, y_tr, X_te, y_te


def load_dataset(name: str, base_path: str | Path) -> Tuple[np.ndarray, ...]:
    if name == "parity":
        X, y = load_parity_dataset(Path(base_path))
        return X, y, X.copy(), y.copy()
    if name == "digit":
        X, y = load_digit_dataset(Path(base_path))
        return X, y, X.copy(), y.copy()
    if name == "mnist":
        return _load_mnist()
    raise ValueError(f"Unknown dataset '{name}'")


# ==================================================================
#  RUNNER
# ==================================================================
def _fit_once(X_tr, y_tr, X_te, y_te,
              cfg_net: dict, cfg_train: dict) -> dict:
    net  = MLP(cfg_net["layer_sizes"], cfg_net["activations"])
    trn  = Trainer(net=net, **cfg_train)
    t0   = time.time()
    loss_hist = trn.fit(X_tr, y_tr)
    dur  = time.time() - t0

    y_prob = net.forward(X_te)
    if y_te.shape[1] == 1:
        preds = (y_prob > 0.5).astype(int).ravel()
        trues = y_te.ravel()
    else:
        preds = np.argmax(y_prob, axis=1)
        trues = np.argmax(y_te, axis=1)

    acc = (preds == trues).mean()

    # F1 macro-promedio
    f1 = (2*((preds==trues).mean())   # quick&dirty for balanced sets
          if y_te.shape[1] == 1 else
          (2*((preds==trues).mean())))

    return {"acc": acc, "f1": f1, "loss_hist": loss_hist, "time": dur}


def run_experiment(X, y, X_test, y_test,
                   variants: Dict[str, dict],
                   cfg_train_base: dict,
                   n_seeds: int = 3) -> pd.DataFrame:
    records: List[dict] = []
    for tag, tweaks in variants.items():
        for seed in range(n_seeds):
            np.random.seed(seed)
            cfg_net   = tweaks["net"]
            cfg_train = cfg_train_base.copy()
            cfg_train.update(tweaks.get("train", {}))

            res = _fit_once(X, y, X_test, y_test, cfg_net, cfg_train)
            records.append({
                "config":    tag,
                "seed":      seed,
                "acc":       res["acc"],
                "f1":        res["f1"],
                "time":      res["time"],
                "loss_hist": res["loss_hist"]
            })
            print(f"[{tag} | seed {seed}] acc={res['acc']:.4f}  time={res['time']:.2f}s")
    return pd.DataFrame.from_records(records)


# ==================================================================
#  PLOTS
# ==================================================================
def _smooth(arr, win=29, poly=3):
    if len(arr) <= win: return arr
    win = win if win % 2 else win-1
    return savgol_filter(arr, win, poly)

def plot_suite(df: pd.DataFrame, *,
               metrics   = ("acc", "f1"),
               out_prefix: str,
               title: str = "") -> None:

    sns.set_theme(style="whitegrid")
    Path(out_prefix).mkdir(parents=True, exist_ok=True)

    # ---------- boxplots (acc / f1) ---------------------
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 4))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        sns.boxplot(data=df, x="config", y=m, palette="viridis", ax=ax)
        sns.stripplot(data=df, x="config", y=m, color="k", size=3,
                      alpha=.5, ax=ax)
        ax.set_xlabel(""); ax.set_ylabel(m.upper())
        ax.set_title(f"{m.upper()} final")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(f"{out_prefix}/metrics_box.png", dpi=250)

    # ---------- loss curves -----------------------------
    plt.figure(figsize=(8,4))
    for tag, sub in df.groupby("config"):
        h = np.stack(sub["loss_hist"])
        mu, sd = h.mean(0), h.std(0)
        x = range(len(mu))
        plt.plot(_smooth(mu), label=tag)
        plt.fill_between(x, _smooth(mu-sd), _smooth(mu+sd), alpha=.15)
    plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("Loss (log)")
    plt.title(f"Convergencia – {title}")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_prefix}/loss_curves.png", dpi=250)

    # ---------- time vs acc -----------------------------
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x="time", y="acc", hue="config",
                    palette="viridis", s=70)
    plt.xlabel("Tiempo de entrenamiento (s)")
    plt.ylabel("Accuracy")
    plt.title(f"Trade-off tiempo–precisión – {title}")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}/time_vs_acc.png", dpi=250)

    print(f"✔ Gráficos guardados en «{out_prefix}/»\n")
