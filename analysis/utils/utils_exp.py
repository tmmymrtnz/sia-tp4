# ------------------------------------------------------------
#  src/analysis/utils_exp.py
# ------------------------------------------------------------
"""
Funciones utilitarias compartidas por los experimentos de activación y pérdida.

▪ load_dataset  : parity | digit | mnist
▪ run_experiment: entrena varias seeds, devuelve métricas + historial
▪ plot_suite    : genera los gráficos estándar
"""
from __future__ import annotations
import time, json, sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedKFold

# --- core NN stack ---
sys.path.insert(0, "src")
from common.perceptrons.multilayer.network  import MLP
from common.perceptrons.multilayer.trainer  import Trainer
from ex3.runner_parity import load_parity_dataset      # ya existentes
from ex3.runner_digit  import load_digit_dataset

# --- keras para MNIST ---
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# ------------------------------------------------------------------ #
#  DATASETS
# ------------------------------------------------------------------ #
def _load_mnist_keras() -> tuple[np.ndarray, ...]:
    (x_tr, y_tr), (x_te, y_te) = mnist.load_data()          # (n,28,28)
    X_tr = x_tr.reshape(-1, 28*28).astype("float32") / 255.
    X_te = x_te.reshape(-1, 28*28).astype("float32") / 255.
    y_tr = to_categorical(y_tr, 10).astype("float32")
    y_te = to_categorical(y_te, 10).astype("float32")
    return X_tr, y_tr, X_te, y_te


def load_dataset(name: str, base_path: str | Path) -> tuple[np.ndarray, ...]:
    """
    name ∈ {'parity','digit','mnist'}
    ▸ parity / digit cargan desde archivo 7×5.
    ▸ mnist usa el loader de Keras y devuelve train + test.
    """
    if name == "parity":
        X, y = load_parity_dataset(Path(base_path))
        return X, y, X.copy(), y.copy()         # mismo set para “test”
    if name == "digit":
        X, y = load_digit_dataset(Path(base_path))
        return X, y, X.copy(), y.copy()
    if name == "mnist":
        return _load_mnist_keras()

    raise ValueError(f"Unknown dataset {name}")


# ------------------------------------------------------------------ #
#  RUNNER
# ------------------------------------------------------------------ #
def _fit_once(
    X_tr, y_tr, X_te, y_te,
    cfg_net: dict, cfg_train: dict
) -> dict:
    net  = MLP(cfg_net["layer_sizes"], cfg_net["activations"])
    trn  = Trainer(net=net, **cfg_train)
    t0   = time.time()
    loss_hist = trn.fit(X_tr, y_tr)           # Trainer devuelve list[loss]
    dur  = time.time() - t0

    # predicciones test
    y_prob = net.forward(X_te)
    if y_te.shape[1] == 1:                    # binario
        preds = (y_prob > 0.5).astype(int).ravel()
        trues = y_te.ravel()
        acc   = (preds == trues).mean()
    else:                                     # multiclase
        preds = np.argmax(y_prob, axis=1)
        trues = np.argmax(y_te, axis=1)
        acc   = (preds == trues).mean()

    return {"acc": acc, "loss_hist": loss_hist, "time": dur}


def run_experiment(
    X, y, X_test, y_test,
    variants: Dict[str, dict],
    cfg_train_base: dict,
    n_seeds: int = 5
) -> pd.DataFrame:
    """
    variants = {tag: {net_cfg_overrides, train_cfg_overrides}, ...}
    Devuelve dataframe wide con columnas:
        tag, seed, acc, time, loss_hist (object)
    """
    records: List[dict] = []
    for tag, tweaks in variants.items():
        for seed in range(n_seeds):
            np.random.seed(seed)
            cfg_net   = tweaks.get("net", {})
            cfg_train = cfg_train_base.copy()
            cfg_train.update(tweaks.get("train", {}))

            res = _fit_once(X, y, X_test, y_test, cfg_net, cfg_train)
            records.append({
                "config": tag,
                "seed":   seed,
                "acc":    res["acc"],
                "time":   res["time"],
                "loss_hist": res["loss_hist"]
            })
            print(f"[{tag} | seed {seed}] acc={res['acc']:.4f}  time={res['time']:.2f}s")
    return pd.DataFrame.from_records(records)


# ------------------------------------------------------------------ #
#  PLOTS
# ------------------------------------------------------------------ #
def _smooth(arr, win=31, poly=3):
    n = len(arr)
    if n < win: return arr
    return savgol_filter(arr, win if win < n else n - (n%2==0), poly)

def plot_suite(df: pd.DataFrame, metric="acc", out_prefix="plots"):
    """
    ▸ Box-plot de accuracy  
    ▸ Curvas de pérdida (promedio + banda)  
    ▸ Scatter tiempo versus accuracy
    """
    sns.set_theme(style="whitegrid")
    # --- boxplot accuracy -----------------------
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df, x="config", y="acc", palette="viridis")
    sns.stripplot(data=df, x="config", y="acc", color="k", size=3, alpha=.5)
    plt.ylabel("Accuracy"); plt.xlabel("")
    plt.title("Accuracy final por configuración")
    plt.tight_layout(); Path(out_prefix).mkdir(exist_ok=True, parents=True)
    plt.savefig(f"{out_prefix}/accuracy_box.png", dpi=250)

    # --- curvas de pérdida ----------------------
    plt.figure(figsize=(8,4))
    for tag, sub in df.groupby("config"):
        hists = np.stack(sub["loss_hist"].values)      # (n_seeds, epochs)
        mu    = hists.mean(axis=0)
        sd    = hists.std(axis=0)
        smu   = _smooth(mu)
        plt.plot(smu, label=tag)
        plt.fill_between(range(len(mu)), _smooth(mu-sd), _smooth(mu+sd), alpha=.15)
    plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("Loss (log)")
    plt.title("Convergencia – media ± desv.")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_prefix}/loss_curves.png", dpi=250)

    # --- tiempo vs accuracy ---------------------
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x="time", y="acc", hue="config", palette="viridis", s=70)
    plt.xlabel("Tiempo de entrenamiento (s)"); plt.ylabel("Accuracy")
    plt.title("Trade-off tiempo – precisión")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}/time_vs_acc.png", dpi=250)
    print(f"\nGráficos guardados en ‘{out_prefix}/’\n")
