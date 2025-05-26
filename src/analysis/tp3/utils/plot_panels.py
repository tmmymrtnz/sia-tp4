# ------------------------------------------------------------
#  src/analysis/utils/plot_panels.py
# ------------------------------------------------------------
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter


# ------------------------------------------------------------------ #
def _smooth(arr, win=21, poly=3):
    n = len(arr)
    if n < win:
        return arr
    win = win if win % 2 else win - 1        # ventana impar
    return savgol_filter(arr, min(win, n), poly)


# ------------------------------------------------------------------ #
#  PANEL PARITY ───────────────────────────────────────────────────── #
def panel_parity(df: pd.DataFrame, out_dir="plots/panel_parity"):
    """
    Graba tres png:
        acc_box.png        (Accuracy final)
        loss_curves.png    (curva media ± σ)
        time_bar.png       (barra tiempo mean ± σ)

    Espera que df['config'] sea  {sigmoid, tanh, relu}.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    order = ["sigmoid", "tanh", "relu"]

    # ---------- ACC -------------------------------------------------
    plt.figure(figsize=(4, 4))
    sns.boxplot(data=df, x="config", y="acc", order=order,
                palette="viridis")
    sns.stripplot(data=df, x="config", y="acc", order=order,
                  color="k", size=3, alpha=.6)
    plt.title("PARITY · Accuracy final")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/acc_box.png", dpi=250)

    # ---------- LOSS ------------------------------------------------
    plt.figure(figsize=(5, 4))
    for cfg, sub in df.groupby("config"):
        h = np.stack(sub["loss_hist"].values)   # (seeds, epochs)
        mu, sd = h.mean(0), h.std(0)
        plt.plot(_smooth(mu), label=cfg)
        plt.fill_between(range(len(mu)),
                         _smooth(mu - sd), _smooth(mu + sd),
                         alpha=.15)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log)")
    plt.title("PARITY · Convergencia")
    plt.legend(title="hidden act")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/loss_curves.png", dpi=250)

    # ---------- TIME ------------------------------------------------
    stats = df.groupby("config")["time"].agg(["mean", "std"]).loc[order]

    plt.figure(figsize=(4, 4))
    sns.barplot(x=stats.index, y=stats["mean"], palette="viridis",
                yerr=stats["std"], capsize=.15, errorbar=None)
    plt.ylabel("Tiempo (s)")
    plt.xlabel("")
    plt.title("PARITY · Tiempo de entrenamiento")
    for i, (m, s) in enumerate(zip(stats["mean"], stats["std"])):
        plt.text(i, m + s + .05, f"{m:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/time_bar.png", dpi=250)

    print(f"[OK] Panel parity → {out_dir}/")


# ------------------------------------------------------------------ #
#  PANEL MNIST ────────────────────────────────────────────────────── #
def panel_mnist(df: pd.DataFrame, out_dir="plots/panel_mnist"):
    """
    Genera tres PNG en *out_dir*:
        • acc_box.png
        • loss_curves.png
        • time_bar.png
    El DataFrame debe contener:
        • 'config'     → literal p.ej.  'sigmoid-softmax'
        • 'acc', 'time', 'loss_hist' …
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── nos quedamos SOLO con las dos configuraciones deseadas ──────────
    mask = df["config"].str.match(r"^sigmoid-(sigmoid|softmax)$")
    df   = df.loc[mask].copy()

    if df.empty:
        raise ValueError("El DataFrame no contiene las configuraciones necesarias")

    # columna auxiliar = activación de salida
    df["out_act"] = df["config"].str.split("-").str[1]
    order = ["softmax", "sigmoid"]

    # helper para suavizar las curvas de loss
    _smooth = lambda y, k=7: np.convolve(y, np.ones(k) / k, mode="same")

    # ───────────────────── Accuracy (box + scatter) ─────────────────────
    plt.figure(figsize=(3.8, 4))
    sns.boxplot(data=df, x="out_act", y="acc",
                palette="viridis", order=order)
    sns.stripplot(data=df, x="out_act", y="acc",
                  color="k", size=3, alpha=.6, order=order)
    plt.title("MNIST · Accuracy final")
    plt.xlabel("Output activation")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/acc_box.png", dpi=250)
    plt.close()

    # ───────────────────── Curvas de pérdida ────────────────────────────
    plt.figure(figsize=(5, 4))
    for act, sub in df.groupby("out_act"):
        h         = np.stack(sub["loss_hist"].values)      # [runs, epochs]
        mean, std = h.mean(0), h.std(0)
        plt.plot(_smooth(mean), label=act)
        plt.fill_between(range(len(mean)),
                         _smooth(mean - std), _smooth(mean + std), alpha=.15)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log)")
    plt.title("MNIST · Convergencia")
    plt.legend(title="Output act")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/loss_curves.png", dpi=250)
    plt.close()

    # ───────────────────── Tiempo de entrenamiento ──────────────────────
    stats = df.groupby("out_act")["time"].agg(["mean", "std"]).loc[order]

    plt.figure(figsize=(3.8, 4))
    sns.barplot(x=stats.index, y=stats["mean"], palette="viridis",
                yerr=stats["std"], capsize=.15, errorbar=None)
    plt.ylabel("Tiempo (s)")
    plt.xlabel("Output activation")
    plt.title("MNIST · Tiempo de entrenamiento")
    for i, (m, s) in enumerate(zip(stats["mean"], stats["std"])):
        plt.text(i, m + s + .05, f"{m:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/time_bar.png", dpi=250)
    plt.close()

    print(f"[OK] Panel MNIST guardado en «{out_dir}/»")