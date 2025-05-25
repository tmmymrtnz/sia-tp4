#!/usr/bin/env python3
# ------------------------------------------------------------
#  src/ex3/runner_mnist.py
# ------------------------------------------------------------
"""
Runner para el Ejercicio – Clasificación de dígitos MNIST
usando el loader nativo de Keras (ya NO requiere CSV).

Uso:
    python src/ex3/runner_mnist.py <config.json>

El JSON de configuración mantiene la misma estructura que antes.
"""
import sys
import json
from pathlib import Path
import numpy as np

# --- Keras / TensorFlow ---
import tensorflow as tf
from tensorflow import keras

# --- tu stack “core” ---
sys.path.insert(0, "src")
from common.perceptrons.multilayer.trainer import Trainer
from common.perceptrons.multilayer.network  import MLP


# ------------------------------------------------------------------ #
#  DATASET
# ------------------------------------------------------------------ #
def load_mnist_keras():
    """
    Descarga (la primera vez) y devuelve:
        X_train, y_train  →  (n_train, 784), (n_train, 10)
        X_test,  y_test   →  (n_test,  784), (n_test,  10)
    Las imágenes vienen en uint8 [0,255] y se normalizan a [0,1].
    """
    (x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()  # (n, 28, 28)
    # Normalizar y aplanar
    X_train = x_tr.astype("float32").reshape(-1, 28 * 28) / 255.0
    X_test  = x_te.astype("float32").reshape(-1, 28 * 28) / 255.0

    # One-hot
    y_train = keras.utils.to_categorical(y_tr, 10).astype("float32")
    y_test  = keras.utils.to_categorical(y_te, 10).astype("float32")
    return X_train, y_train, X_test, y_test


# ------------------------------------------------------------------ #
#  MÉTRICAS / UTILS
# ------------------------------------------------------------------ #
def print_confusion_matrix(y_true, y_pred, labels=range(10)):
    m = len(labels)
    cm = np.zeros((m, m), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    header = "     " + " ".join(f"{i:^5}" for i in labels)
    sep    = "    " + "-" * (6 * m)
    print(header)
    print(sep)
    for i, row in enumerate(cm):
        print(f"{i:^3} | " + " ".join(f"{v:^5}" for v in row))


def evaluate_dataset(net, X, y_true):
    """
    Evalúa la red en (X, y_true), imprime accuracy,
    primeras 10 predicciones y la matriz de confusión.
    """
    y_prob = net.forward(X)                 # (n,10)
    preds  = np.argmax(y_prob, axis=1)
    trues  = np.argmax(y_true, axis=1)

    acc = np.mean(preds == trues)
    print(f"\n=== Test Accuracy: {acc:.4f} ===\n")

    print("Primeras 10 predicciones:")
    for i in range(min(10, len(preds))):
        p = preds[i]; t = trues[i]; prob = y_prob[i, p]
        print(f"  idx {i:3d}: True={t}  Pred={p}  (p={prob:.3f})")

    print("\nConfusion Matrix:")
    print_confusion_matrix(trues, preds)


# ------------------------------------------------------------------ #
#  MAIN
# ------------------------------------------------------------------ #
def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    if not cfg_path.exists():
        print("Error: el archivo de configuración no existe.")
        sys.exit(1)

    # ------------ Config ------------
    cfg = json.loads(cfg_path.read_text())

    # ------------ Datos -------------
    print("Descargando / cargando MNIST con Keras…")
    X_train, y_train, X_test, y_test = load_mnist_keras()

    # ------------ Red ---------------
    layer_sizes = cfg["layer_sizes"]
    activs      = cfg["activations"]
    if len(activs) == len(layer_sizes) - 1:
        activs = [""] + activs         # dummy p/ capa de entrada

    net = MLP(layer_sizes, activs)

    # ------------ Entrenador --------
    trainer = Trainer(
        net            = net,
        loss_name      = cfg["loss"],
        optimizer_name = cfg["optimizer"],
        optim_kwargs   = cfg.get("optim_kwargs", {}),
        batch_size     = cfg.get("batch_size", 128),
        max_epochs     = cfg.get("max_epochs", 20)
    )

    # ------------ Entrenamiento -----
    print("Entrenando…")
    trainer.fit(X_train, y_train)

    # ------------ Evaluación --------
    print("\nEvaluando en el test set…")
    evaluate_dataset(net, X_test, y_test)


if __name__ == "__main__":
    main()
