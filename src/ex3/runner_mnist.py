#!/usr/bin/env python3
"""
Runner para el Ejercicio – Clasificación de dígitos MNIST desde CSV.
Uso:
    python src/ex3/runner_mnist.py <train.csv> <test.csv> <config.json>

Se asume que los CSV tienen formato:
    label,pixel0,pixel1,...,pixel783
con una fila de encabezado.
"""
import sys
import json
import numpy as np
from pathlib import Path

# --- importar red y trainer ---
sys.path.insert(0, "src")
from ex3.trainer import Trainer
from ex3.network import MLP

def load_csv_dataset(path: Path):
    """
    Carga un CSV type MNIST:
      - Primera columna: etiqueta 0–9
      - Columnas 1–784: pixeles 0–255
      - retorna X normalizado [0,1] de shape (n,784),
        y one-hot de shape (n,10).
    """
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Archivo inválido: {path}")
    labels = data[:,0].astype(int)
    X = data[:,1:] / 255.0
    n = len(labels)
    y = np.zeros((n, 10), dtype=float)
    y[np.arange(n), labels] = 1.0
    return X, y

def print_confusion_matrix(y_true, y_pred, labels=range(10)):
    """
    Imprime matrix de confusión en consola.
    """
    m = len(labels)
    cm = np.zeros((m,m), dtype=int)
    for t,p in zip(y_true, y_pred):
        cm[t,p] += 1
    header = "     " + " ".join(f"{i:^5}" for i in labels)
    sep    = "    " + "-"*(6*m)
    print(header)
    print(sep)
    for i,row in enumerate(cm):
        print(f"{i:^3} | " + " ".join(f"{v:^5}" for v in row))

def evaluate_dataset(net, X, y_true):
    """
    Evalúa net en X, y_true (one-hot), imprime accuracy,
    confusion matrix y primeras predicciones.
    """
    y_prob = net.forward(X)          # (n,10)
    preds  = np.argmax(y_prob, axis=1)
    trues  = np.argmax(y_true, axis=1)
    acc = np.mean(preds == trues)
    print(f"\n=== Test Accuracy: {acc:.4f} ===\n")
    print("Primeras 10 predicciones:")
    for i in range(min(10, len(preds))):
        p = preds[i]; t = trues[i]; prob = y_prob[i,p]
        print(f"  idx {i:3d}: True={t:1d}  Pred={p:1d}  (p={prob:.3f})")
    print("\nConfusion Matrix:")
    print_confusion_matrix(trues, preds)

def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    train_csv, test_csv, cfg_path = map(Path, sys.argv[1:])
    if not train_csv.exists() or not test_csv.exists() or not cfg_path.exists():
        print("Error: uno de los archivos no existe.")
        sys.exit(1)

    # cargar config
    cfg = json.loads(cfg_path.read_text())

    # cargar datasets
    print("Cargando datos de entrenamiento…")
    X_train, y_train = load_csv_dataset(train_csv)
    print("Cargando datos de prueba…")
    X_test,  y_test  = load_csv_dataset(test_csv)

    # preparar activaciones (añadir dummy si falta)
    layer_sizes = cfg["layer_sizes"]
    activs = cfg["activations"]
    if len(activs) == len(layer_sizes) - 1:
        activs = [""] + activs

    # inicializar red y trainer
    net = MLP(layer_sizes, activs)
    trainer = Trainer(
        net=net,
        loss_name=cfg["loss"],
        optimizer_name=cfg["optimizer"],
        optim_kwargs=cfg.get("optim_kwargs", {}),
        batch_size=cfg.get("batch_size", 128),
        max_epochs=cfg.get("max_epochs", 20)
    )

    # entrenamiento
    print("Entrenando…")
    trainer.fit(X_train, y_train)

    # evaluación
    print("\nEvaluando en test set…")
    evaluate_dataset(net, X_test, y_test)

if __name__ == "__main__":
    main()
