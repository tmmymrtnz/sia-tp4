"""
Runner para el Ejercicio 3.3 – Discriminación de dígitos (0–9) sobre imágenes 7×5.
Uso:
    python src/ex3/runner_digit.py src/ex3/configs/digit.json

Se asume que el archivo de datos “TP3-ej3-digitos.txt” está ubicado en data/,
y que los archivos de prueba con ruido “noisy1.txt”, “noisy2.txt” y “noisy3.txt”
también están en ese mismo directorio.
"""
import sys
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# --- importar red y trainer ---
sys.path.insert(0, "src")
from ex3.trainer import Trainer
from ex3.network import MLP


def load_digit_dataset(path: Path):
    lines = [ln.strip() for ln in path.open() if ln.strip()]
    if len(lines) % 7 != 0:
        raise ValueError(f"Número de líneas inválido: se esperaba múltiplo de 7, pero hay {len(lines)} líneas.")
    num_digits = len(lines) // 7
    X, y = [], []
    for idx in range(num_digits):
        bits = []
        for row in lines[idx*7:(idx+1)*7]:
            row_bits = list(map(int, row.split()))
            if len(row_bits) != 5:
                raise ValueError(f"Cada fila debe tener 5 bits, pero se encontró {len(row_bits)} en '{row}'")
            bits.extend(row_bits)
        X.append(bits)
        onehot = np.zeros(10, dtype=float)
        onehot[idx] = 1.0
        y.append(onehot)
    return np.array(X, dtype=float), np.array(y, dtype=float)

def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels=range(10), title="Confusion Matrix", show_heatmap=False):
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    print(f"\n{title}")
    header = "     " + " ".join(f"{l:^5}" for l in labels)
    print(header)
    print("    " + "-" * (6 * len(labels)))
    for i, row in enumerate(matrix):
        row_str = " ".join(f"{val:^5}" for val in row)
        print(f"{i:^3} | {row_str}")

    if show_heatmap:
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

def evaluate_dataset(net, X, y_true, prefix="Dataset", show_heatmap=False):
    y_hat = net.forward(X)
    preds = np.argmax(y_hat, axis=1)
    trues = np.argmax(y_true, axis=1)
    acc = np.mean(preds == trues)
    print("--" * 40)
    print(f"\n[{prefix}] Accuracy: {acc:.4f}")
    for i, (p, prob, t) in enumerate(zip(preds, y_hat, trues)):
        print(f"{prefix} idx {i}: Pred={p} (p={prob[p]:.3f}) | True={t}")
    print_confusion_matrix(trues, preds, title=f"[{prefix}] – Confusion Matrix", show_heatmap=show_heatmap)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python runner_digit.py <config.json>")
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    with cfg_path.open() as f:
        cfg = json.load(f)

    data_dir = Path("data")
    main_file = data_dir / "TP3-ej3-digitos.txt"
    if not main_file.exists():
        print(f"Error: no se encuentra el archivo base en {main_file}")
        sys.exit(1)

    # cargar datos originales
    X, y = load_digit_dataset(main_file)

    # preparar activaciones (añadir dummy si falta)
    layer_sizes = cfg.get("layer_sizes", [])
    activs = cfg.get("activations", [])
    if len(activs) == len(layer_sizes) - 1:
        activs = [""] + activs

    heatmap = cfg.get("heatmap", False)

    # inicializar red y trainer
    net = MLP(layer_sizes, activs)
    trainer = Trainer(
        net=net,
        loss_name=cfg.get("loss", "cross_entropy"),
        optimizer_name=cfg.get("optimizer", "adam"),
        optim_kwargs=cfg.get("optim_kwargs", {}),
        batch_size=cfg.get("batch_size", len(X)),
        max_epochs=cfg.get("max_epochs", 1000)
    )

    # entrenamiento
    trainer.fit(X, y)

    # evaluación sobre el dataset original
    evaluate_dataset(net, X, y, prefix="Original", show_heatmap=heatmap)

    # evaluación sobre archivos con ruido
    for i in range(1, 5):
        noisy_path = data_dir / f"noisy{i}.txt"
        if not noisy_path.exists():
            print(f"Advertencia: {noisy_path} no existe, omitiendo.")
            continue
        Xn, yn = load_digit_dataset(noisy_path)
        evaluate_dataset(net, Xn, yn, prefix=f"Noisy{i}", show_heatmap=heatmap)

