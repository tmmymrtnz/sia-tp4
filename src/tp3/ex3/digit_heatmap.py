import sys
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# --- importar red y trainer ---
sys.path.insert(0, "src")
from common.perceptrons.multilayer.trainer import Trainer
from common.perceptrons.multilayer.network import MLP

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
        onehot[idx % 10] = 1.0
        y.append(onehot)
    return np.array(X, dtype=float), np.array(y, dtype=float)

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels=range(10), title="Confusion Matrix"):
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def evaluate_and_plot(net, X, y, label):
    y_hat = net.forward(X)
    preds = np.argmax(y_hat, axis=1)
    trues = np.argmax(y, axis=1)
    acc = np.mean(preds == trues)
    print(f"\n== {label} Evaluation ==")
    print(f"Accuracy: {acc:.4f}")
    plot_confusion_matrix(trues, preds, title=f"{label} - Confusion Matrix")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python runner_digit_heatmap.py <config.json>")
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    with cfg_path.open() as f:
        cfg = json.load(f)

    data_dir = Path("data")
    main_file = data_dir / "TP3-ej3-digitos.txt"
    large_file = data_dir / "large_data.txt"

    if not main_file.exists() or not large_file.exists():
        print("Error: no se encuentran los archivos requeridos.")
        sys.exit(1)

    X_train, y_train = load_digit_dataset(main_file)
    X_test, y_test = load_digit_dataset(large_file)

    layer_sizes = cfg.get("layer_sizes", [])
    activs = cfg.get("activations", [])
    if len(activs) == len(layer_sizes) - 1:
        activs = [""] + activs

    net = MLP(layer_sizes, activs)
    trainer = Trainer(
        net=net,
        loss_name=cfg.get("loss", "cross_entropy"),
        optimizer_name=cfg.get("optimizer", "adam"),
        optim_kwargs=cfg.get("optim_kwargs", {}),
        batch_size=cfg.get("batch_size", len(X_train)),
        max_epochs=cfg.get("max_epochs", 1000)
    )

    trainer.fit(X_train, y_train)
    evaluate_and_plot(net, X_train, y_train, label="Original")
    evaluate_and_plot(net, X_test, y_test, label="Large")
