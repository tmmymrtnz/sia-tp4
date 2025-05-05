"""
Runner para el Ejercicio 3.3 – Discriminación de dígitos (0–9) sobre imágenes 7×5.
Uso:
    python src/ex3/runner_digit.py src/ex3/configs/digit.json

Se asume que el archivo de datos “TP3-ej3-digitos.txt” está ubicado en /data.
El archivo contiene 70 líneas (10 dígitos × 7 filas), cada fila con 5 valores 0/1 separados por espacios.
"""
import sys
import json
import numpy as np
from pathlib import Path

# --- importar red y trainer ---
sys.path.insert(0, "src")
from ex3.trainer import Trainer
from ex3.network import MLP


def load_digit_dataset(path: Path):
    lines = [ln.strip() for ln in path.open() if ln.strip()]
    # Deben haber 70 líneas: 10 dígitos × 7 filas
    if len(lines) % 7 != 0:
        raise ValueError(f"Número de líneas inválido: se esperaba múltiplo de 7, pero hay {len(lines)} líneas.")
    num_digits = len(lines) // 7
    X, y = [], []
    for idx in range(num_digits):
        digit_bits = []
        for row in lines[idx*7:(idx+1)*7]:
            row_bits = list(map(int, row.split()))
            if len(row_bits) != 5:
                raise ValueError(f"Cada fila debe tener 5 bits, pero se encontró {len(row_bits)} en la fila: '{row}'")
            digit_bits.extend(row_bits)
        X.append(digit_bits)
        # etiqueta original: dígito idx (0–9)
        y_onehot = np.zeros(10, dtype=float)
        y_onehot[idx] = 1.0
        y.append(y_onehot)
    return np.array(X, dtype=float), np.array(y, dtype=float)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python runner_digit.py <config.json>")
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    with cfg_path.open() as f:
        cfg = json.load(f)

    data_path = Path("data/TP3-ej3-digitos.txt")
    if not data_path.exists():
        print(f"Error: no se encuentra el archivo de datos en {data_path}")
        sys.exit(1)

    # cargar datos y etiquetas one-hot
    X, y = load_digit_dataset(data_path)

    # preparar activaciones: si falta dummy, prepender identidad
    layer_sizes = cfg.get("layer_sizes", [])
    activs = cfg.get("activations", [])
    if len(activs) == len(layer_sizes) - 1:
        activs = [""] + activs

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

    # evaluación
    y_hat = net.forward(X)  # (N,10)
    preds = np.argmax(y_hat, axis=1)
    trues = np.arange(len(X))
    accuracy = np.mean(preds == trues)
    print(f"\nAccuracy en todo el dataset: {accuracy:.4f}")

    print("\nPredicciones finales:")
    for idx, pred in enumerate(preds):
        print(f"Dígito {idx}: Predicción → {pred}  |  Verdadero → {idx}")

    # matriz de confusión
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(trues, preds)
    print("\nMatriz de confusión:")
    print(cm)