"""
Runner para el Ejercicio 3.2 – Discriminación de paridad sobre dígitos 7×5.
Uso:
    python src/ex3/runner_parity.py src/ex3/configs/parity.json

Se asume que el archivo de datos “TP3-ej3-digitos.txt” está ubicado en /data,
y los archivos de prueba con ruido “noisy1.txt”, “noisy2.txt” y “noisy3.txt” en data/.
El archivo base contiene 70 líneas (10 dígitos × 7 filas), cada fila con 5 valores 0/1.
"""
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

# --- importar red y trainer ---
sys.path.insert(0, "src")
from common.perceptrons.multilayer.trainer import Trainer
from common.perceptrons.multilayer.network import MLP


def load_parity_dataset(path: Path):
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
        y.append([idx % 2])
    return np.array(X, dtype=float), np.array(y, dtype=float)

def print_confusion_matrix(y_true, y_pred, label_prefix=""):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n[{label_prefix}] Confusion Matrix:")
    print("          Pred 0    Pred 1")
    print(f"True 0   {cm[0,0]:>7}   {cm[0,1]:>7}")
    print(f"True 1   {cm[1,0]:>7}   {cm[1,1]:>7}\n")

def evaluate_and_print(net, X, y_true, label_prefix="Dataset"):
    y_hat = net.forward(X)
    y_pred = (y_hat > 0.5).astype(int)
    accuracy = np.mean(y_pred == y_true)
    print("--" * 40)
    print(f"\n[{label_prefix}] Accuracy: {accuracy:.4f}")
    print_confusion_matrix(y_true, y_pred, label_prefix=label_prefix)
    for i, (pred, prob, true) in enumerate(zip(y_pred, y_hat, y_true)):
        print(f"{label_prefix} idx {i}: Pred={pred[0]} (p={prob[0]:.3f}) | True={true[0]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python runner_parity.py <config.json>")
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    with cfg_path.open() as f:
        cfg = json.load(f)

    # ruta base /data
    base_path = Path("data")
    main_file = base_path / "TP3-ej3-digitos.txt"
    if not main_file.exists():
        print(f"Error: no se encuentra el archivo base en {main_file}")
        sys.exit(1)

    # cargar datos originales
    X, y = load_parity_dataset(main_file)

    # inicializar modelo y trainer
    net = MLP(cfg["layer_sizes"], cfg["activations"], cfg.get("dropout_rate", 0.0))
    trainer = Trainer(
        net=net,
        loss_name=cfg["loss"],
        optimizer_name=cfg["optimizer"],
        optim_kwargs=cfg.get("optim_kwargs", {}),
        batch_size=cfg.get("batch_size", len(X)),
        max_epochs=cfg.get("max_epochs", 1000)
    )

    # entrenamiento
    trainer.fit(X, y)

    # evaluación sobre dataset original
    evaluate_and_print(net, X, y, label_prefix="Original")

    # evaluar archivos con ruido
    for i in range(1, 5):
        noisy_file = base_path / f"noisy{i}.txt"
        if not noisy_file.exists():
            print(f"Advertencia: {noisy_file} no existe, omitiendo.")
            continue
        Xn, yn = load_parity_dataset(noisy_file)
        evaluate_and_print(net, Xn, yn, label_prefix=f"Noisy{i}")