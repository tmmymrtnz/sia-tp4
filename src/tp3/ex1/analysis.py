#!/usr/bin/env python3
import json
import tempfile
import itertools
import numpy as np
import matplotlib.pyplot as plt
import sys
from runner import run_experiment


def single_run(config: dict, X, Y, target_mse: float) -> int:
    """
    Ejecuta run_experiment con la configuración dada, captura stdout,
    y devuelve la época en que MSE <= target_mse.
    Si nunca ocurre, devuelve max_epochs.
    """
    import io, contextlib
    buf = io.StringIO()
    # Guardar config en archivo temporal
    tmp = tempfile.NamedTemporaryFile('w+', suffix='.json', delete=False)
    json.dump(config, tmp)
    tmp.flush()

    # Redirige stdout
    with contextlib.redirect_stdout(buf):
        run_experiment(tmp.name, X, Y)
    lines = buf.getvalue().splitlines()

    # Buscar primera época donde el MSE sea exactamente target_mse
    for line in lines:
        if line.startswith("Epoch") and "MSE" in line:
            parts = line.split()
            try:
                epoch = int(parts[1])
                mse = float(parts[-1])
            except (ValueError, IndexError):
                continue
            if mse == target_mse:
                return epoch
    return config.get('max_epochs', 0)


def main(config_path: str):
    # Carga configuración
    cfg = json.load(open(config_path))

    X = cfg['dataset']['X']
    Y = cfg['dataset']['Y']
    runs = cfg.get('runs', 1)
    max_epochs = cfg.get('max_epochs', 100)
    target_mse = cfg.get('target_mse', 0.0)

    # Hiperparámetros a barrer
    hyperparams = cfg['hyperparameters']
    keys = list(hyperparams.keys())
    combos = list(itertools.product(*(hyperparams[k] for k in keys)))

    labels, means, err_lo, err_hi = [], [], [], []

    for combo in combos:
        # Construir configuración de base
        base_cfg = {
            'learning_rate': None,
            'bias': None,
            'max_epochs': max_epochs,
        }
        for k, v in zip(keys, combo):
            base_cfg[k] = v

        # Ejecutar runs veces
        results = [single_run(base_cfg, X, Y, target_mse) for _ in range(runs)]
        mean = np.mean(results)
        mn = np.min(results)
        mx = np.max(results)

        labels.append(' | '.join(f"{k}={v}" for k, v in zip(keys, combo)))
        means.append(mean)
        err_lo.append(mean - mn)
        err_hi.append(mx - mean)

    # Graficar
    x = np.arange(len(labels))
    plt.figure(figsize=(10, 6))
    plt.bar(x, means, yerr=[err_lo, err_hi], capsize=5)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel(f"Épocas hasta MSE = {target_mse}")
    plt.title(f"Convergencia a MSE = {target_mse} en {runs} corridas")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python analysis.py <analysis.json>")
        sys.exit(1)
    main(sys.argv[1])