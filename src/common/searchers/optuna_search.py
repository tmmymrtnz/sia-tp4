#!/usr/bin/env python3
"""
Búsqueda de hiper-parámetros con Optuna para 3 objetivos:

    • digit      → discriminación 0-9 sobre matriz 7×5
    • parity     → par-impar sobre la misma matriz
    • mnist      → MNIST (28×28) cargado vía Keras

Ejecutar:
    python src/common/searchers/optuna_search.py [digit|parity|mnist]

Si no se pasa argumento, el objetivo default es 'digit'.
"""
import sys, json, optuna, numpy as np
from pathlib import Path
from copy import deepcopy

# ── tu stack de redes ------------------------------------------------ #
sys.path.insert(0, "src")
from common.perceptrons.multilayer.trainer   import Trainer
from common.perceptrons.multilayer.network   import MLP
from ex3.runner_digit   import load_digit_dataset
from ex3.runner_parity  import load_parity_dataset
from ex4.runner_mnist   import load_mnist_keras

# ── espacios de búsqueda globales ----------------------------------- #
H_ACTIVATIONS = ["tanh", "sigmoid", "relu", "identity", "step"]
OPTIMIZERS    = ["adam", "momentum", "sgd"]
LOSSES_ALL    = ["mse", "cross_entropy", "bce"]

DATA_DIR = Path("data")
NOISE_FILES = [DATA_DIR / f"noisy{i}.txt" for i in range(1, 5)]
BASE_FILE   = DATA_DIR / "TP3-ej3-digitos.txt"

# -------------------------------------------------------------------- #
# 1. Carga de datos según objetivo
# -------------------------------------------------------------------- #
def load_data(task):
    if task == "digit":
        X, y = load_digit_dataset(BASE_FILE)
        noise_sets = [load_digit_dataset(p) for p in NOISE_FILES if p.exists()]
        return (X, y), noise_sets, 35, 10, "softmax", ["mse", "cross_entropy"]
    if task == "parity":
        X, y = load_parity_dataset(BASE_FILE)
        noise_sets = [load_parity_dataset(p) for p in NOISE_FILES if p.exists()]
        return (X, y), noise_sets, 35, 1, "sigmoid", ["mse", "bce"]
    if task == "mnist":
        X_tr, y_tr, X_te, y_te = load_mnist_keras()
        return (X_tr, y_tr, X_te, y_te), [], 784, 10, "softmax", ["mse", "cross_entropy"]
    raise ValueError(f"Tarea desconocida: {task}")

# -------------------------------------------------------------------- #
# 2. Construcción del modelo para Optuna
# -------------------------------------------------------------------- #
def build_config(trial, in_dim, out_dim, out_act, allowed_losses):
    n_hidden = trial.suggest_int("n_hidden", 1, 3)
    hidden_sizes = [trial.suggest_categorical(f"n_units_{i}", [8, 16, 32, 64])
                    for i in range(n_hidden)]
    hidden_acts  = [trial.suggest_categorical(f"act_{i}", H_ACTIVATIONS)
                    for i in range(n_hidden)]

    layer_sizes = [in_dim] + hidden_sizes + [out_dim]
    activations = [""] + hidden_acts + [out_act]

    cfg = {
        "layer_sizes"  : layer_sizes,
        "activations"  : activations,
        "dropout_rate" : trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        "loss"         : trial.suggest_categorical("loss", allowed_losses),
        "optimizer"    : trial.suggest_categorical("optim", OPTIMIZERS),
        "optim_kwargs" : { "learning_rate":
                           trial.suggest_float("lr", 1e-4, 1e-1, log=True) },
        "batch_size"   : trial.suggest_categorical("batch", [5, 10, 20, 35, 128]),
        "max_epochs"   : 2000
    }
    return cfg

# -------------------------------------------------------------------- #
# 3. Evaluación: accuracy medio (base + ruido/test)
# -------------------------------------------------------------------- #
def evaluate(cfg, task, data_main, noise_sets):
    # --- instanciar red + trainer
    net = MLP(cfg["layer_sizes"], cfg["activations"], cfg["dropout_rate"])
    trainer = Trainer(
        net            = net,
        loss_name      = cfg["loss"],
        optimizer_name = cfg["optimizer"],
        optim_kwargs   = cfg["optim_kwargs"],
        batch_size     = cfg["batch_size"],
        max_epochs     = cfg["max_epochs"]
    )

    if task == "mnist":
        X_tr, y_tr, X_te, y_te = data_main
        trainer.fit(X_tr, y_tr)
        preds = np.argmax(net.forward(X_te), axis=1)
        trues = np.argmax(y_te,           axis=1)
        return float(np.mean(preds == trues))

    # digit / parity
    X, y = data_main
    trainer.fit(X, y)

    def _acc(a, b):
        if b.shape[1] == 1:                      # paridad
            return float(np.mean((a > .5).astype(int) == b.astype(int)))
        preds = np.argmax(a, axis=1)            # dígitos
        trues = np.argmax(b, axis=1)
        return float(np.mean(preds == trues))

    accs = [_acc(net.forward(X), y)]
    for Xn, yn in noise_sets:
        accs.append(_acc(net.forward(Xn), yn))
    return float(np.mean(accs))

# -------------------------------------------------------------------- #
# 4. Función objetivo Optuna
# -------------------------------------------------------------------- #
def make_objective(task, data_main, noise_sets, in_dim, out_dim,
                   out_act, allowed_losses):
    def _objective(trial):
        cfg = build_config(trial, in_dim, out_dim, out_act, allowed_losses)
        acc = evaluate(cfg, task, data_main, noise_sets)
        trial.set_user_attr("config", deepcopy(cfg))
        trial.set_user_attr("avg_accuracy", acc)
        return acc
    return _objective

# -------------------------------------------------------------------- #
# 5. Main
# -------------------------------------------------------------------- #
def main():
    task = sys.argv[1].lower() if len(sys.argv) > 1 else "digit"
    if task not in {"digit", "parity", "mnist"}:
        print(__doc__); sys.exit(1)

    # --- datos & meta-info
    data_main, noise_sets, in_dim, out_dim, out_act, losses = load_data(task)

    # --- búsqueda Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(
        make_objective(task, data_main, noise_sets,
                       in_dim, out_dim, out_act, losses),
        n_trials=30
    )

    best = study.best_trial
    print("\n=== Best Trial ===")
    print(f"Average accuracy: {best.value:.4f}")
    print(json.dumps(best.user_attrs["config"], indent=2))

    # --- guardar resultado
    out_path = Path(f"configs/ex3/best_optuna_config_{task}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "accuracy" : best.user_attrs["avg_accuracy"],
            "config"   : best.user_attrs["config"]
        }, f, indent=2)
    print(f"\nGuardado en {out_path}")

# -------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
