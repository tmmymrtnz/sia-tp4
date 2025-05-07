import optuna
import numpy as np
from pathlib import Path
from copy import deepcopy
import sys
import json

sys.path.insert(0, "src")
from common.perceptrons.multilayer.trainer import Trainer
from common.perceptrons.multilayer.network import MLP
from ex3.runner_digit import load_digit_dataset

# Global dataset paths
DATA_DIR = Path("data")
FILES = [
    DATA_DIR / "TP3-ej3-digitos.txt",
    DATA_DIR / "noisy1.txt",
    DATA_DIR / "noisy2.txt",
    DATA_DIR / "noisy3.txt",
    DATA_DIR / "noisy4.txt"
]

# Hyperparameter space
ACTIVATIONS = ["tanh", "sigmoid", "relu"]
OPTIMIZERS = ["adam", "momentum", "sgd"]
LOSSES = ["mse", "cross_entropy"]

INPUT_SIZE = 35
OUTPUT_SIZE = 10

def build_model(trial):
    num_hidden = trial.suggest_int("num_hidden_layers", 1, 3)
    neurons = [trial.suggest_categorical(f"n_units_l{i}", [8, 16, 32, 64]) for i in range(num_hidden)]
    activs = [""] + [trial.suggest_categorical(f"act_l{i}", ACTIVATIONS) for i in range(num_hidden)]

    layer_sizes = [INPUT_SIZE] + neurons + [OUTPUT_SIZE]
    activs.append("sigmoid")  # output layer

    dropout = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1)
    loss = trial.suggest_categorical("loss", LOSSES)
    optimizer = trial.suggest_categorical("optimizer", OPTIMIZERS)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [5, 10, 20, 35])

    config = {
        "layer_sizes": layer_sizes,
        "activations": activs,
        "dropout_rate": dropout,
        "loss": loss,
        "optimizer": optimizer,
        "optim_kwargs": {"learning_rate": lr},
        "batch_size": batch_size,
        "max_epochs": 2000
    }
    return config

def evaluate_model(cfg):
    net = MLP(cfg["layer_sizes"], cfg["activations"], cfg["dropout_rate"])
    trainer = Trainer(
        net=net,
        loss_name=cfg["loss"],
        optimizer_name=cfg["optimizer"],
        optim_kwargs=cfg["optim_kwargs"],
        batch_size=cfg["batch_size"],
        max_epochs=cfg["max_epochs"]
    )

    X_base, y_base = load_digit_dataset(FILES[0])
    trainer.fit(X_base, y_base)

    accs = []
    for path in FILES:
        if not path.exists():
            continue
        Xn, yn = load_digit_dataset(path)
        y_hat = net.forward(Xn)
        preds = np.argmax(y_hat, axis=1)
        trues = np.argmax(yn, axis=1)
        acc = np.mean(preds == trues)
        accs.append(acc)

    return np.mean(accs)

def objective(trial):
    cfg = build_model(trial)
    avg_acc = evaluate_model(cfg)
    trial.set_user_attr("config", deepcopy(cfg))
    trial.set_user_attr("avg_accuracy", avg_acc)
    return avg_acc

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("\n=== Best Trial ===")
    best = study.best_trial
    print(f"Average Accuracy: {best.value:.4f}")
    print(json.dumps(best.user_attrs["config"], indent=2))

    # Save only the best configuration
    results_path = "configs/ex3/best_optuna_config.json"
    with open(results_path, "w") as f:
        json.dump({
            "accuracy": best.user_attrs["avg_accuracy"],
            "config": best.user_attrs["config"]
        }, f, indent=2)
    print(f"\nSaved best configuration to {results_path}")

if __name__ == "__main__":
    main()