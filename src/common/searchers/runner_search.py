"""
Random Hyperparameter Search for Digit Recognition (Exercise 3.3)
Usage:
    python src/ex3/runner_random_search.py
"""

import json
import random
import numpy as np
import csv
from pathlib import Path
from copy import deepcopy
import sys

sys.path.insert(0, "src")
from common.perceptrons.multilayer.trainer import Trainer
from common.perceptrons.multilayer.network import MLP
from ex3.runner_digit import load_digit_dataset, evaluate_dataset


# --- SEARCH SPACE CONFIGURATION ---
ACTIVATIONS = ["tanh", "sigmoid", "relu"]
OPTIMIZERS = ["adam", "momentum", "sgd"]
LOSS_FUNCS = ["mse", "cross_entropy"]

SEARCH_SPACE = {
    "num_hidden_layers": [1, 2, 3],
    "neurons_per_layer": [8, 16, 32, 64],
    "activations": ACTIVATIONS,
    "dropout_rate": [0.0, 0.2, 0.3, 0.5],
    "learning_rate": [1e-4, 1e-3, 1e-2, 1e-1],
    "optimizer": OPTIMIZERS,
    "loss": LOSS_FUNCS,
    "batch_size": [5, 10, 20, 35],
}


def random_config(input_size=35, output_size=10):
    num_hidden = random.choice(SEARCH_SPACE["num_hidden_layers"])
    layer_sizes = [input_size]
    activs = [""]  # input layer has no activation

    for _ in range(num_hidden):
        neurons = random.choice(SEARCH_SPACE["neurons_per_layer"])
        layer_sizes.append(neurons)
        activs.append(random.choice(ACTIVATIONS))

    layer_sizes.append(output_size)
    activs.append("sigmoid")  # final layer for multi-class

    return {
        "layer_sizes": layer_sizes,
        "activations": activs,
        "dropout_rate": random.choice(SEARCH_SPACE["dropout_rate"]),
        "loss": random.choice(SEARCH_SPACE["loss"]),
        "optimizer": random.choice(SEARCH_SPACE["optimizer"]),
        "optim_kwargs": {
            "learning_rate": random.choice(SEARCH_SPACE["learning_rate"])
        },
        "batch_size": random.choice(SEARCH_SPACE["batch_size"]),
        "max_epochs": 2000
    }


def run_search(n_trials=30, output_csv="search_results.csv"):
    data_dir = Path("data")
    dataset_files = [data_dir / "TP3-ej3-digitos.txt"] + [
        data_dir / f"noisy{i}.txt" for i in range(1, 5)
    ]
    datasets = []

    for path in dataset_files:
        if path.exists():
            X, y = load_digit_dataset(path)
            datasets.append((path.name, X, y))
        else:
            print(f"Warning: {path} not found, skipping.")
    
    if not datasets:
        raise RuntimeError("No datasets found to evaluate!")

    best_avg_acc = 0.0
    best_configs = []

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial", "avg_accuracy", "accuracies_per_dataset",
            "layer_sizes", "activations", "dropout_rate",
            "loss", "optimizer", "learning_rate", "batch_size"
        ])

        for i in range(n_trials):
            print(f"\n=== Trial {i + 1}/{n_trials} ===")
            cfg = random_config()

            net = MLP(cfg["layer_sizes"], cfg["activations"], cfg["dropout_rate"])
            trainer = Trainer(
                net=net,
                loss_name=cfg["loss"],
                optimizer_name=cfg["optimizer"],
                optim_kwargs=cfg["optim_kwargs"],
                batch_size=cfg["batch_size"],
                max_epochs=cfg["max_epochs"]
            )

            # Train only on original (clean) dataset
            trainer.fit(datasets[0][1], datasets[0][2])  # (X, y) of the clean dataset

            accuracies = []
            for name, X_eval, y_eval in datasets:
                y_hat = net.forward(X_eval)
                preds = np.argmax(y_hat, axis=1)
                trues = np.argmax(y_eval, axis=1)
                acc = np.mean(preds == trues)
                accuracies.append(acc)
                print(f"{name:<20} Accuracy: {acc:.4f}")

            avg_acc = np.mean(accuracies)
            print(f"--> Avg Accuracy: {avg_acc:.4f}")

            writer.writerow([
                i + 1, avg_acc, accuracies,
                cfg["layer_sizes"], cfg["activations"], cfg["dropout_rate"],
                cfg["loss"], cfg["optimizer"], cfg["optim_kwargs"]["learning_rate"],
                cfg["batch_size"]
            ])

            if avg_acc > best_avg_acc + 1e-4:
                best_avg_acc = avg_acc
                best_configs = [deepcopy(cfg)]
            elif abs(avg_acc - best_avg_acc) <= 1e-4:
                best_configs.append(deepcopy(cfg))

    print("\n=== Best Configurations (Tied) ===")
    for idx, cfg in enumerate(best_configs, 1):
        print(f"\n--- Config {idx} ---")
        print(json.dumps(cfg, indent=2))
    print(f"\nBest Avg Accuracy: {best_avg_acc:.4f}")


if __name__ == "__main__":
    run_search(n_trials=30)