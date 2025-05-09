"""
Runner for Hidden Layer Size Hypothesis Test
Evaluates: “Increasing the number of hidden neurons improves performance on digit classification
but hurts generalization with noise.”

Usage:
    python src/analysis/hidden.py
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, "src")
from common.perceptrons.multilayer.trainer import Trainer
from common.perceptrons.multilayer.network import MLP

# Define layer configurations to test
HIDDEN_LAYER_SIZES = [4, 16, 32, 64]
INPUT_SIZE = 35
OUTPUT_SIZE = 10
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]

def load_digit_dataset(path: Path):
    lines = [ln.strip() for ln in path.open() if ln.strip()]
    num = len(lines) // 7
    X, y = [], []
    for idx in range(num):
        bits = []
        for row in lines[idx*7:(idx+1)*7]:
            bits.extend(map(int, row.split()))
        X.append(bits)
        onehot = np.zeros(10, dtype=float)
        onehot[idx%10] = 1.0
        y.append(onehot)
    return np.array(X, dtype=float), np.array(y, dtype=float)

def augment_digits(X, y, copies_per_digit, noise_p):
    X_aug, y_aug = [], []
    for x_vec, label in zip(X, y):
        X_aug.append(x_vec); y_aug.append(label)
        for _ in range(copies_per_digit):
            mask    = np.random.rand(x_vec.size) < noise_p
            x_noisy = np.where(mask, 1 - x_vec, x_vec)
            X_aug.append(x_noisy); y_aug.append(label)
    return np.array(X_aug), np.array(y_aug)

def evaluate(net, X, y):
    y_hat = net.forward(X)
    preds = np.argmax(y_hat, axis=1)
    trues = np.argmax(y, axis=1)
    acc = np.mean(preds == trues)
    return acc

def main():
    data_dir = Path("data")
    base_file = data_dir / "TP3-ej3-digitos.txt"
    clean_large_file = data_dir / "large_clean.txt"

    X_train, y_train = load_digit_dataset(base_file)
    X_clean, y_clean = load_digit_dataset(clean_large_file)
    noisy_datasets = [
        augment_digits(X_clean, y_clean, copies_per_digit=1, noise_p=noise)
        for noise in NOISE_LEVELS
    ]

    NUM_RUNS = 5  # Number of repeated runs for averaging

    avg_results = {noise: [] for noise in NOISE_LEVELS}

    for size in HIDDEN_LAYER_SIZES:
        print(f"\n=== Hidden Layer Size: {size} ===")
        accs_clean = []
        accs_by_noise = {noise: [] for noise in NOISE_LEVELS}

        for run in range(NUM_RUNS):
            print(f"--- Run {run + 1}/{NUM_RUNS} ---")
            cfg = {
                "layer_sizes": [INPUT_SIZE, size, OUTPUT_SIZE],
                "activations": ["", "tanh", "sigmoid"],
                "dropout_rate": 0.0,
                "loss": "cross_entropy",
                "optimizer": "adam",
                "optim_kwargs": {"learning_rate": 0.01},
                "batch_size": 10,
                "max_epochs": 1000
            }

            net = MLP(cfg["layer_sizes"], cfg["activations"], cfg["dropout_rate"])
            trainer = Trainer(
                net=net,
                loss_name=cfg["loss"],
                optimizer_name=cfg["optimizer"],
                optim_kwargs=cfg["optim_kwargs"],
                batch_size=cfg["batch_size"],
                max_epochs=cfg["max_epochs"]
            )
            trainer.fit(X_train, y_train)

            acc_clean = evaluate(net, X_clean, y_clean)
            accs_clean.append(acc_clean)

            for (noise, (Xn, yn)) in zip(NOISE_LEVELS, noisy_datasets):
                acc = evaluate(net, Xn, yn)
                accs_by_noise[noise].append(acc)

        mean_clean = np.mean(accs_clean)
        std_clean = np.std(accs_clean)

        print(f"Mean Accuracy (Clean): {mean_clean:.4f} ± {std_clean:.4f}")

        for noise in NOISE_LEVELS:
            mean_noisy = np.mean(accs_by_noise[noise])
            std_noisy = np.std(accs_by_noise[noise])
            avg_results[noise].append((size, mean_noisy, std_noisy))
            print(f"Noise {noise:.1f} Accuracy: {mean_noisy:.4f} ± {std_noisy:.4f}")

    plt.figure(figsize=(10, 6))
    for noise, results in avg_results.items():
        sizes, means, stds = zip(*results)
        label = f"Noise {noise:.1f}"
        plt.errorbar(sizes, means, yerr=stds, label=label, marker="o", capsize=4)

    plt.title("Accuracy vs Hidden Layer Size at Various Noise Levels")
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Accuracy")
    plt.xticks(HIDDEN_LAYER_SIZES)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()