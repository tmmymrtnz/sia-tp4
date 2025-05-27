import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
sys.path.insert(0, "src")
from tp4.ex1.runner_oja import OjaNeuron, load_data, standardize, sample_weights_from_data


def compare_learning_rates(lrs, epochs, sample=True):
    data_path = Path("data/tp4/europe.csv")
    countries, X_df = load_data(data_path)
    X_std, _ = standardize(X_df)

    from sklearn.decomposition import PCA
    X_pca = PCA(n_components=1).fit_transform(X_std).flatten()
    sorted_idx = np.argsort(X_pca)
    sorted_countries = countries.iloc[sorted_idx]

    plt.figure(figsize=(12, 6))

    # Graficar PCA real
    plt.plot(sorted_countries, X_pca[sorted_idx], label="PCA (lib)", linestyle="--", linewidth=2)

    for lr in lrs:
        neuron = OjaNeuron(input_dim=X_std.shape[1], learning_rate=lr)
        if sample:
            neuron.weights = sample_weights_from_data(X_std)
        neuron.train(X_std, epochs=epochs)
        proj = neuron.project(X_std)
        corr = np.corrcoef(proj, X_pca)[0, 1]
        label = f"Oja lr={lr:.3f} (corr={corr:.3f})"
        plt.plot(sorted_countries, proj[sorted_idx], label=label)

    plt.title("Comparación Oja vs PCA por país (PC1)")
    plt.ylabel("PC1")
    plt.xlabel("Países")
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_epochs(epoch_list, lr=0.01, sample=True):
    data_path = Path("data/tp4/europe.csv")
    countries, X_df = load_data(data_path)
    X_std, _ = standardize(X_df)

    from sklearn.decomposition import PCA
    X_pca = PCA(n_components=1).fit_transform(X_std).flatten()

    plt.figure(figsize=(10, 6))

    for ep in epoch_list:
        neuron = OjaNeuron(input_dim=X_std.shape[1], learning_rate=lr)
        if sample:
            neuron.weights = sample_weights_from_data(X_std)
        neuron.train(X_std, epochs=ep)
        proj = neuron.project(X_std)

        corr = np.corrcoef(proj, X_pca)[0, 1]
        label = f"epochs={ep} (corr={corr:.3f})"
        plt.plot(sorted(proj), label=label)

    plt.title("Impacto de cantidad de epochs sobre PC1")
    plt.ylabel("PC1 (ordenado)")
    plt.xlabel("Países (ordenados)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_learning_rates([0.001, 0.01, 0.05], epochs=300)
    compare_epochs([1, 10, 50, 100, 300, 500], lr=0.01)