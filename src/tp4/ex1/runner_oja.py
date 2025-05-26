import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

class OjaNeuron:
    def __init__(self, input_dim, learning_rate=0.01):
        self.weights = np.random.randn(input_dim)
        self.lr = learning_rate

    def train(self, X, epochs):
        for epoch in range(epochs):
            for x in X:
                y = np.dot(self.weights, x)
                self.weights += self.lr * y * (x - y * self.weights)

    def project(self, X):
        return X @ self.weights


def load_data(path):
    df = pd.read_csv(path)
    countries = df["Country"]
    X = df.drop(columns=["Country"])
    return countries, X


def standardize(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), X.columns


def sample_weights_from_data(X):
    idx = np.random.choice(len(X))
    return X[idx].copy()


def plot_pc1_bar(values, country_labels):
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_countries = country_labels.iloc[sorted_idx]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=sorted_countries, y=sorted_vals, palette="coolwarm")
    plt.xticks(rotation=90)
    plt.ylabel("PC1 Value")
    plt.title("PC1 per Country (Oja Model)")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def plot_weights(weights, feature_names):
    plt.figure(figsize=(10, 4))
    sns.barplot(x=feature_names, y=weights)
    plt.xticks(rotation=45)
    plt.ylabel("Weight")
    plt.title("Learned Weights (PC1 direction)")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)

    data_path = Path("data/tp4/europe.csv")
    countries, X_df = load_data(data_path)
    X_std, feature_names = standardize(X_df)

    input_dim = X_std.shape[1]
    neuron = OjaNeuron(input_dim, learning_rate=cfg.get("lr", 0.01))

    if cfg.get("sample", False):
        neuron.weights = sample_weights_from_data(X_std)

    neuron.train(X_std, epochs=cfg.get("max_epochs", 500))
    projections = neuron.project(X_std)

    plot_pc1_bar(projections, countries)
    plot_weights(neuron.weights, feature_names)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python oja_model.py config.json")
        sys.exit(1)

    main(sys.argv[1])