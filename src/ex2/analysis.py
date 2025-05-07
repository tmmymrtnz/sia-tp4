#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit
from sklearn.cluster import KMeans
sys.path.insert(0, "src")

from common.perceptron.simple.perceptron import Perceptron
from common.activations import identity, identity_deriv, tanh, tanh_deriv
from common.losses import mse

# ------------------------------------------------------------------
# Helper: train with explicit epoch loop and early stopping
# ------------------------------------------------------------------
def train_with_validation(
    X_train, y_train, X_val, y_val,
    activation, deriv,
    lr, bias_init, max_epochs
):
    """
    Train perceptron for up to max_epochs, capturing train & validation MSE each epoch.
    Scales y to [-1,1] for tanh, leaves X unscaled.
    Implements early stopping based on validation MSE.
    Returns (train_history, val_history, y_min, y_max, model).
    """
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    # Scale y for tanh
    if activation is tanh:
        y_min, y_max = y_train.min(), y_train.max()
        if y_max != y_min:
            y_train_scaled = 2 * (y_train - y_min) / (y_max - y_min) - 1
            y_val_scaled = 2 * (y_val - y_min) / (y_max - y_min) - 1
        else:
            y_train_scaled = y_train.copy()
            y_val_scaled = y_val.copy()
    else:
        y_min = y_max = None
        y_train_scaled = y_train.copy()
        y_val_scaled = y_val.copy()

    # Initialize model
    p = Perceptron(
        input_size=X_train.shape[1],
        learning_rate=lr,
        max_epochs=max_epochs,
        bias_init=bias_init,
        activation_func=activation,
        activation_deriv=deriv
    )

    train_history, val_history = [], []
    best_val = float('inf')
    wait = 0
    best_weights = p.weights.copy()
    best_bias = p.bias

    # Epoch loop
    for epoch in range(1, max_epochs + 1):
        # Training pass
        sq_err = 0.0
        for xi, target in zip(X_train, y_train_scaled):
            a = p._weighted_sum(xi)
            y_pred = p.act(a)
            error = target - y_pred
            grad = error * p.act_deriv(a)
            p.weights = [w + lr * grad * xi_i for w, xi_i in zip(p.weights, xi)]
            p.bias += lr * grad
            sq_err += error * error
        train_mse = sq_err / len(X_train)
        train_history.append(train_mse)

        # Validation pass
        preds_val = np.array([p.predict(xi) for xi in X_val])
        val_mse = mse(y_val_scaled, preds_val)
        val_history.append(val_mse)

        # Early stopping
        if val_mse < best_val:
            best_val = val_mse
            best_weights = p.weights.copy()
            best_bias = p.bias
            wait = 0
        else:
            wait += 1

    # Restore best model
    p.weights = best_weights
    p.bias = best_bias

    return train_history, val_history, (y_min, y_max), p

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def plot_learning_curves(curves, title):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    all_vals = []
    for label, (train_hist, val_hist) in curves.items():
        ax.plot(train_hist, label=f"{label} (train)")
        ax.plot(val_hist, '--', label=f"{label} (val)")
        all_vals.extend(train_hist)
        all_vals.extend(val_hist)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    if all_vals:
        max_val = max(all_vals)
        ax.set_ylim(0, max_val * 1.1)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.show()



# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(config_path):
    cfg = json.load(open(config_path))
    df = pd.read_csv(cfg['dataset'])
    X = df[['x1','x2','x3']].values
    y = df['y'].values

    # Train/val/test split
    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=0.25, random_state=42
    )

    hp = cfg['hyperparameters']
    settings = cfg['plot_settings']
    act_map = {
        'linear': (identity, identity_deriv),
        'tanh': (tanh, tanh_deriv)
    }

    best = {'mse': float('inf'), 'label': '', 'test_mse': None}
    # Prepare curves per epoch
    curves_by_epoch = {ep: {} for ep in hp['max_epochs']}

    # Hyperparameter sweep
    for lr in hp['learning_rate']:
        for bias in hp['bias_init']:
            for ep in hp['max_epochs']:
                for act_name in hp['activation']:
                    activation, deriv = act_map[act_name]
                    combo_label = f"lr={lr}, bias={bias}, act={act_name}"

                    tr_hist, val_hist, (y_min, y_max), model = train_with_validation(
                        X_train, y_train, X_val, y_val,
                        activation, deriv,
                        lr, bias, ep
                    )

                    # Store under the correct epoch
                    curves_by_epoch[ep][combo_label] = (tr_hist, val_hist)

                    # Compute scaled validation MSE
                    final_val = val_hist[-1] if val_hist else float('inf')
                    if act_name == 'tanh' and y_min is not None:
                        scale2 = ((y_max - y_min) / 2)**2
                        final_val *= scale2

                    # Evaluate test MSE
                    if act_name == 'tanh' and y_min is not None:
                        y_test_scaled = 2 * (y_test - y_min) / (y_max - y_min) - 1
                        preds_test = [model.predict(xi) for xi in X_test]
                        test_mse = mse(y_test_scaled, preds_test) * scale2
                    else:
                        preds_test = [model.predict(xi) for xi in X_test]
                        test_mse = mse(y_test, preds_test)

                    # Select best
                    if final_val < best['mse']:
                        best = {'mse': final_val, 'label': f"{combo_label}, ep={ep}", 'test_mse': test_mse}

    # Plot learning curves grouped by epoch (separate linear & tanh with individual scales)
    if settings.get('learning_curve'):
        for ep, curves in curves_by_epoch.items():
            # Determine best performer for this epoch (smallest validation MSE)
            best_label, best_val = None, float('inf')
            for label, (_, val_hist) in curves.items():
                if val_hist:
                    final = val_hist[-1]
                    if final < best_val:
                        best_val = final
                        best_label = label
            print(f"Best performer for {ep} epochs: {best_label} with val MSE={best_val:.6f}")

            # Split curves by activation
            linear_curves = {lbl: data for lbl, data in curves.items() if 'act=linear' in lbl}
            tanh_curves = {lbl: data for lbl, data in curves.items() if 'act=tanh' in lbl}
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
            # Linear subplot
            axes[0].set_title(f'Linear - {ep} Epochs', fontsize=14)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('MSE', fontsize=12)
            lin_vals = []
            for label, (tr_hist, val_hist) in linear_curves.items():
                axes[0].plot(tr_hist, label=f"{label} (train)")
                axes[0].plot(val_hist, '--', label=f"{label} (val)")
                lin_vals.extend(tr_hist); lin_vals.extend(val_hist)
            if lin_vals:
                max_lin = max(lin_vals)
                axes[0].set_ylim(0, max_lin * 1.1)
            axes[0].legend(fontsize=8)
            axes[0].grid(True, linestyle=':', linewidth=0.5)
            # Tanh subplot
            axes[1].set_title(f'Tanh - {ep} Epochs', fontsize=14)
            axes[1].set_xlabel('Epoch', fontsize=12)
            tanh_vals = []
            for label, (tr_hist, val_hist) in tanh_curves.items():
                axes[1].plot(tr_hist, label=f"{label} (train)")
                axes[1].plot(val_hist, '--', label=f"{label} (val)")
                tanh_vals.extend(tr_hist); tanh_vals.extend(val_hist)
            if tanh_vals:
                max_tanh = max(tanh_vals)
                axes[1].set_ylim(0, max_tanh * 1.1)
            axes[1].legend(fontsize=8)
            axes[1].grid(True, linestyle=':', linewidth=0.5)
            plt.tight_layout()
            plt.show()

    # Report best modelf"\n=== Best model: {best['label']} ===")
    print(f"Val MSE: {best['mse']:.6f}, Test MSE: {best['test_mse']:.6f}")

    # K-fold cross-validation with best hyperparams
    print("\n=== 5-Fold CV on full dataset ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    # parse best params
    parts = best['label'].split(', ')
    best_lr = float(parts[0].split('=')[1])
    best_bias = float(parts[1].split('=')[1])
    best_act = parts[2].split('=')[1]
    best_ep = int(parts[3].split('=')[1])

    for act_name, (activation, deriv) in act_map.items():
        mses = []
        for train_idx, val_idx in kf.split(X):
            tr_X, vr_X = X[train_idx], X[val_idx]
            tr_y, vr_y = y[train_idx], y[val_idx]
            th, vh, (ymin, ymax), _ = train_with_validation(
                tr_X, tr_y, vr_X, vr_y,
                activation, deriv,
                best_lr, best_bias, best_ep
            )
            final = vh[-1] if vh else float('inf')
            if act_name == 'tanh' and ymin is not None:
                final *= ((ymax - ymin) / 2)**2
            mses.append(final)
        cv_results[act_name] = (np.mean(mses), np.std(mses))
        print(f"{act_name}: mean={cv_results[act_name][0]:.6f}, std={cv_results[act_name][1]:.6f}")

    # ------------------------------------------------------------------
    # Sampling–methods comparison
    # ------------------------------------------------------------------
    print("\n=== Sampling methods comparison (10 runs each) ===")

    activation, deriv = act_map[best_act]
    sample_n = int(0.5 * X_train.shape[0])
    runs = 50

    # Sampling helper functions
    def sample_random(X, y, n):
        idx = np.random.choice(len(X), n, replace=False)
        return X[idx], y[idx]

    def sample_extreme(X, y, n):
        deviations = np.abs(y - np.mean(y))
        idx = np.argsort(deviations)[-n:]
        return X[idx], y[idx]

    def sample_stratified(X, y, n):
        bins = pd.qcut(y, q=4, labels=False, duplicates='drop')
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=None)
        for train_idx, _ in sss.split(X, bins):
            return X[train_idx], y[train_idx]

    def sample_cluster_centroid(X, y, n):
        kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        selected = []
        for i in range(n):
            cluster_idx = np.where(labels == i)[0]
            pts = X[cluster_idx]
            dists = np.linalg.norm(pts - centers[i], axis=1)
            nearest = cluster_idx[np.argmin(dists)]
            selected.append(nearest)
        idx = np.array(selected)
        return X[idx], y[idx]

    sampling_methods = {
        'Random': sample_random,
        'Extreme': sample_extreme,
        'Stratified': sample_stratified,
        'Cluster': sample_cluster_centroid
    }

    # Aggregate and average histories
    avg_histories = {}
    for name, sampler in sampling_methods.items():
        all_runs = []
        for _ in range(runs):
            X_s, y_s = sampler(X_train, y_train, sample_n)
            _, val_hist, _, _ = train_with_validation(
                X_s, y_s, X_val, y_val,
                activation, deriv,
                best_lr, best_bias,
                best_ep
            )
            all_runs.append(val_hist)
        epochs = len(all_runs[0])
        avg_histories[name] = [np.mean([run[e] for run in all_runs]) for e in range(epochs)]

    # Plot averaged validation curves
    plt.figure(figsize=(10, 6), dpi=100)
    for name, avg_hist in avg_histories.items():
        plt.plot(avg_hist, label=f"{name} (avg)")
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation MSE', fontsize=12)
    plt.title('Average Validation MSE by Sampling Method', fontsize=14, pad=15)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Cross-validated performance per sampling method
    # ------------------------------------------------------------------
    print("\n=== 5-Fold CV per sampling method ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {name: [] for name in sampling_methods}

    for name, sampler in sampling_methods.items():
        for train_idx, val_idx in kf.split(X_trval):
            X_tr, y_tr = X_trval[train_idx], y_trval[train_idx]
            X_va, y_va = X_trval[val_idx], y_trval[val_idx]
            # sample subset of training fold
            X_s, y_s = sampler(X_tr, y_tr, int(0.5 * len(train_idx)))
            _, val_hist, _, _ = train_with_validation(
                X_s, y_s, X_va, y_va,
                activation, deriv,
                best_lr, best_bias,
                best_ep
            )
            cv_results[name].append(val_hist[-1])

    # bar chart of mean and std CV MSE
    names = list(sampling_methods.keys())
    means = [np.mean(cv_results[n]) for n in names]
    stds  = [np.std(cv_results[n])  for n in names]

    plt.figure(figsize=(8, 5), dpi=100)
    x = np.arange(len(names))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, names)
    plt.ylabel('Validation MSE')
    plt.title('5-Fold CV MSE by Sampling Method')
    plt.grid(True, axis='y', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python analysis.py <config.json>')
        sys.exit(1)
    main(sys.argv[1])
