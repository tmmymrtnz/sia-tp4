import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path
from sklearn.model_selection import (
    KFold, StratifiedKFold, ShuffleSplit, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

sys.path.insert(0, "src")
from ex3.runner_parity import load_parity_dataset
from ex3.network       import MLP
from ex3.trainer       import Trainer

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
        onehot[idx] = 1.0
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

def _get_labels(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[1] > 1:
        return np.argmax(y, axis=1)
    return y.ravel()

def run_cv(cv, X, y, cfg):
    y_labels = _get_labels(y)
    fold_metrics = []
    for ti, tj in cv.split(X, y_labels):
        X_tr, X_te = X[ti], X[tj]
        y_tr, y_te = y[ti], y[tj]

        net = MLP(cfg['layer_sizes'], cfg['activations'])
        trainer = Trainer(
            net=net,
            loss_name=cfg['loss'],
            optimizer_name=cfg['optimizer'],
            optim_kwargs=cfg.get('optim_kwargs', {}),
            batch_size=cfg.get('batch_size', len(X_tr)),
            max_epochs=cfg.get('max_epochs'),
            log_every=0
        )
        trainer.fit(X_tr, y_tr)

        y_prob = net.forward(X_te)
        y_pred = (y_prob > 0.5).astype(int)

        if y_te.ndim == 2 and y_te.shape[1] > 1:
            y_true_lbl = np.argmax(y_te, axis=1)
            y_pred_lbl = np.argmax(y_pred, axis=1)
            avg = 'macro'
        else:
            y_true_lbl = y_te.ravel()
            y_pred_lbl = y_pred.ravel()
            avg = 'binary'

        fold_metrics.append({
            'accuracy' : accuracy_score(y_true_lbl, y_pred_lbl),
            'precision': precision_score(y_true_lbl, y_pred_lbl, zero_division=0, average=avg),
            'recall'   : recall_score(y_true_lbl, y_pred_lbl, zero_division=0, average=avg),
            'f1'       : f1_score(y_true_lbl, y_pred_lbl, zero_division=0, average=avg)
        })
    return fold_metrics

def eval_holdout(test_size, X, y, cfg, n_runs, seed=42):
    y_labels = _get_labels(y)
    metrics = {m: [] for m in ('accuracy','precision','recall','f1')}
    for run in range(n_runs):
        X_tr, X_te, y_tr, y_te, lbl_tr, lbl_te = train_test_split(
            X, y, y_labels,
            test_size=test_size,
            stratify=y_labels,
            random_state=seed + run
        )

        net = MLP(cfg['layer_sizes'], cfg['activations'])
        trainer = Trainer(
            net=net,
            loss_name=cfg['loss'],
            optimizer_name=cfg['optimizer'],
            optim_kwargs=cfg.get('optim_kwargs', {}),
            batch_size=cfg.get('batch_size', len(X_tr)),
            max_epochs=cfg.get('max_epochs'),
            log_every=0
        )
        trainer.fit(X_tr, y_tr)

        y_prob = net.forward(X_te)
        y_pred = (y_prob > 0.5).astype(int)

        if y_te.ndim == 2 and y_te.shape[1] > 1:
            y_true_lbl = np.argmax(y_te, axis=1)
            y_pred_lbl = np.argmax(y_pred, axis=1)
            avg = 'macro'
        else:
            y_true_lbl = y_te.ravel()
            y_pred_lbl = y_pred.ravel()
            avg = 'binary'

        metrics['accuracy'].append( accuracy_score(y_true_lbl, y_pred_lbl) )
        metrics['precision'].append( precision_score(y_true_lbl, y_pred_lbl, zero_division=0, average=avg) )
        metrics['recall'].append(    recall_score(y_true_lbl, y_pred_lbl, zero_division=0, average=avg) )
        metrics['f1'].append(        f1_score(y_true_lbl, y_pred_lbl, zero_division=0, average=avg) )

    stats = {}
    for m, vals in metrics.items():
        arr = np.array(vals)
        stats[m] = {
            'mean': arr.mean(),
            'std':  arr.std(),
            'min':  arr.min(),
            'max':  arr.max()
        }
    return stats

def main():
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())
    task = cfg['task']

    if task == 'parity':
        X, y = load_parity_dataset(Path(cfg['data_path']))
    elif task == 'digit':
        X, y = load_digit_dataset(Path(cfg['data_path']))
    else:
        raise ValueError(f"Unknown task {task}")

    if 'augment' in cfg:
        X, y = augment_digits(
            X, y,
            cfg['augment']['copies_per_digit'],
            cfg['augment']['noise_p']
        )
        print(f"After augment: {len(X)} samples\n")

    # Cross‐Validation
    n_splits = cfg['cv']['n_splits']
    n_runs   = cfg['cv']['n_runs']
    ss_ts    = cfg['cv']['shuffle_split_test_size']
    methods = {
        'KFold'          : KFold(n_splits=n_splits, shuffle=True,  random_state=0),
        'StratifiedKFold': StratifiedKFold(n_splits=n_splits, shuffle=True,  random_state=0),
        'ShuffleSplit'   : ShuffleSplit(n_splits=n_splits, test_size=ss_ts, random_state=0)
    }
    all_cv = {}
    for name, cv in methods.items():
        runs = []
        for run in range(n_runs):
            if hasattr(cv, 'random_state'):
                cv.random_state = run * 100 + 1
            folds = run_cv(cv, X, y, cfg)
            runs.append({ m: np.mean([f[m] for f in folds]) for m in folds[0] })
        all_cv[name] = runs

    print("=== Cross-Validation Results ===")
    for name, runs in all_cv.items():
        print(f"\n{name}:")
        for m in ('accuracy','precision','recall','f1'):
            arr = np.array([r[m] for r in runs])
            print(f"  {m:10s}: {arr.mean():.3f} ± {arr.std():.3f}")

    # CV boxplots 2×2
    metrics = ['accuracy','precision','recall','f1']
    fig, axs = plt.subplots(2, 2, figsize=(12,8))
    axs = axs.flatten()
    for ax, m in zip(axs, metrics):
        data = [[r[m] for r in all_cv[name]] for name in methods]
        ax.boxplot(data, labels=list(methods.keys()), showmeans=True)
        ax.set_title(m.capitalize()); ax.tick_params(axis='x', rotation=45)
    plt.tight_layout(); plt.show()

    # Hold-out Sweep
    ts_list = cfg['holdout']['test_sizes']
    if ts_list:
        ho = {m:{'mean':[], 'min':[], 'max':[]} for m in metrics}
        print("\n=== Hold-out Sweep ===")
        for ts in ts_list:
            s = eval_holdout(ts, X, y, cfg, cfg['holdout']['n_runs'])
            print(f"test_size={ts:.2f} » " +
                  "  ".join(f"{m}={s[m]['mean']:.3f}±{s[m]['std']:.3f}" for m in metrics))
            for m in metrics:
                ho[m]['mean'].append(s[m]['mean'])
                ho[m]['min'] .append(s[m]['min'])
                ho[m]['max'] .append(s[m]['max'])

        # errorbar plots 2×2
        fig, axs = plt.subplots(2, 2, figsize=(12,10))
        axs = axs.flatten()
        for ax, m, c in zip(axs, metrics, ['C0','C1','C2','C3']):
            mu = np.array(ho[m]['mean'])
            lo = mu - np.array(ho[m]['min'])
            hi = np.array(ho[m]['max']) - mu
            ax.errorbar(ts_list, mu, yerr=[lo,hi], fmt='o-', color=c, capsize=5)
            ax.set_title(m.capitalize())
            ax.set_xlabel('Test fraction'); ax.set_ylabel(m); ax.grid(True)
        plt.tight_layout(); plt.show()

if __name__ == '__main__':
    main()
