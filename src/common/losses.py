import math
import numpy as np

# ------------------------------------------------------------------
def mse(y_true, y_pred) -> float:
    """Mean-Squared Error for numerical lists/iterables."""
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)

# ------------------------------------------------------------------
def cross_entropy(y_true, y_pred, epsilon=1e-12) -> float:
    """
    Categorical Cross-Entropy loss.
    - y_true: one-hot encoded true labels.
    - y_pred: predicted probabilities (e.g., softmax outputs).
    - epsilon: small value to avoid log(0).
    """
    # Clamp predictions to avoid log(0)
    y_pred = [max(min(p, 1 - epsilon), epsilon) for p in y_pred]
    
    return -sum(t * math.log(p) for t, p in zip(y_true, y_pred))

# ------------------------------------------------------------------
def binary_cross_entropy(y_true, y_pred, eps=1e-12):
    """
    BCE vectorizado para problemas 0/1 con salida sigmoid.
    y_true, y_pred: arrays de shape (B, 1)
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()