import math

def mse(y_true, y_pred) -> float:
    """Mean-Squared Error for numerical lists/iterables."""
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)

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