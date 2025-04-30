def mse(y_true, y_pred) -> float:
    """Mean-Squared Error para listas/iterables numéricos."""
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)
