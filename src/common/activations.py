import math
import numpy as np

# ---------- básicas ----------
def step(x: float) -> int:
    """Escalón bipolar (+1 / -1)."""
    return 1 if x >= 0 else -1


def identity(x: float) -> float:          # <- lineal
    return x


def identity_deriv(_: float) -> float:    # derivada constante
    return 1.0


def tanh(x: float) -> float:              # <- no-lineal
    return np.tanh(x)


def tanh_deriv(x: float) -> float:
    return 1.0 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def step_deriv(x):
    # Derivada de step es cero en todos lados excepto en 0 (donde es indefinida)
    return np.zeros_like(x)

# --- diccionarios de activación ---
ACT = {
    "identity": identity,
    "tanh": tanh,
    "sigmoid": sigmoid,
    "relu": relu,
    "step": step,
}

DACT = {
    "identity": identity_deriv,
    "tanh": tanh_deriv,
    "sigmoid": sigmoid_deriv,
    "relu": relu_deriv,
    "step": step_deriv,
}