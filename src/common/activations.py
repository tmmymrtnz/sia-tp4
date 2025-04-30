import math

# ---------- básicas ----------
def step(x: float) -> int:
    """Escalón bipolar (+1 / -1)."""
    return 1 if x >= 0 else -1


def identity(x: float) -> float:          # <- lineal
    return x


def identity_deriv(_: float) -> float:    # derivada constante
    return 1.0


def tanh(x: float) -> float:              # <- no-lineal
    return math.tanh(x)


def tanh_deriv(x: float) -> float:
    t = math.tanh(x)
    return 1.0 - t * t
