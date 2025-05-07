# ------------------------------------------------------------
#  src/ex3/layers.py
# ------------------------------------------------------------
from __future__ import annotations
import numpy as np


class DenseLayer:
    """
    Capa totalmente conectada (W @ x + b) + activación.
    ‑ in_dim, out_dim  : tamaños
    ‑ act_name         : "linear", "tanh", "sigmoid", "relu", "leaky_relu", …
    """
    # ------------------------------------------------------ #
    def __init__(self, in_dim: int, out_dim: int, act_name: str):
        self.act_name = act_name.lower()

        # ----- pesos segun la no‑linealidad ---------------- #
        self.W = self._init_weights(in_dim, out_dim)
        self.b = np.zeros((1, out_dim))

        # grads
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # cache para back‑prop
        self._x:  np.ndarray | None = None
        self._z:  np.ndarray | None = None

    # ------------------------------------------------------ #
    #  He / Xavier
    def _init_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        if self.act_name in {"relu", "leaky_relu"}:
            std = np.sqrt(2.0 / fan_in)          # He
            return np.random.randn(fan_in, fan_out) * std
        else:                                    # tanh, sigmoid, …
            limit = np.sqrt(6.0 / (fan_in + fan_out))   # Xavier/Glorot
            return np.random.uniform(-limit, limit, size=(fan_in, fan_out))

    # ------------------------------------------------------ #
    #  activación + derivada (sin dependencias externas)
    def _activation(self, z: np.ndarray, deriv: bool = False) -> np.ndarray:
        if self.act_name in {"identity", "linear"}:
            return np.ones_like(z) if deriv else z

        if self.act_name == "tanh":
            if deriv:
                a = np.tanh(z)
                return 1.0 - a * a
            return np.tanh(z)

        if self.act_name == "sigmoid":
            a = 1.0 / (1.0 + np.exp(-z))
            return a * (1.0 - a) if deriv else a

        if self.act_name == "relu":
            return (z > 0).astype(z.dtype) if deriv else np.maximum(0, z)

        if self.act_name == "leaky_relu":
            alpha = 0.01
            return np.where(z > 0, 1.0, alpha) if deriv else np.where(z > 0, z, alpha * z)

        raise ValueError(f"Unknown activation '{self.act_name}'")

    # ------------------------------------------------------ #
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._z = x @ self.W + self.b
        return self._activation(self._z)

    # ------------------------------------------------------ #
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        grad_out = ∂L/∂a  (del layer superior)
        Devuelve ∂L/∂x para la capa inferior y deja dW/db listos.
        """
        grad_z = grad_out * self._activation(self._z, deriv=True)

        self.dW[:] = self._x.T @ grad_z / len(self._x)
        self.db[:] = grad_z.mean(axis=0, keepdims=True)

        return grad_z @ self.W.T

    # ------------------------------------------------------ #
    def params_and_grads(self):
        yield self.W, self.dW
        yield self.b, self.db
