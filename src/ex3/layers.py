import numpy as np
from common.activations import ACT, DACT


class DenseLayer:
    """
    Capa completamente conectada (out × in) + activación.
    Forward  :  x  ->  a = f(W·x + b)
    Backward :  recibe dL/da y devuelve dL/dx;
               guarda dL/dW, dL/db para el optimizador.
    """
    # ------------------------------------------------------------------ #
    def __init__(self, in_dim: int, out_dim: int, act_name: str = "tanh"):
        # He init para activaciones simétricas (tanh/relu)
        self.W = np.random.randn(out_dim, in_dim) * np.sqrt(2 / in_dim)
        self.b = np.zeros(out_dim)

        self.f  = ACT[act_name]
        self.df = DACT[act_name]

        # espacios para gradientes
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # cachés
        self.x  = None
        self.z  = None

    # ------------------------------------------------------------------ #
    # DenseLayer.forward (batch-aware)
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x                             # (B, in_dim)
        self.z = x @ self.W.T + self.b         # (B, out_dim)
        return self.f(self.z)                  # (B, out_dim)

    # ------------------------------------------------------------------ #
    # DenseLayer.backward (batch-aware)
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        grad_out = dL/da   (misma shape que la salida)
        Devuelve          = dL/dx  (shape in_dim,)
        """
        dz = grad_out * self.df(self.z)        # (B, out_dim)
        self.dW = dz.T @ self.x / len(self.x)  # (out_dim, in_dim)
        self.db = dz.mean(axis=0)              # (out_dim,)
        return dz @ self.W                     # (B, in_dim)
    
    # ------------------------------------------------------------------ #
    def params_and_grads(self):
        """Itera (param, grad) para que el optimizador actualice."""
        yield self.W, self.dW
        yield self.b, self.db
