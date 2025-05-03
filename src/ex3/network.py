from typing import List
import numpy as np
from .layers import DenseLayer


class MLP:
    """
    Contenedor de capas densas – permite forward y backward sobre todo
    el stack.  Se configura con dos listas paralelas:
        layer_sizes = [in, h1, h2, ..., out]
        activations = ["tanh", "tanh", ..., "softmax"]
    Por convenio, activations[0] se ignora (es la entrada).
    """
    # ------------------------------------------------------------------ #
    def __init__(self, layer_sizes: List[int], activations: List[str]):
        assert len(layer_sizes) == len(activations), \
            "activations debe tener mismo largo que layer_sizes"
        self.layers: List[DenseLayer] = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                DenseLayer(
                    in_dim=layer_sizes[i],
                    out_dim=layer_sizes[i + 1],
                    act_name=activations[i + 1]
                )
            )

    # ------------------------------------------------------------------ #
    # Cada capa es batch-aware
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Propaga x a lo largo de todas las capas y devuelve la salida."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # ------------------------------------------------------------------ #
    def backward(self, loss_grad: np.ndarray) -> None:
        """Propaga el gradiente desde la salida hacia la entrada."""
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    # ------------------------------------------------------------------ #
    def params_and_grads(self):
        """Agrupa (param, grad) de TODAS las capas."""
        for layer in self.layers:
            yield from layer.params_and_grads()
