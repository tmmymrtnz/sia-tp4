# ------------------------------------------------------------
#  src/ex3/mlp.py
# ------------------------------------------------------------
from typing import List
import numpy as np

from .layers import DenseLayer


class MLP:
    """
    Stack de capas densas con activaciones:
        layer_sizes = [in, h1, ..., out]
        activations = ["linear", "tanh", ..., "sigmoid"]
    Por convenio, activations[0] corresponde a la entrada (se ignora).
    """
    # ------------------------------------------------------ #
    def __init__(self, layer_sizes: List[int], activations: List[str]):
        # Aceptamos tanto N  como  N‑1 activaciones (sin la de entrada).
        if len(activations) == len(layer_sizes) - 1:
            activations = ["linear"] + activations
        assert len(layer_sizes) == len(activations), \
            "`activations` debe tener len = layer_sizes  ó  layer_sizes-1"

        self.layers: List[DenseLayer] = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                DenseLayer(
                    in_dim=layer_sizes[i],
                    out_dim=layer_sizes[i + 1],
                    act_name=activations[i + 1]   # se salta la de entrada
                )
            )

    # ------------------------------------------------------ #
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # ------------------------------------------------------ #
    def backward(self, loss_grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    # ------------------------------------------------------ #
    def params_and_grads(self):
        for layer in self.layers:
            yield from layer.params_and_grads()
