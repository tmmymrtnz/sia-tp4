import numpy as np
from common.losses import mse, cross_entropy, binary_cross_entropy
from common.optimizers import SGD, Momentum, Adam
from .network import MLP


LOSS_FUNS = {
    "mse": mse,
    "cross_entropy": cross_entropy,
    "bce":  binary_cross_entropy
}

OPTIMIZERS = {
    "sgd": SGD,
    "momentum": Momentum,
    "adam": Adam
}


class Trainer:
    """
    Gestiona el ciclo de entrenamiento:
        • baraja cada epoch
        • corta en mini‑batches
        • hace forward, calcula pérdida y su gradiente
        • back‑prop, actualiza por optimizador
    """
    def __init__(
        self,
        net: MLP,
        loss_name: str,
        optimizer_name: str,
        optim_kwargs: dict,
        batch_size: int,
        max_epochs: int,
        shuffle: bool = True
    ):
        self.net = net
        self.loss_fn = LOSS_FUNS[loss_name]
        self.optim = OPTIMIZERS[optimizer_name](**optim_kwargs)
        self.batch = batch_size
        self.epochs = max_epochs
        self.shuffle = shuffle

    def _loss_and_grad(self, y_hat: np.ndarray, y_true: np.ndarray):
        """
        Devuelve (loss_scalar, grad_yhat).
        Soporta mse, binary_cross_entropy y cross_entropy.
        """
        if self.loss_fn is mse:
            loss = mse(y_true, y_hat)
            grad = (y_hat - y_true) / len(y_true)
            return loss, grad

        elif self.loss_fn is binary_cross_entropy:
            loss = binary_cross_entropy(y_true, y_hat)
            grad = (y_hat - y_true) / len(y_true)
            return loss, grad

        elif self.loss_fn is cross_entropy:
            # Asumimos softmax en salida
            loss = cross_entropy(y_true, y_hat)
            grad = (y_hat - y_true) / len(y_true)
            return loss, grad

        else:
            raise ValueError(f"Función de pérdida no soportada: {self.loss_fn}")

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = True):
        N = len(X)
        for epoch in range(1, self.epochs + 1):
            # ---- shuffle ----
            if self.shuffle:
                perm = np.random.permutation(N)
                X, Y = X[perm], Y[perm]

            epoch_loss = 0.0
            # ---- mini‑batches ----
            for i in range(0, N, self.batch):
                xb = X[i:i + self.batch]
                yb = Y[i:i + self.batch]

                # vectorizado forward
                y_hat = self.net.forward(xb)

                # loss & grad
                loss, grad_yhat = self._loss_and_grad(y_hat, yb)
                epoch_loss += loss * len(xb)

                # backward
                self.net.backward(grad_yhat)

                # optim update
                self.optim.update(self.net.params_and_grads())

            epoch_loss /= N
            if verbose:
                print(f"Epoch {epoch}  |  loss = {epoch_loss:.6f}")
