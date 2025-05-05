# ───────────────────────────────────────────────────────────────
#  src/ex3/trainer.py           ← completely replaces the old file
# ───────────────────────────────────────────────────────────────
import numpy as np
from common.losses      import mse, cross_entropy, binary_cross_entropy
from common.optimizers  import SGD, Momentum, Adam
from .network           import MLP


LOSS_FUNS = {
    "mse" : mse,
    "cross_entropy": cross_entropy,
    "bce": binary_cross_entropy
}

OPTIMIZERS = {
    "sgd"     : SGD,
    "momentum": Momentum,
    "adam"    : Adam
}


class Trainer:
    """
    Handles the training loop:
        • shuffles every epoch
        • splits into mini‑batches
        • forward → loss → grad
        • back‑prop → optimiser update
        • optional weight/stat logging
    """
    # ───────────────────────────────────────────────────────────
    def __init__(
        self,
        net            : MLP,
        loss_name      : str,
        optimizer_name : str,
        optim_kwargs   : dict,
        batch_size     : int,
        max_epochs     : int,
        log_every      : int = 100,        # ← new
        log_weights    : bool = False      # ← new
    ):
        self.net        = net
        self.loss_fn    = LOSS_FUNS[loss_name]
        self.optim      = OPTIMIZERS[optimizer_name](**optim_kwargs)
        self.batch      = batch_size
        self.epochs     = max_epochs
        self.log_every  = max(1, log_every)
        self.log_weights= log_weights
        self.weight_hist= []               # stores snapshots *iff* log_weights=True

    # ───────────────────────────────────────────────────────────
    def _loss_and_grad(self, y_hat, y_true):
        """
        Returns (scalar_loss, grad_wrt_ŷ).
        Implements analytic grads for MSE & BCE with sigmoid output.
        """
        if self.loss_fn is mse:
            loss = mse(y_true, y_hat)
            grad = (y_hat - y_true) / len(y_true)
            return loss, grad

        elif self.loss_fn is binary_cross_entropy:
            loss = binary_cross_entropy(y_true, y_hat)
            grad = (y_hat - y_true) / len(y_true)          # exact for sigmoid+BCE
            return loss, grad

        else:
            raise ValueError("loss function not supported")

    # ───────────────────────────────────────────────────────────
    def _log_weights(self, epoch):
        # flatten all trainable params into one vector for stats
        flat = np.concatenate([w.ravel() for w, _ in self.net.params_and_grads()])
        mean, abs_max = flat.mean(), np.abs(flat).max()
        print(f"    ↳  weights: mean = {mean:+.4e}   |   |w|_∞ = {abs_max:.4e}")

        if self.log_weights:                       # optional deep copy
            snapshot = [w.copy() for w, _ in self.net.params_and_grads()]
            self.weight_hist.append((epoch, snapshot))

    # ───────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, Y: np.ndarray):
        N = len(X)
        for epoch in range(1, self.epochs + 1):

            # ---- shuffle whole dataset ----
            perm = np.random.permutation(N)
            X, Y = X[perm], Y[perm]

            epoch_loss = 0.0

            # ---- mini‑batch loop ----
            for i in range(0, N, self.batch):
                xb = X[i : i + self.batch]
                yb = Y[i : i + self.batch]

                # forward (vectorised)
                y_hat = self.net.forward(xb)                    # (B, out_dim)

                # loss & grad
                loss_batch, grad_yhat = self._loss_and_grad(y_hat, yb)
                epoch_loss += loss_batch * len(xb)

                # backward + optimiser update
                self.net.backward(grad_yhat)
                self.optim.update(self.net.params_and_grads())

            # ---- aggregate stats for epoch ----
            epoch_loss /= N

            if epoch % self.log_every == 0 or epoch == 1 or epoch == self.epochs:
                print(f"Epoch {epoch:>5d}/{self.epochs}  |  loss = {epoch_loss:.6f}")
                self._log_weights(epoch)
