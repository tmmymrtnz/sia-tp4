import math
import numpy as np

class SGD:
    """Plain Stochastic Gradient Descent (no momentum)."""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params_and_grads):
        # params_and_grads: iterable of (param, grad)
        for param, grad in params_and_grads:
            param -= self.learning_rate * grad

class Momentum:
    """Gradient Descent with Momentum."""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params_and_grads):
        # Convert generator to list to reuse
        pg = list(params_and_grads)  # [(param, grad), ...]
        if self.velocity is None:
            # Initialize velocity vectors to zeros, matching each param shape
            self.velocity = [np.zeros_like(param) for param, _ in pg]
        # Update each parameter
        for i, (param, grad) in enumerate(pg):
            # v = μ v − η g
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
            # p ← p + v
            param += self.velocity[i]

class Adam:
    """Adaptive Moment Estimation optimizer."""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # timestep
        self.m = None  # first moment vectors
        self.v = None  # second moment vectors

    def update(self, params_and_grads):
        # Convert generator to list to reuse
        pg = list(params_and_grads)  # [(param, grad), ...]
        # Initialize m and v on first call
        if self.m is None:
            self.m = [np.zeros_like(param) for param, _ in pg]
            self.v = [np.zeros_like(param) for param, _ in pg]
        # Increment time step
        self.t += 1
        # Compute bias-corrected learning rate
        lr_t = self.learning_rate * math.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        # Update each parameter
        for i, (param, grad) in enumerate(pg):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)
            # Compute bias-corrected moment estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # Update parameter
            param -= lr_t * m_hat / (np.sqrt(v_hat) + self.epsilon)