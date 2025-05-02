import math

class SGD:
    """Plain Stochastic Gradient Descent (no momentum)."""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.learning_rate * grads[i]


class Momentum:
    """Gradient Descent with Momentum."""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = [0 for _ in params]

        for i in range(len(params)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grads[i]
            params[i] += self.velocity[i]


class Adam:
    """Adaptive Moment Estimation optimizer."""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # timestep
        self.m = None  # first moment
        self.v = None  # second moment

    def update(self, params, grads):
        if self.m is None:
            self.m = [0 for _ in params]
            self.v = [0 for _ in params]

        self.t += 1
        lr_t = self.learning_rate * math.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= lr_t * m_hat / (math.sqrt(v_hat) + self.epsilon)