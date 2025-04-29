import random

class Perceptron:
    def __init__(self, input_size, learning_rate, bias_init, activation_func, max_epochs):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias = bias_init if bias_init is not None else random.uniform(-1, 0)
        self.learning_rate = learning_rate
        self.activation = activation_func
        self.max_epochs = max_epochs

    def predict(self, inputs):
        # Weighted sum of inputs and weights + bias
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        # Apply activation function to predict 1 or 0
        return self.activation(weighted_sum)

    def train(self, X, Y):
        for epoch in range(self.max_epochs):
            errors = 0
            for x_vec, y_true in zip(X, Y):
                y_pred = self.predict(x_vec)
                if y_pred != y_true:
                    errors += 1
                    # Update weights
                    for i in range(len(self.weights)):
                        self.weights[i] += self.learning_rate * (y_true - y_pred) * x_vec[i]
                    # Update bias
                    self.bias += self.learning_rate * (y_true - y_pred)
            print(f"Epoch {epoch + 1}: Errors = {errors}")
            if errors == 0:
                break