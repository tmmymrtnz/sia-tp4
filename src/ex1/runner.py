# src/common/runner.py
import json
import sys
sys.path.insert(0, "src")
from common.activations import step
from perceptron import Perceptron

def run_experiment(config_path, X, Y):
    with open(config_path, "r") as f:
        config = json.load(f)

    p = Perceptron(
        input_size=2,
        learning_rate=config["learning_rate"],
        bias_init=config["bias"],
        activation_func=step,
        max_epochs=config["max_epochs"]
    )

    p.train(X, Y)