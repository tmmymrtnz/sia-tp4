"""
Runner para el Ejercicio 3 – caso XOR usando red MLP.
Ejemplo de uso:
    python src/ex3/runner_xor.py src/ex3/configs/xor.json
"""
import sys, json
import numpy as np
from pathlib import Path

# --- importar red y trainer ---
sys.path.insert(0, "src")
from common.activations import step
from ex3.trainer import Trainer
from ex3.network import MLP

# --- leer config ---
if len(sys.argv) != 2:
    print("Uso: python runner_xor.py <config.json>")
    sys.exit(1)

cfg_path = Path(sys.argv[1])
with cfg_path.open() as f:
    cfg = json.load(f)

# --- datos XOR ---
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# --- red y trainer ---
net = MLP(cfg["layer_sizes"], cfg["activations"])

trainer = Trainer(
    net=net,
    loss_name=cfg["loss"],
    optimizer_name=cfg["optimizer"],
    optim_kwargs=cfg["optim_kwargs"],
    batch_size=cfg["batch_size"],
    max_epochs=cfg["max_epochs"]
)

# --- entrenar ---
trainer.fit(X, y)

# --- test final ---
print("\nPredicciones finales:")
for x_i, y_i in zip(X, y):
    # pred = step(net.forward(x_i))
    pred = net.forward(x_i)
    if(pred[0] > 0.5):
        pred = 1
    else:
        pred = 0
    # if(pred == -1):
    #     pred = 0
    print(f"Input: {x_i} → Pred: {pred}  |  True: {y_i}")