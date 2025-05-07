"""
Runner para el Ejercicio 3 – caso XOR usando red MLP.
Ejemplo de uso:
    python src/ex3/runner_xor.py src/ex3/configs/xor.json
"""
import sys
import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# 1.  importar red y trainer
# --------------------------------------------------------------------------- #
sys.path.insert(0, "src")          # para que “src/…” sea import‑root
from common.perceptrons.multilayer.trainer  import Trainer
from common.perceptrons.multilayer.network import MLP        # tu implementación de la red

# --------------------------------------------------------------------------- #
# 2. leer la configuración
# --------------------------------------------------------------------------- #
if len(sys.argv) != 2:
    print("Uso: python runner_xor.py <config.json>")
    sys.exit(1)

cfg_path = Path(sys.argv[1])
with cfg_path.open() as f:
    cfg = json.load(f)

# --------------------------------------------------------------------------- #
# 3. datos XOR
# --------------------------------------------------------------------------- #
X = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ],
    dtype=np.float32,
)

y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# --------------------------------------------------------------------------- #
# 4. instanciar red y trainer
# --------------------------------------------------------------------------- #
net = MLP(cfg["layer_sizes"], cfg["activations"])

trainer = Trainer(
    net=net,
    loss_name=cfg["loss"],
    optimizer_name=cfg["optimizer"],
    optim_kwargs=cfg["optim_kwargs"],
    batch_size=cfg["batch_size"],
    max_epochs=cfg["max_epochs"],

    log_every= 100,
    log_weights= True
)

# --------------------------------------------------------------------------- #
# 5. entrenar
# --------------------------------------------------------------------------- #
trainer.fit(X, y)

# --------------------------------------------------------------------------- #
# 6. test final
# --------------------------------------------------------------------------- #
print("\nPredicciones finales:")
for x_i, y_i in zip(X, y):
    # Siempre pasamos un array 2‑D al forward;
    # después quitamos la primera dimensión.
    prob = net.forward(x_i[None, :])[0, 0]
    pred = int(prob > 0.5)
    print(f"Input: {x_i} → Pred: {pred}  |  True: {int(y_i)}")
