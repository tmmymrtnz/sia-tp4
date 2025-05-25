"""
Runner genérico para el Ejercicio 2 (sin scaler y sin seed).

Ejemplos:
    python src/ex2/runner.py src/ex2/configs/linear.json
    python src/ex2/runner.py src/ex2/configs/nonlinear.json
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path

# --- importar utilidades comunes ------------------------------------------
sys.path.insert(0, "src")
from common.activations import identity, identity_deriv, tanh, tanh_deriv
from common.perceptrons.simple.perceptron import Perceptron
from common.losses import mse
# --------------------------------------------------------------------------

# -------- leer configuración ----------
if len(sys.argv) != 2:
    print("Uso: python runner.py <config.json>")
    sys.exit(1)

cfg_path = Path(sys.argv[1])
with cfg_path.open() as f:
    cfg = json.load(f)

act_name = cfg["activation"].lower()
if act_name == "linear":
    act, act_d = identity, identity_deriv
    scale_y    = False
elif act_name == "tanh":
    act, act_d = tanh, tanh_deriv
    scale_y    = True
else:
    raise ValueError("activation debe ser 'linear' o 'tanh'")

# ---------- dataset sin escalar ----------
df = pd.read_csv("data/TP3-ej2-conjunto.csv")
X = df[["x1", "x2", "x3"]].values.astype(float)
y = df["y"].values.astype(float)

# escalar output sólo si usamos tanh
if scale_y:
    y_min, y_max = y.min(), y.max()
    y = 2 * (y - y_min) / (y_max - y_min) - 1

# ---------- modelo ----------
p = Perceptron(
    input_size      = 3,
    learning_rate   = cfg["learning_rate"],
    max_epochs      = cfg["max_epochs"],
    bias_init       = cfg.get("bias_init", 0.0),
    activation_func = act,
    activation_deriv= act_d
)

p.train(X.tolist(), y.tolist())
y_hat = [p.predict(x) for x in X]

# volver a escala original si hacía falta
if scale_y:
    y_hat = (np.array(y_hat) + 1) * 0.5 * (y_max - y_min) + y_min

print("\nMSE entrenamiento:", mse(df["y"].values, y_hat))
