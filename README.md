# TP3 SIA - Perceptrón Simple y Multicapa

[Enunciado](docs/Enunciado_TP3.pdf)

## Instalación

Parado en la carpeta del tp3 ejecutar

```sh
pip install -r requirements.txt
```

para instalar las dependencias necesarias en el ambiente virtual

## Ejecución

### Ex 1 - Perceptrón simple con función de activación escalón

Para la función lógica **"AND"**:

```sh
python src/ex1/run_and.py configs/ex1/test.json
```

Para la función lógica **"XOR"**:

```sh
python src/ex1/run_xor.py configs/ex1/test.json
```

### Ex 2 - Comparación lineal vs no-lineal

**Modelo lineal:**
```sh
python src/ex2/runner.py configs/ex2/linear.json
```

**Modelo no lineal (tanh):**
```sh
python src/ex2/runner.py configs/ex2/nonlinear.json
```

### Ex 3 - Perceptrón multicapa

Para la función lógica **"XOR"** (3.1):

```sh
python3 src/ex3/runner_xor.py configs/ex3/xor.json
```

Para la función **par** (3.2):

```sh
python3 src/ex3/runner_parity.py configs/ex3/parity.json
```

Para la función **dígito** (3.3):

```sh
python src/ex3/runner_digit.py configs/ex3/digit.json
```

Para la función **dígito** con matriz de confusión (heatmap):
```sh
python src/ex3/digit_heatmap.py configs/ex3/digit.json
```

### Ex 4 - Perceptrón multicapa con dataset MNIST

Para la función dígito con el dataset **MNIST**:

```sh
python src/ex4/runner_mnist.py configs/ex4/mnist.json
```

### Análisis

Para el análisis de funciones de activación:

```sh
python src/analysis/activations.py configs/analysis/activations/config.json
```

Para el análisis de funciones de losses:

```sh
python src/analysis/losses.py configs/analysis/losses/[dataset].json
```

Para el análisis de dropout:

```sh
pyhton src/analysis/dropout.py [config_file]
```

### Optuna Search

Para buscar la mejor configuración de parámetros para cierto caso:

```sh
python src/common/searchers/optuna_search.py [digit|parity|mnist]
```