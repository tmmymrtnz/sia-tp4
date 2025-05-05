# TP3 SIA - Perceptrón Simple y Multicapa

[Enunciado](docs/Enunciado\ TP3.pdf)

## Instalación

Parado en la carpeta del tp3 ejecutar

```sh
pip install -r requirements.txt
```

para instalar las dependencias necesarias en el ambiente virtual

## Ejecución

### Ex 1

Para la función lógica **"AND"**:

```bash
python src/ex1/run_and.py configs/ex1/test.json
```

Para la función lógica **"XOR"**:

```bash
python src/ex1/run_xor.py configs/ex1/test.json
```

### Ex 2 - Comparación lineal vs no-lineal

**Modelo lineal:**
```bash
python src/ex2/runner.py configs/ex2/linear.json
```

**Modelo no lineal (tanh):**
```bash
python src/ex2/runner.py configs/ex2/nonlinear.json
```

### Ex 3

Para la función lógica **"XOR"**:

```bash
python3 src/ex3/runner.py configs/ex3/xor.json
```

Para la función par:

```bash
python3 src/ex3/runner_parity.py configs/ex3/parity.json
```

Para la función dígito:

```bash
python src/ex3/runner_digit.py configs/ex3/digit.json
```