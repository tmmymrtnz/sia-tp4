# TP4 SIA - Aprendizaje No Supervisado

[Enunciado](docs/sia-tp4.pdf)

## Instalación

Parado en la carpeta del tp4 ejecutar

```sh
pip install -r requirements.txt
```

para instalar las dependencias necesarias en el ambiente virtual

## Ejecución

### Ex 1.1 - Red de Kohonen

```sh
python -m src.tp4.ex1.runner_kohonen.py configs/tp4/kohonen/kohonen_europe.json
```

### Ex 1.2 Modelo de Oja

### Ex 2.1 Modelo de Hopfield

## Tests

### Red de Kohonen

```sh
pytest tests/test_som.py -q
```