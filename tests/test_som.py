# ----------------------------------------------
# File: tests/test_som.py
# ----------------------------------------------
"""Unidad mínima para comprobar que SOM converge y transforma."""
import numpy as np

from common.kohonen.som import SOM


def test_fit_and_transform():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))

    som = SOM(5, 5, dim=4, n_iterations=500, random_state=0)
    som.fit(X)

    coords = som.transform(X)
    assert coords.shape == (80, 2)

    # Las coordenadas deben estar dentro de la grilla
    assert coords[:, 0].min() >= 0 and coords[:, 0].max() < som.m
    assert coords[:, 1].min() >= 0 and coords[:, 1].max() < som.n

    # El código libro (codebook) debe coincidir con pesos reshapeados
    assert som.codebook.shape == (som.m, som.n, 4)
