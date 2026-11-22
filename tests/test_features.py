"""Tests de features.py: geometría del tiro y preprocesado."""

import math

import numpy as np
import pandas as pd
import pytest

from football_shots.features import (
    NO_LAST_ACTION,
    add_geometry,
    build_preprocessor,
    shot_distance_angle,
)

# Punto de penalti: 11 m de la línea de gol, centrado.
PENALTY_SPOT_X = 1 - 11 / 105


def test_distancia_y_angulo_desde_el_punto_de_penalti():
    distancia, angulo = shot_distance_angle([PENALTY_SPOT_X], [0.5])

    assert distancia[0] == pytest.approx(11.0)
    # Ángulo entre los postes visto desde 11 m: 2 * atan(3.66 / 11).
    assert angulo[0] == pytest.approx(2 * math.atan(3.66 / 11))


@pytest.mark.parametrize(
    ("x", "y", "esperado"),
    [
        (1.0, 0.5, math.pi),  # en la línea de gol, entre los postes
        (1.0, 0.0, 0.0),  # en la línea de fondo, fuera de los postes
    ],
)
def test_angulo_en_casos_limite(x, y, esperado):
    _, angulo = shot_distance_angle([x], [y])

    assert angulo[0] == pytest.approx(esperado)


def test_mas_lejos_implica_menos_angulo():
    distancias, angulos = shot_distance_angle([0.95, 0.85, 0.70], [0.5, 0.5, 0.5])

    assert np.all(np.diff(distancias) > 0)
    assert np.all(np.diff(angulos) < 0)


def test_add_geometry_rellena_last_action_y_no_modifica_la_entrada():
    df = pd.DataFrame({"x": [0.9], "y": [0.5], "minute": [10], "last_action": [None]})

    out = add_geometry(df)

    assert out.loc[0, "last_action"] == NO_LAST_ACTION
    assert {"distance", "angle"} <= set(out.columns)
    assert "distance" not in df.columns


def test_preprocessor_genera_una_columna_por_categoria(shots_mini):
    datos = add_geometry(shots_mini)

    matriz = build_preprocessor(min_frequency=1).fit_transform(datos)

    n_categorias = sum(datos[c].nunique() for c in ["situation", "shot_type", "last_action"])
    assert matriz.shape == (len(datos), 3 + n_categorias)
