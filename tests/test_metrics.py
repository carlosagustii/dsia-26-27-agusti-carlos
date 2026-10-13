"""Tests de metrics.py con un DataFrame pequeño hecho a mano."""

import pandas as pd
import pytest

from football_shots.metrics import (
    maximos_goleadores,
    porcentaje_acierto,
    total_goles,
    total_tiros,
)


@pytest.fixture
def tiros() -> pd.DataFrame:
    """4 tiros: Messi marca 2, Benzema 1 y Aspas falla."""
    return pd.DataFrame(
        {
            "player": ["Lionel Messi", "Lionel Messi", "Karim Benzema", "Iago Aspas"],
            "is_goal": [1, 1, 1, 0],
        }
    )


def test_total_tiros(tiros):
    assert total_tiros(tiros) == 4


def test_total_goles(tiros):
    assert total_goles(tiros) == 3


def test_total_goles_devuelve_int(tiros):
    assert isinstance(total_goles(tiros), int)


def test_porcentaje_acierto(tiros):
    assert porcentaje_acierto(tiros) == 75.0


def test_porcentaje_acierto_redondea_a_2_decimales():
    df = pd.DataFrame({"player": ["A", "B", "C"], "is_goal": [1, 0, 0]})

    assert porcentaje_acierto(df) == 33.33


def test_porcentaje_acierto_sin_tiros_es_cero():
    df = pd.DataFrame({"player": [], "is_goal": []})

    assert porcentaje_acierto(df) == 0.0


def test_maximos_goleadores_ordenados(tiros):
    ranking = maximos_goleadores(tiros)

    assert list(ranking.index) == ["Lionel Messi", "Karim Benzema"]
    assert list(ranking) == [2, 1]


def test_maximos_goleadores_ignora_a_quien_no_marca(tiros):
    assert "Iago Aspas" not in maximos_goleadores(tiros).index


def test_maximos_goleadores_respeta_n(tiros):
    assert len(maximos_goleadores(tiros, n=1)) == 1
