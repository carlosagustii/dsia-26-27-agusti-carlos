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


# --- Proyecto II: métricas por jugador ---

from football_shots.metrics import player_metrics  # noqa: E402


@pytest.fixture
def tiros_puntuados() -> pd.DataFrame:
    """Messi 2020: 3 tiros (1 penalti marcado, 1 gol de jugada, 1 fallo)."""
    return pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2],
            "player": ["Lionel Messi"] * 3 + ["Iago Aspas"],
            "season": [2020, 2020, 2020, 2019],
            "situation": ["Penalty", "OpenPlay", "OpenPlay", "OpenPlay"],
            "is_goal": [1, 1, 0, 0],
            "xg_model": [0.8, 0.3, 0.1, 0.2],
            "xg_understat": [0.76, 0.25, 0.05, 0.1],
        }
    )


def test_player_metrics_separa_penaltis(tiros_puntuados):
    fila = player_metrics(tiros_puntuados).set_index("player_id").loc[1]

    assert (fila["shots"], fila["np_shots"], fila["goals"], fila["np_goals"]) == (3, 2, 2, 1)
    assert fila["xg"] == pytest.approx(1.2)
    assert fila["npxg"] == pytest.approx(0.4)
    assert fila["np_goals_minus_npxg"] == pytest.approx(0.6)
    assert fila["npxg_per_shot"] == pytest.approx(0.2)


def test_player_metrics_marca_in_sample_la_temporada_de_train(tiros_puntuados):
    tabla = player_metrics(tiros_puntuados, train_season=2019).set_index("player_id")

    assert bool(tabla.loc[2, "in_sample"]) is True
    assert bool(tabla.loc[1, "in_sample"]) is False


def test_player_metrics_jugador_solo_con_penaltis_no_divide_por_cero():
    df = pd.DataFrame(
        {
            "player_id": [9], "player": ["X"], "season": [2020], "situation": ["Penalty"],
            "is_goal": [1], "xg_model": [0.8], "xg_understat": [0.76],
        }
    )

    assert player_metrics(df).loc[0, "npxg_per_shot"] == 0.0
