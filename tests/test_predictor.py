"""Tests de predictor.py: carga del modelo e inferencia por las dos rutas."""

import pandas as pd
import pytest

from football_shots.predictor import (
    SOURCE_MODEL,
    SOURCE_PENALTY,
    ModelNotFoundError,
    XGPredictor,
    score_shots,
)

TIRO_JUGADA = {
    "x": 0.9,
    "y": 0.5,
    "minute": 30,
    "situation": "OpenPlay",
    "shot_type": "RightFoot",
    "last_action": "Pass",
}


def test_load_sin_modelo_lanza_error_claro(tmp_path):
    with pytest.raises(ModelNotFoundError, match="train"):
        XGPredictor.load(tmp_path / "no_existe.joblib")


def test_load_carga_pipeline_y_penalty_xg(mini_predictor):
    assert hasattr(mini_predictor.pipeline, "predict_proba")
    assert 0 <= mini_predictor.penalty_xg <= 1


def test_ruta_modelo_devuelve_probabilidad(mini_predictor):
    resultado = mini_predictor.predict_one(TIRO_JUGADA)

    assert resultado["source"] == SOURCE_MODEL
    assert 0 < resultado["xg"] < 1


def test_ruta_penalti_devuelve_penalty_xg(mini_predictor):
    resultado = mini_predictor.predict_one({**TIRO_JUGADA, "situation": "Penalty"})

    assert resultado == {"xg": round(mini_predictor.penalty_xg, 4), "source": SOURCE_PENALTY}


def test_last_action_es_opcional(mini_predictor):
    sin_accion = {k: v for k, v in TIRO_JUGADA.items() if k != "last_action"}

    assert mini_predictor.predict_one(sin_accion)["source"] == SOURCE_MODEL


def test_categoria_nueva_no_rompe_la_prediccion(mini_predictor):
    # handle_unknown en el OneHotEncoder: una acción nunca vista no lanza error.
    resultado = mini_predictor.predict_one({**TIRO_JUGADA, "last_action": "AccionInventada"})

    assert 0 < resultado["xg"] < 1


def test_predict_sin_columnas_lanza_error(mini_predictor):
    with pytest.raises(ValueError, match="Faltan columnas"):
        mini_predictor.predict(pd.DataFrame({"x": [0.9]}))


def test_score_shots_mezcla_las_dos_rutas(mini_predictor, shots_mini):
    puntuados = score_shots(mini_predictor, shots_mini)

    assert len(puntuados) == len(shots_mini)
    assert (puntuados["xg_model"].between(0, 1)).all()
    penaltis = puntuados[puntuados["situation"] == "Penalty"]
    assert (penaltis["xg_source"] == SOURCE_PENALTY).all()
    assert (penaltis["xg_model"] == mini_predictor.penalty_xg).all()
    assert (puntuados.loc[puntuados["situation"] != "Penalty", "xg_source"] == SOURCE_MODEL).all()


def test_mas_cerca_de_porteria_mas_xg(mini_predictor):
    cerca = mini_predictor.predict_one({**TIRO_JUGADA, "x": 0.95})["xg"]
    lejos = mini_predictor.predict_one({**TIRO_JUGADA, "x": 0.70})["xg"]

    assert cerca > lejos
