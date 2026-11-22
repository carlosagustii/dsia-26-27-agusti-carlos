"""Tests de train.py y evaluate.py con el dataset mini (sin depender de models/)."""

import json

import numpy as np
import pandas as pd
import pytest

from football_shots.evaluate import evaluate, plot_calibration
from football_shots.features import INPUT_COLUMNS
from football_shots.train import (
    ModelConfig,
    fit_pipeline,
    make_estimator,
    penalty_conversion,
    run_training,
    select_model,
    split_penalties,
)


def test_split_penalties(shots_mini):
    sin_penaltis, penaltis = split_penalties(shots_mini)

    assert len(penaltis) == 4
    assert (penaltis["situation"] == "Penalty").all()
    assert len(sin_penaltis) + len(penaltis) == len(shots_mini)


def test_penalty_conversion_es_la_tasa_de_gol():
    penaltis = pd.DataFrame({"is_goal": [1, 1, 1, 0]})

    assert penalty_conversion(penaltis) == 0.75


def test_penalty_conversion_sin_penaltis_lanza_error():
    with pytest.raises(ValueError, match="penaltis"):
        penalty_conversion(pd.DataFrame({"is_goal": []}))


def test_modelo_desconocido_lanza_error():
    with pytest.raises(ValueError, match="desconocido"):
        make_estimator(ModelConfig("red_neuronal"))


@pytest.mark.parametrize(
    "model_config",
    [ModelConfig("logistic", {"C": 1.0}), ModelConfig("hist_gb", {"max_iter": 20})],
)
def test_fit_pipeline_devuelve_probabilidades(shots_mini, model_config):
    sin_penaltis, _ = split_penalties(shots_mini)

    pipeline = fit_pipeline(sin_penaltis, model_config, min_frequency=1)
    probs = pipeline.predict_proba(sin_penaltis[INPUT_COLUMNS])[:, 1]

    assert probs.shape == (len(sin_penaltis),)
    assert np.all((probs >= 0) & (probs <= 1))


def test_select_model_elige_el_de_menor_log_loss(shots_mini):
    sin_penaltis, _ = split_penalties(shots_mini)
    candidatos = [ModelConfig("logistic", {"C": 0.01}), ModelConfig("logistic", {"C": 1.0})]

    elegido, resultados = select_model(sin_penaltis, candidatos, n_splits=3)

    assert len(resultados) == 2
    mejor = min(resultados, key=lambda r: r["cv_log_loss"])
    assert elegido == ModelConfig(mejor["name"], mejor["params"])


def test_evaluate_con_resultado_conocido():
    metricas = evaluate([0, 1], [0.0, 1.0])

    assert metricas["brier"] == 0.0
    assert metricas["roc_auc"] == 1.0
    assert metricas["log_loss"] == pytest.approx(0.0, abs=1e-4)


def test_plot_calibration_crea_el_png(tmp_path):
    y = np.array([0, 0, 1, 1] * 10)
    ruta = plot_calibration(y, {"modelo": np.linspace(0.1, 0.9, 40)}, tmp_path / "cal.png", n_bins=4)

    assert ruta.exists() and ruta.stat().st_size > 0


def test_run_training_guarda_modelo_metadatos_y_calibracion(tmp_path, shots_mini):
    shots = tmp_path / "shots.csv"
    shots_mini.to_csv(shots, index=False)
    modelo = tmp_path / "models" / "xg.joblib"

    # Un solo candidato y 3 pliegues para que sea rápido con 50 tiros.
    metadata = run_training(
        shots, modelo, tmp_path / "reports", candidates=[ModelConfig("logistic", {"C": 1.0})], n_splits=3
    )
    penaltis_2019 = shots_mini[(shots_mini["season"] == 2019) & (shots_mini["situation"] == "Penalty")]

    assert modelo.exists()
    guardado = json.loads(modelo.with_suffix(".json").read_text())
    assert guardado["penalty_xg"] == metadata["penalty_xg"] == round(penaltis_2019["is_goal"].mean(), 4)
    assert set(guardado["test_metrics"]) == {"model", "xg_understat"}
    assert (tmp_path / "reports" / "calibration.png").exists()
