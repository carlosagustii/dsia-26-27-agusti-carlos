"""Tests de validator.py: el contrato ShotRecord y la función validar_tiros."""

from datetime import datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from football_shots.validator import ShotRecord
from football_shots.validator import ValidationError as DatosInvalidosError
from football_shots.validator import validar_tiros


def test_tiro_valido_se_acepta(tiro_valido):
    tiro = ShotRecord(**tiro_valido)

    assert tiro.shot_id == 1
    assert tiro.date == datetime(2021, 5, 22, 16, 0, 0)


def test_columna_extra_se_rechaza(tiro_valido):
    with pytest.raises(ValidationError):
        ShotRecord(**{**tiro_valido, "columna_rara": 1})


@pytest.mark.parametrize("campo", ["shot_id", "match_id", "player_id"])
@pytest.mark.parametrize("valor", [0, -1])
def test_ids_no_positivos_se_rechazan(tiro_valido, campo, valor):
    with pytest.raises(ValidationError):
        ShotRecord(**{**tiro_valido, campo: valor})


def test_fecha_invalida_se_rechaza(tiro_valido):
    with pytest.raises(ValidationError):
        ShotRecord(**{**tiro_valido, "date": "no es una fecha"})


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("season", 2013),
        ("season", 2022),
        ("minute", -1),
        ("minute", 131),
        ("x", -0.1),
        ("x", 1.1),
        ("y", -0.1),
        ("y", 1.1),
        ("xg_understat", -0.1),
        ("xg_understat", 1.1),
    ],
)
def test_valores_fuera_de_rango_se_rechazan(tiro_valido, campo, valor):
    with pytest.raises(ValidationError):
        ShotRecord(**{**tiro_valido, campo: valor})


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("season", 2014),
        ("season", 2021),
        ("minute", 0),
        ("minute", 130),
        ("x", 0.0),
        ("x", 1.0),
        ("xg_understat", 0.0),
        ("xg_understat", 1.0),
    ],
)
def test_limites_de_rango_se_aceptan(tiro_valido, campo, valor):
    tiro = ShotRecord(**{**tiro_valido, campo: valor})

    assert getattr(tiro, campo) == valor


@pytest.mark.parametrize("campo", ["league", "h_team", "a_team", "player"])
def test_textos_vacios_se_rechazan(tiro_valido, campo):
    with pytest.raises(ValidationError):
        ShotRecord(**{**tiro_valido, campo: ""})


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("h_a", "x"),
        ("situation", "Corner"),
        ("shot_type", "Chest"),
        ("preferred_foot", "Left"),
        ("is_goal", 2),
    ],
)
def test_valores_no_permitidos_se_rechazan(tiro_valido, campo, valor):
    with pytest.raises(ValidationError):
        ShotRecord(**{**tiro_valido, campo: valor})


def test_last_action_puede_ser_none(tiro_valido):
    tiro = ShotRecord(**{**tiro_valido, "last_action": None})

    assert tiro.last_action is None


def test_last_action_puede_faltar(tiro_valido):
    sin_last_action = {k: v for k, v in tiro_valido.items() if k != "last_action"}

    tiro = ShotRecord(**sin_last_action)

    assert tiro.last_action is None


def test_preferred_foot_right_se_corrige(tiro_valido):
    tiro = ShotRecord(**{**tiro_valido, "preferred_foot": "Right"})

    assert tiro.preferred_foot == "RightFoot"


@pytest.mark.parametrize("pie", ["RightFoot", "LeftFoot", "both"])
def test_preferred_foot_validos_no_cambian(tiro_valido, pie):
    tiro = ShotRecord(**{**tiro_valido, "preferred_foot": pie})

    assert tiro.preferred_foot == pie


# --- validar_tiros ---


def test_validar_tiros_separa_validos_y_duplicados(tiros_df):
    validos, errores = validar_tiros(tiros_df)

    assert len(validos) == 2
    assert len(errores) == 1
    assert errores.iloc[0]["motivo"] == "shot_id duplicado"


def test_validar_tiros_fila_invalida_va_a_errores(tiro_valido):
    malo = {**tiro_valido, "shot_id": 2, "minute": 200}
    frame = pd.DataFrame([tiro_valido, malo])

    validos, errores = validar_tiros(frame)

    assert len(validos) == 1
    assert len(errores) == 1
    assert errores.iloc[0]["motivo"].startswith("minute")


def test_validar_tiros_nan_en_last_action_es_valido(tiro_valido):
    frame = pd.DataFrame([{**tiro_valido, "last_action": float("nan")}])

    validos, errores = validar_tiros(frame)

    assert len(validos) == 1
    assert len(errores) == 0
    assert validos.iloc[0]["last_action"] is None


def test_validar_tiros_corrige_right(tiro_valido):
    frame = pd.DataFrame([{**tiro_valido, "preferred_foot": "Right"}])

    validos, _ = validar_tiros(frame)

    assert validos.iloc[0]["preferred_foot"] == "RightFoot"


def test_validar_tiros_vacio_lanza_error():
    with pytest.raises(DatosInvalidosError):
        validar_tiros(pd.DataFrame())
