"""Tests de loader.py: carga del CSV y sus errores."""

import pandas as pd
import pytest

from football_shots.loader import DataLoadError, load


def test_load_csv_correcto(tmp_path, tiro_valido):
    ruta = tmp_path / "tiros.csv"
    pd.DataFrame([tiro_valido]).to_csv(ruta, index=False)

    df = load(ruta)

    assert len(df) == 1
    assert df.iloc[0]["player"] == "Iago Aspas"


def test_load_fichero_que_no_existe(tmp_path):
    with pytest.raises(DataLoadError, match="No existe"):
        load(tmp_path / "no_existe.csv")


def test_load_fichero_solo_cabecera(tmp_path, tiro_valido):
    # Tiene columnas pero ninguna fila.
    ruta = tmp_path / "vacio.csv"
    pd.DataFrame(columns=list(tiro_valido)).to_csv(ruta, index=False)

    with pytest.raises(DataLoadError, match="vacío"):
        load(ruta)


def test_load_faltan_columnas(tmp_path, tiro_valido):
    sin_player = {k: v for k, v in tiro_valido.items() if k != "player"}
    ruta = tmp_path / "incompleto.csv"
    pd.DataFrame([sin_player]).to_csv(ruta, index=False)

    with pytest.raises(DataLoadError, match="player"):
        load(ruta)
