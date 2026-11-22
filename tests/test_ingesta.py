"""Tests de ingesta.py: duplicados aparte, rechazados con motivo y ficheros de salida."""

import pandas as pd
import pytest

from football_shots.ingesta import drop_duplicates, run_ingest
from football_shots.loader import DataLoadError


def test_drop_duplicates_cuenta_y_se_queda_con_el_primero(tiros_df):
    # tiros_df: 3 filas, la tercera repite el shot_id de la primera.
    sin_duplicados, n_duplicados = drop_duplicates(tiros_df)

    assert n_duplicados == 1
    assert list(sin_duplicados["shot_id"]) == [1, 2]


def test_run_ingest_separa_validos_rechazados_y_duplicados(tmp_path, tiro_valido):
    # Arrange: 2 válidos, 1 duplicado y 1 con x fuera de rango.
    filas = [
        tiro_valido,
        {**tiro_valido, "shot_id": 2},
        {**tiro_valido},
        {**tiro_valido, "shot_id": 3, "x": 1.5},
    ]
    entrada = tmp_path / "raw.csv"
    pd.DataFrame(filas).to_csv(entrada, index=False)
    salida = tmp_path / "processed" / "shots.csv"
    rechazados = tmp_path / "rejected" / "invalid.csv"

    # Act
    resultado = run_ingest(entrada, salida, rechazados)

    # Assert
    assert (resultado.loaded, resultado.duplicates, resultado.rejected, resultado.valid) == (4, 1, 1, 2)
    assert len(pd.read_csv(salida)) == 2
    invalidos = pd.read_csv(rechazados)
    assert invalidos.loc[0, "shot_id"] == 3
    assert invalidos.loc[0, "motivo"].startswith("x:")


def test_run_ingest_sin_rechazos_escribe_fichero_vacio_con_cabecera(tmp_path, shots_mini):
    entrada = tmp_path / "raw.csv"
    shots_mini.to_csv(entrada, index=False)
    rechazados = tmp_path / "invalid.csv"

    resultado = run_ingest(entrada, tmp_path / "shots.csv", rechazados)

    assert resultado.rejected == 0
    vacio = pd.read_csv(rechazados)
    assert vacio.empty
    assert "motivo" in vacio.columns


def test_run_ingest_fichero_inexistente(tmp_path):
    with pytest.raises(DataLoadError):
        run_ingest(tmp_path / "no_existe.csv", tmp_path / "a.csv", tmp_path / "b.csv")
