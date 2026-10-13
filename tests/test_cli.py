"""Tests de cli.py: se ejecuta main() y se mira lo que imprime."""

import pytest

from football_shots.cli import main


def test_cli_con_csv_pequeno(tmp_path, tiros_df, capsys):
    # tiros_df tiene 2 tiros válidos y 1 duplicado.
    ruta = tmp_path / "tiros.csv"
    tiros_df.to_csv(ruta, index=False)

    main(["--input", str(ruta)])

    salida = capsys.readouterr().out
    assert "Tiros válidos: 2" in salida
    assert "Tiros inválidos: 1" in salida
    assert "Goles: 1" in salida
    assert "Acierto (%): 50.0" in salida
    assert "Iago Aspas" in salida


@pytest.mark.integration
def test_cli_con_datos_reales(capsys):
    # Usa data/shots_laliga.csv por defecto.
    main([])

    salida = capsys.readouterr().out
    assert "Tiros válidos: 16128" in salida
    assert "Tiros inválidos: 4704" in salida
