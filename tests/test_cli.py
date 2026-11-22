"""Tests de cli.py: se ejecuta main() y se mira lo que imprime."""

import pandas as pd
import pytest

from football_shots.cli import find_player, main


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
    # Usa data/raw/shots_laliga.csv por defecto.
    main([])

    salida = capsys.readouterr().out
    assert "Tiros válidos: 16128" in salida
    assert "Tiros inválidos: 4704" in salida


# --- Proyecto II: comandos ingest / train / score / report ---


def test_cli_pipeline_completo_con_datos_mini(tmp_path, shots_mini, capsys):
    """ingest -> train -> score -> report sobre 50 tiros, todo en tmp_path."""
    raw = tmp_path / "raw.csv"
    shots_mini.to_csv(raw, index=False)
    rutas = {
        "--input": raw,
        "--output": tmp_path / "shots.csv",
        "--rejected": tmp_path / "invalid.csv",
        "--model": tmp_path / "xg.joblib",
        "--reports": tmp_path / "reports",
        "--scored": tmp_path / "scored.csv",
        "--player-metrics": tmp_path / "players.csv",
    }

    def args(*flags):
        return [x for f in flags for x in (f, str(rutas[f]))]

    assert main(["ingest", *args("--input", "--output", "--rejected")]) == 0
    assert main(["train", "--shots", str(rutas["--output"]), *args("--model", "--reports")]) == 0
    assert main(["score", "--shots", str(rutas["--output"]), *args("--model", "--scored", "--player-metrics")]) == 0
    salida = capsys.readouterr().out
    assert "Válidas: 50" in salida
    assert "Modelo elegido:" in salida
    assert "Tiros puntuados: 50" in salida

    jugador = shots_mini[shots_mini["season"] == 2020].iloc[0]
    codigo = main(
        ["report", "--player-id", str(jugador["player_id"]), "--provider", "mock", *args("--player-metrics")]
    )
    assert codigo == 0
    assert "modo mock" in capsys.readouterr().out


def test_cli_report_jugador_inexistente_devuelve_1(tmp_path, capsys):
    metricas = tmp_path / "players.csv"
    pd.DataFrame(
        {"player_id": [1], "player": ["A"], "season": [2020], "shots": [1]}
    ).to_csv(metricas, index=False)

    codigo = main(["report", "--player", "Nadie", "--player-metrics", str(metricas), "--provider", "mock"])

    assert codigo == 1
    assert "No hay métricas" in capsys.readouterr().err


def test_cli_score_sin_modelo_devuelve_1(tmp_path, shots_mini, capsys):
    shots = tmp_path / "shots.csv"
    shots_mini.to_csv(shots, index=False)

    codigo = main(["score", "--shots", str(shots), "--model", str(tmp_path / "no.joblib")])

    assert codigo == 1
    assert "train" in capsys.readouterr().err


def test_find_player_por_nombre_parcial_elige_el_de_mas_tiros():
    metricas = pd.DataFrame(
        {
            "player_id": [1, 2],
            "player": ["Iñaki Williams", "Nico Williams"],
            "season": [2020, 2020],
            "shots": [80, 10],
        }
    )

    assert find_player(metricas, 2020, name="williams")["player_id"] == 1
