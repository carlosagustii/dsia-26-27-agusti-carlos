"""CLI del proyecto.

Comandos (pipeline del Proyecto II, en orden):
    ingest   CSV crudo -> data/processed/shots.csv + data/rejected/invalid_rows.csv
    train    shots.csv -> models/xg_model.joblib + .json + reports/calibration.png
    score    shots.csv + modelo -> shots_scored.csv + player_metrics.csv
    report   informe de scouting con IA generativa de un jugador
    run      ingest + train + score de una vez

    summary  resumen de validación y métricas (Proyecto I). Es el comando por
             defecto, así que `python -m football_shots.cli --input X` sigue funcionando.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from football_shots import config
from football_shots.genai import GenAIError, generate_report, get_client
from football_shots.ingesta import run_ingest
from football_shots.loader import DataLoadError, load
from football_shots.metrics import (
    maximos_goleadores,
    player_metrics,
    porcentaje_acierto,
    total_goles,
    total_tiros,
)
from football_shots.predictor import XGPredictor, score_shots
from football_shots.train import run_training
from football_shots.validator import validar_tiros

logger = logging.getLogger("football_shots.cli")


def cmd_summary(args) -> int:
    df = load(Path(args.input))
    validos, errores = validar_tiros(df)

    print("Tiros válidos:", total_tiros(validos))
    print("Tiros inválidos:", len(errores))
    print("Goles:", total_goles(validos))
    print("Acierto (%):", porcentaje_acierto(validos))
    print("Máximos goleadores:")
    print(maximos_goleadores(validos))
    return 0


def cmd_ingest(args) -> int:
    result = run_ingest(Path(args.input), Path(args.output), Path(args.rejected))
    print(
        f"Cargadas: {result.loaded} | Duplicados: {result.duplicates} | "
        f"Rechazadas: {result.rejected} | Válidas: {result.valid}"
    )
    return 0


def cmd_train(args) -> int:
    metadata = run_training(Path(args.shots), Path(args.model), Path(args.reports))
    model = metadata["test_metrics"]["model"]
    bench = metadata["test_metrics"]["xg_understat"]
    print(f"Modelo elegido: {metadata['model']['name']} {metadata['model']['params']}")
    print(f"penalty_xg: {metadata['penalty_xg']}")
    print(f"{'Métrica':<10}{'Modelo':>10}{'Understat':>12}")
    for key in ("log_loss", "brier", "roc_auc"):
        print(f"{key:<10}{model[key]:>10.4f}{bench[key]:>12.4f}")
    return 0


def cmd_score(args) -> int:
    predictor = XGPredictor.load(Path(args.model))
    shots = pd.read_csv(args.shots)
    scored = score_shots(predictor, shots)
    train_season = predictor.metadata.get("train_season", config.TRAIN_SEASON)
    metrics = player_metrics(scored, train_season=train_season)

    Path(args.scored).parent.mkdir(parents=True, exist_ok=True)
    Path(args.player_metrics).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.scored, index=False)
    metrics.to_csv(args.player_metrics, index=False)
    logger.info("score_saved scored=%s players=%d", args.scored, len(metrics))
    print(f"Tiros puntuados: {len(scored)} | Filas jugador-temporada: {len(metrics)}")
    return 0


def find_player(metrics: pd.DataFrame, season: int, player_id: int | None = None, name: str | None = None) -> dict:
    """Busca un jugador por id o por (parte del) nombre en una temporada.

    Si el nombre coincide con varios, se queda con el que más tiros tiene.
    """
    rows = metrics[metrics["season"] == season]
    if player_id is not None:
        rows = rows[rows["player_id"] == player_id]
    elif name:
        rows = rows[rows["player"].str.contains(name, case=False, regex=False)]
        rows = rows.sort_values("shots", ascending=False)
    if rows.empty:
        raise LookupError(f"No hay métricas para ese jugador en la temporada {season}")
    return rows.iloc[0].to_dict()


def cmd_report(args) -> int:
    metrics = pd.read_csv(args.player_metrics)
    player = find_player(metrics, args.season, args.player_id, args.player)
    print(generate_report(player, get_client(args.provider)))
    return 0


def cmd_run(args) -> int:
    cmd_ingest(args)
    cmd_train(args)
    return cmd_score(args)


PATH_OPTIONS = {
    "input": ("--input", config.RAW_CSV, "CSV crudo de tiros"),
    "output": ("--output", config.SHOTS_CSV, "tiros válidos"),
    "rejected": ("--rejected", config.INVALID_CSV, "filas rechazadas"),
    "shots": ("--shots", config.SHOTS_CSV, "tiros válidos (salida de ingest)"),
    "model": ("--model", config.MODEL_PATH, "ruta del .joblib"),
    "reports": ("--reports", config.REPORTS_DIR, "carpeta de informes"),
    "scored": ("--scored", config.SCORED_CSV, "tiros con xg_model"),
    "player_metrics": ("--player-metrics", config.PLAYER_METRICS_CSV, "métricas por jugador"),
}


def _add_paths(parser: argparse.ArgumentParser, *names: str) -> None:
    for name in names:
        flag, default, help_text = PATH_OPTIONS[name]
        parser.add_argument(flag, default=str(default), help=f"{help_text} (por defecto: %(default)s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="football_shots", description="Pipeline de xG de tiros de LaLiga")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING… (por defecto LOG_LEVEL)")
    sub = parser.add_subparsers(dest="command", required=True)

    _add_paths(sub.add_parser("summary", help="validación + métricas del Proyecto I"), "input")
    _add_paths(sub.add_parser("ingest", help="limpia y valida el CSV crudo"), "input", "output", "rejected")
    _add_paths(sub.add_parser("train", help="entrena y evalúa el modelo xG"), "shots", "model", "reports")
    _add_paths(
        sub.add_parser("score", help="puntúa tiros y agrega por jugador"),
        "shots", "model", "scored", "player_metrics",
    )
    _add_paths(sub.add_parser("run", help="ingest + train + score"), *PATH_OPTIONS)

    report = sub.add_parser("report", help="informe de scouting con IA de un jugador")
    _add_paths(report, "player_metrics")
    who = report.add_mutually_exclusive_group(required=True)
    who.add_argument("--player-id", type=int)
    who.add_argument("--player", help="nombre o parte del nombre, p. ej. 'Benzema'")
    report.add_argument("--season", type=int, default=config.TEST_SEASON)
    report.add_argument("--provider", choices=["mock", "nvidia"], default=None, help="por defecto GENAI_PROVIDER")
    return parser


HANDLERS = {
    "summary": cmd_summary,
    "ingest": cmd_ingest,
    "train": cmd_train,
    "score": cmd_score,
    "report": cmd_report,
    "run": cmd_run,
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Compatibilidad con el Proyecto I: sin comando se ejecuta `summary`.
    if not argv or (argv[0] not in HANDLERS and argv[0] not in ("-h", "--help", "--log-level")):
        argv = ["summary", *argv]

    args = build_parser().parse_args(argv)
    config.setup_logging(args.log_level)
    try:
        return HANDLERS[args.command](args)
    except (DataLoadError, LookupError, GenAIError, FileNotFoundError) as exc:
        logger.error("%s_failed error=%s", args.command, exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
