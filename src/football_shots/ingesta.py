"""Ingesta: CSV crudo -> tiros válidos (processed) y filas rechazadas (rejected)."""

import logging
from pathlib import Path

import pandas as pd

from football_shots.loader import load
from football_shots.validator import validar_tiros

logger = logging.getLogger(__name__)

MOTIVO_DUPLICADO = "shot_id duplicado"


def ingest(input_path: Path, processed_dir: Path, rejected_dir: Path) -> pd.DataFrame:
    """Carga, valida y guarda. Devuelve el DataFrame de tiros válidos."""
    df = load(Path(input_path))
    logger.info("Filas cargadas: %d", len(df))

    validos, errores = validar_tiros(df)

    # Los duplicados no son errores: se quitan y se cuentan aparte.
    if errores.empty:
        es_duplicado = pd.Series(dtype=bool)
    else:
        es_duplicado = errores["motivo"] == MOTIVO_DUPLICADO
    invalidos = errores[~es_duplicado] if not errores.empty else errores

    logger.info("Duplicados eliminados: %d", int(es_duplicado.sum()))
    logger.info("Filas rechazadas: %d", len(invalidos))
    logger.info("Filas finales: %d", len(validos))

    processed_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    validos.to_csv(processed_dir / "shots.csv", index=False)
    invalidos.to_csv(rejected_dir / "invalid_rows.csv", index=False)

    return validos