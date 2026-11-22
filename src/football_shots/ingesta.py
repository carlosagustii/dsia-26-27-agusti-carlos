"""Ingesta: CSV crudo -> tiros válidos (data/processed) + rechazados (data/rejected).

Pasos:
1. Carga con `loader.load` (comprueba columnas y fichero vacío).
2. Elimina duplicados por `shot_id` y los cuenta aparte: no son errores.
3. Valida cada fila con el contrato `ShotRecord`.
4. Guarda válidos e inválidos (con el motivo) en CSV.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from football_shots import config
from football_shots.loader import load
from football_shots.validator import validar_tiros

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    """Resumen de una ejecución de la ingesta."""

    loaded: int
    duplicates: int
    rejected: int
    valid: int


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Quita filas con `shot_id` repetido (se queda con la primera)."""
    deduplicated = df.drop_duplicates(subset="shot_id", keep="first")
    return deduplicated, len(df) - len(deduplicated)


def run_ingest(
    input_path: Path = config.RAW_CSV,
    output_path: Path = config.SHOTS_CSV,
    rejected_path: Path = config.INVALID_CSV,
) -> IngestResult:
    """Ejecuta la ingesta completa y devuelve los conteos de cada paso."""
    raw = load(Path(input_path))
    logger.info("ingest_loaded rows=%d path=%s", len(raw), input_path)

    deduplicated, n_duplicates = drop_duplicates(raw)
    logger.info("ingest_duplicates removed=%d", n_duplicates)

    valid, invalid = validar_tiros(deduplicated)
    logger.info("ingest_validated valid=%d rejected=%d", len(valid), len(invalid))

    output_path = Path(output_path)
    rejected_path = Path(rejected_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    valid.to_csv(output_path, index=False)
    # Se escribe siempre, aunque esté vacío, para que se vea que no hubo rechazos.
    invalid.reindex(columns=[*raw.columns, "motivo"]).to_csv(rejected_path, index=False)
    logger.info("ingest_saved valid_path=%s rejected_path=%s", output_path, rejected_path)

    return IngestResult(
        loaded=len(raw),
        duplicates=n_duplicates,
        rejected=len(invalid),
        valid=len(valid),
    )
