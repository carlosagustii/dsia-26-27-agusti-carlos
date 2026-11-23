"""Configuración del proyecto: rutas, variables de entorno (.env) y logging.

Todas las rutas se pueden cambiar con variables de entorno; los valores por
defecto son relativos a la raíz del repositorio.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Lee .env si existe (no falla si no está). Las variables ya definidas en el
# entorno tienen prioridad sobre las del fichero.
load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_CSV = DATA_DIR / "raw" / "shots_laliga.csv"
PROCESSED_DIR = DATA_DIR / "processed"
REJECTED_DIR = DATA_DIR / "rejected"

SHOTS_CSV = PROCESSED_DIR / "shots.csv"
SCORED_CSV = PROCESSED_DIR / "shots_scored.csv"
PLAYER_METRICS_CSV = PROCESSED_DIR / "player_metrics.csv"
INVALID_CSV = REJECTED_DIR / "invalid_rows.csv"

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/xg_model.joblib"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "reports"))

# Temporadas del split temporal: se entrena con una y se evalúa con la otra.
TRAIN_SEASON = 2019
TEST_SEASON = 2020

GENAI_PROVIDER = os.getenv("GENAI_PROVIDER", "mock")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "nvidia/nemotron-3.5-bolt-e4b")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
GENAI_TIMEOUT_S = float(os.getenv("GENAI_TIMEOUT_S", "30"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


def setup_logging(level: str | None = None) -> None:
    """Configura el logging de la stdlib para toda la aplicación."""
    logging.basicConfig(level=(level or LOG_LEVEL).upper(), format=LOG_FORMAT, force=True)
