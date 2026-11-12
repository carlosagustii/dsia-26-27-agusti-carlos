"""Configuración del proyecto: lee el .env y prepara los logs."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Lee el fichero .env (si existe) y mete sus variables en os.environ.
load_dotenv()

# Rutas de datos, modelo e informes.
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_CSV = DATA_DIR / "raw" / "shots_laliga.csv"
PROCESSED_DIR = DATA_DIR / "processed"
REJECTED_DIR = DATA_DIR / "rejected"
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/xg_model.joblib"))
REPORTS_DIR = Path("reports")

# IA generativa.
GENAI_PROVIDER = os.getenv("GENAI_PROVIDER", "mock")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GENAI_TIMEOUT_S = float(os.getenv("GENAI_TIMEOUT_S", "30"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def setup_logging() -> None:
    """Formato común para todos los logs: hora, nivel, módulo y mensaje."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )