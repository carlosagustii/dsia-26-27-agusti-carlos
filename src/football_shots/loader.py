from pathlib import Path

import pandas as pd


class DataLoadError(Exception):
    """Error al cargar datos de origen."""


REQUIRED_COLUMNS = [
    "shot_id", "match_id", "date", "season", "league", "h_team", "a_team", "h_a",
    "player", "player_id", "minute", "x", "y", "situation", "shot_type",
    "last_action", "preferred_foot", "is_goal", "xg_understat",
]

def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataLoadError(f"No existe el fichero: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise DataLoadError(f"El fichero está vacío: {path}")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataLoadError(f"Faltan columnas: {missing}")

    return df