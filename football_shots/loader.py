from pathlib import Path

import pandas as pd


class DataLoadError(Exception):
    """Error al cargar datos de origen."""


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataLoadError(f"No existe el fichero: {path}")

    return pd.read_csv(path)