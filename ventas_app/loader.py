from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

class DataLoadError(Exception):
    """Error al cargar datos de origen."""


class ValidationError(Exception):
    """Datos que no cumplen reglas de negocio."""