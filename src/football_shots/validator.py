"""Valida el DataFrame de tiros fila a fila usando el contrato ShotRecord.

- Reglas de UNA fila (tipos, rangos, valores permitidos) -> contracts.py
- Reglas del CONJUNTO (shot_id duplicado) -> aquí, porque una fila sola no
  puede saber si otra fila tiene su mismo id.
"""

import pandas as pd
from pydantic import ValidationError as PydanticValidationError

from football_shots.contracts import ShotRecord


class ValidationError(Exception):
    """Datos que no cumplen reglas de negocio."""


def validar_tiros(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (validos, errores). Los errores llevan una columna 'motivo'."""
    if frame.empty:
        raise ValidationError("No hay tiros que validar")

    # pandas guarda los vacíos como NaN; Pydantic espera None.
    filas = frame.astype(object).where(frame.notna(), None).to_dict("records")

    validos = []
    errores = []
    ids_vistos = set()

    for fila in filas:
        try:
            tiro = ShotRecord(**fila)
        except PydanticValidationError as exc:
            primer_error = exc.errors()[0]
            motivo = f"{primer_error['loc'][0]}: {primer_error['msg']}"
            errores.append({**fila, "motivo": motivo})
            continue

        if tiro.shot_id in ids_vistos:
            errores.append({**fila, "motivo": "shot_id duplicado"})
            continue

        ids_vistos.add(tiro.shot_id)
        validos.append(tiro.model_dump())

    return pd.DataFrame(validos), pd.DataFrame(errores)
