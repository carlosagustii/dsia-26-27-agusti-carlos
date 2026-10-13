"""Valida el DataFrame de tiros fila a fila usando el contrato ShotRecord.

- Reglas de UNA fila (tipos, rangos, valores permitidos) -> clase ShotRecord
- Reglas del CONJUNTO (shot_id duplicado) -> validar_tiros, porque una fila
  sola no puede saber si otra fila tiene su mismo id.

Valores reales que hay en data/shots_laliga.csv (salen de la exploración):
  - situation:      OpenPlay, FromCorner, SetPiece, DirectFreekick, Penalty
  - shot_type:      RightFoot, LeftFoot, Head, OtherBodyPart
  - preferred_foot: RightFoot, LeftFoot, both  (y "Right", que hay que corregir)
  - last_action:    texto, a veces vacío
"""

from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import ValidationError as PydanticValidationError


class ShotRecord(BaseModel):
    """Un tiro: una fila de data/shots_laliga.csv.

    Al crear un objeto, Pydantic comprueba los datos y lanza ValidationError
    si algo no cuadra.
    """

    # No se admiten columnas que no estén declaradas aquí.
    model_config = ConfigDict(extra="forbid")

    # Identificadores: enteros positivos.
    shot_id: int = Field(gt=0)
    match_id: int = Field(gt=0)
    player_id: int = Field(gt=0)

    # Fecha y hora del partido.
    date: datetime

    # Temporada entre 2014 y 2021.
    season: int = Field(ge=2014, le=2021)

    # Textos que no pueden estar vacíos.
    league: str = Field(min_length=1)
    h_team: str = Field(min_length=1)
    a_team: str = Field(min_length=1)
    player: str = Field(min_length=1)

    # Local ("h") o visitante ("a").
    h_a: Literal["h", "a"]

    # Minuto entre 0 y 130.
    minute: int = Field(ge=0, le=130)

    # Coordenadas normalizadas del campo.
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)

    # Solo los valores que aparecen en los datos.
    situation: Literal["OpenPlay", "FromCorner", "SetPiece", "DirectFreekick", "Penalty"]
    shot_type: Literal["RightFoot", "LeftFoot", "Head", "OtherBodyPart"]

    # Texto opcional: puede venir vacío (None).
    last_action: str | None = None

    preferred_foot: Literal["RightFoot", "LeftFoot", "both"]

    # 1 si fue gol, 0 si no.
    is_goal: int = Field(ge=0, le=1)

    # Probabilidad de gol.
    xg_understat: float = Field(ge=0, le=1)

    # En los datos aparece "Right" en vez de "RightFoot": se corrige antes de validar.
    @field_validator("preferred_foot", mode="before")
    @classmethod
    def normalize_foot(cls, value):
        if value == "Right":
            return "RightFoot"
        return value


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
