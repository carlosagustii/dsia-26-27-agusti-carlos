"""Contrato de datos: describe qué es un tiro válido.

ShotRecord es un modelo de Pydantic: una clase con campos y tipos. Al crear un
objeto, Pydantic comprueba los datos y lanza ValidationError si algo no cuadra.

Valores reales que hay en data/shots_laliga.csv (salen de la exploración):
  - situation:      OpenPlay, FromCorner, SetPiece, DirectFreekick, Penalty
  - shot_type:      RightFoot, LeftFoot, Head, OtherBodyPart
  - preferred_foot: RightFoot, LeftFoot, both  (y "Right", que hay que corregir)
  - last_action:    texto, a veces vacío
"""

from datetime import datetime  # noqa: F401  (lo usarás en los TODO)
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator  # noqa: F401


class ShotRecord(BaseModel):
    """Un tiro: una fila de data/shots_laliga.csv."""

    # Ya hecho: no se admiten columnas que no estén declaradas en este modelo.
    model_config = ConfigDict(extra="forbid")

    # Ejemplo ya hecho: entero positivo.
    shot_id: int = Field(gt=0)

    # Ejemplo ya hecho: solo dos valores posibles.
    h_a: Literal["h", "a"]

    # TODO 1: match_id y player_id. Enteros positivos (como shot_id).

    # TODO 2: date. Fecha y hora del partido (tipo datetime).

    # TODO 3: season. Entero entre 2014 y 2021.

    # TODO 4: league, h_team, a_team y player. Textos que no pueden estar vacíos.

    # TODO 5: minute. Entero entre 0 y 130.

    # TODO 6: x, y. Decimales entre 0 y 1 (coordenadas normalizadas del campo).

    # TODO 7: situation y shot_type. Solo los valores de la lista de arriba (Literal).

    # TODO 8: last_action. Texto OPCIONAL: puede venir vacío (None).

    # TODO 9: preferred_foot. Solo RightFoot, LeftFoot o both (Literal).

    # TODO 10: is_goal. Entero que solo puede valer 0 o 1.

    # TODO 11: xg_understat. Decimal entre 0 y 1 (es una probabilidad).

    # TODO 12: en los datos aparece "Right" en vez de "RightFoot". Haz un
    #          field_validator para preferred_foot (mode="before") que lo corrija:
    #
    # @field_validator("preferred_foot", mode="before")
    # @classmethod
    # def normalize_foot(cls, value):
    #     if value == ...:
    #         return ...
    #     return value
