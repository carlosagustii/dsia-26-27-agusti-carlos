"""Cliente HTTP de la API para el dashboard.

El dashboard solo habla con la API: no importa football_shots ni lee data/ o
models/. La URL de la API se configura con la variable de entorno API_URL.
"""

import os

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

TIMEOUT_S = 5
ANALYZE_TIMEOUT_S = 60  # el LLM puede tardar varios segundos


class ApiError(Exception):
    """Error de la API, con su código HTTP y el request_id para trazarlo en los logs."""

    def __init__(self, status: int | None, message: str, request_id: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.request_id = request_id

    def __str__(self) -> str:
        if self.status is None:
            return self.message
        return f"{self.message} (código {self.status}, request_id {self.request_id or '-'})"


def _request(method: str, path: str, timeout: float = TIMEOUT_S, allow_status: tuple[int, ...] = (), **kwargs):
    """Hace la petición y devuelve el JSON. Lanza ApiError si no es 2xx (salvo `allow_status`)."""
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.Timeout as exc:
        raise ApiError(None, f"La API no respondió en {timeout} s ({API_URL})") from exc
    except requests.RequestException as exc:
        raise ApiError(None, f"La API no responde en {API_URL}") from exc

    if response.ok or response.status_code in allow_status:
        return response.json()

    try:
        body = response.json()
        message = body.get("error", response.reason)
        request_id = body.get("request_id") or response.headers.get("X-Request-ID")
    except ValueError:
        message, request_id = response.reason or "Error desconocido", response.headers.get("X-Request-ID")
    raise ApiError(response.status_code, message, request_id)


def health() -> dict:
    """Estado del servicio. Un 503 (degraded) no es excepción: también es información."""
    return _request("GET", "/health", allow_status=(503,))


def predict(shot: dict) -> dict:
    return _request("POST", "/predict", json=shot)


def players(season: int) -> list[dict]:
    return _request("GET", "/players", params={"season": season})


def player_metrics(player_id: int, season: int) -> dict:
    return _request("GET", f"/players/{player_id}/metrics", params={"season": season})


def analyze(player_id: int, season: int) -> dict:
    return _request("POST", "/analyze", timeout=ANALYZE_TIMEOUT_S, json={"player_id": player_id, "season": season})


def metrics() -> dict:
    return _request("GET", "/metrics")
