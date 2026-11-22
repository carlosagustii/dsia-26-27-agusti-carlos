"""IA generativa: informe de scouting a partir de las métricas de un jugador.

Proveedores (variable `GENAI_PROVIDER`):
- `mock` (por defecto): plantilla determinista. Sin red, sin coste, sin claves.
  Es el que usan los tests y el CI.
- `nvidia`: LLM servido por NVIDIA NIM (build.nvidia.com), API compatible con
  OpenAI. Necesita `NVIDIA_API_KEY` (empieza por `nvapi-`).

Ambos implementan la misma interfaz `generate_report(metrics) -> str`, así que
el resto del código (CLI y, en el Proyecto III, la API) no sabe cuál se usa.
"""

import logging
from typing import Protocol

from football_shots import config

logger = logging.getLogger(__name__)


class GenAIError(Exception):
    """El proveedor de IA devolvió un error (en la API será un 502)."""


class GenAITimeoutError(GenAIError):
    """El proveedor de IA superó el timeout (en la API será un 504)."""


class GenAIConfigError(GenAIError):
    """Proveedor desconocido o falta la API key."""


SYSTEM_PROMPT = (
    "Eres un analista de scouting de fútbol. Escribes en español, en un tono "
    "profesional y conciso (máximo 150 palabras). Reglas obligatorias:\n"
    "1. Usa SOLO las cifras que te da el usuario; no inventes datos, partidos ni contexto.\n"
    "2. Distingue xG (incluye penaltis) de npxG (sin penaltis).\n"
    "3. Avisa de que no hay minutos jugados, así que no hay métricas por 90 minutos.\n"
    "4. Si in_sample es true, avisa de que el modelo se entrenó con esa temporada "
    "y las cifras son optimistas.\n"
    "5. Termina con una conclusión de una frase sobre su eficacia de cara a puerta."
)


def build_prompt(metrics: dict) -> str:
    """Mensaje de usuario con las cifras del jugador (solo los campos necesarios)."""
    season = int(metrics["season"])
    return (
        f"Jugador: {metrics['player']} (id {metrics['player_id']})\n"
        f"Temporada: {season}-{str(season + 1)[-2:]} (LaLiga)\n"
        f"Tiros: {metrics['shots']} (sin penaltis: {metrics['np_shots']})\n"
        f"Goles: {metrics['goals']} (sin penaltis: {metrics['np_goals']})\n"
        f"xG del modelo: {metrics['xg']:.2f} | npxG: {metrics['npxg']:.2f}\n"
        f"xG de Understat (referencia): {metrics['xg_understat']:.2f}\n"
        f"Goles - xG: {metrics['goals_minus_xg']:+.2f} | "
        f"Goles sin penaltis - npxG: {metrics['np_goals_minus_npxg']:+.2f}\n"
        f"npxG por tiro: {metrics['npxg_per_shot']:.3f}\n"
        f"in_sample: {str(bool(metrics['in_sample'])).lower()}\n\n"
        "Escribe el informe de scouting."
    )


class GenAIClient(Protocol):
    """Interfaz común de los proveedores."""

    name: str

    def generate_report(self, metrics: dict) -> str: ...


class MockClient:
    """Informe con plantilla fija: misma entrada -> mismo texto."""

    name = "mock"

    def generate_report(self, metrics: dict) -> str:
        diff = metrics["np_goals_minus_npxg"]
        if diff > 1:
            verdict = "rinde por encima de la calidad de sus ocasiones"
        elif diff < -1:
            verdict = "rinde por debajo de la calidad de sus ocasiones"
        else:
            verdict = "marca en línea con la calidad de sus ocasiones"

        season = int(metrics["season"])
        text = (
            f"{metrics['player']} ({season}-{str(season + 1)[-2:]}): {metrics['goals']} goles "
            f"en {metrics['shots']} tiros, con {metrics['xg']:.2f} xG y {metrics['npxg']:.2f} npxG "
            f"(Understat: {metrics['xg_understat']:.2f}). Sin penaltis suma {metrics['np_goals']} "
            f"goles frente a {metrics['npxg']:.2f} npxG ({diff:+.2f}), así que {verdict}. "
            f"Genera {metrics['npxg_per_shot']:.3f} npxG por tiro. "
            "Aviso: no hay minutos jugados, por lo que no se calculan métricas por 90'."
        )
        if metrics["in_sample"]:
            text += " Aviso: esta temporada se usó para entrenar el modelo (in-sample)."
        return text + " [informe generado en modo mock]"


class NvidiaClient:
    """LLM de NVIDIA NIM a través del cliente oficial de OpenAI (misma API)."""

    name = "nvidia"

    def __init__(
        self,
        api_key: str = config.NVIDIA_API_KEY,
        model: str = config.NVIDIA_MODEL,
        base_url: str = config.NVIDIA_BASE_URL,
        timeout_s: float = config.GENAI_TIMEOUT_S,
    ):
        if not api_key:
            raise GenAIConfigError("Falta NVIDIA_API_KEY en el entorno (.env)")
        # Import aquí: el modo mock no necesita cargar el SDK.
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=1)

    def generate_report(self, metrics: dict) -> str:
        import openai

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(metrics)},
                ],
                temperature=0.3,
                max_tokens=400,
            )
        except openai.APITimeoutError as exc:
            raise GenAITimeoutError(f"NVIDIA no respondió a tiempo: {exc}") from exc
        except openai.APIError as exc:
            raise GenAIError(f"Error del proveedor NVIDIA: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise GenAIError("NVIDIA devolvió una respuesta vacía")
        return content.strip()


def get_client(provider: str | None = None) -> GenAIClient:
    """Devuelve el cliente según `GENAI_PROVIDER` (o el argumento)."""
    provider = (provider or config.GENAI_PROVIDER).lower()
    if provider == "mock":
        return MockClient()
    if provider == "nvidia":
        return NvidiaClient()
    raise GenAIConfigError(f"GENAI_PROVIDER desconocido: {provider!r} (usa 'mock' o 'nvidia')")


def generate_report(metrics: dict, client: GenAIClient | None = None) -> str:
    """Genera el informe y registra proveedor y longitud."""
    client = client or get_client()
    report = client.generate_report(metrics)
    logger.info("genai_report provider=%s player_id=%s chars=%d", client.name, metrics["player_id"], len(report))
    return report
