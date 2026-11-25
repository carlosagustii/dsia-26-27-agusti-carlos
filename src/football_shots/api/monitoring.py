"""Métricas de operación de la API, en memoria (un solo worker, sin Prometheus).

`MetricsStore` guarda contadores, latencias y las últimas peticiones. Es
thread-safe: FastAPI ejecuta los endpoints síncronos en un pool de hilos.
"""

import statistics
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

LATENCY_WINDOW = 500  # latencias guardadas por ruta para los percentiles
RECENT_SIZE = 20  # últimas peticiones que se muestran en el dashboard


def percentiles(values) -> tuple[float | None, float | None]:
    """p50 y p95 (ms, 1 decimal). None si no hay datos."""
    data = list(values)
    if not data:
        return None, None
    if len(data) == 1:
        return round(data[0], 1), round(data[0], 1)
    cuts = statistics.quantiles(data, n=100, method="inclusive")
    return round(cuts[49], 1), round(cuts[94], 1)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _RouteStats:
    def __init__(self):
        self.count = 0
        self.ok_2xx = 0
        self.errors_4xx = 0
        self.errors_5xx = 0
        self.latencies = deque(maxlen=LATENCY_WINDOW)


class MetricsStore:
    """Contadores por ruta, latencia de la IA y últimas peticiones."""

    def __init__(self):
        self._lock = threading.Lock()
        self._started_monotonic = time.monotonic()
        self.started_at = _now_iso()
        self._routes: dict[str, _RouteStats] = defaultdict(_RouteStats)
        self._recent = deque(maxlen=RECENT_SIZE)
        self._genai_latencies = deque(maxlen=LATENCY_WINDOW)
        self._genai_calls = 0
        self._genai_failures = 0
        self._genai_provider: str | None = None

    def record(self, route: str, method: str, status: int, latency_ms: float, request_id: str) -> None:
        """Registra una petición HTTP ya respondida."""
        with self._lock:
            stats = self._routes[route]
            stats.count += 1
            if 200 <= status < 300:
                stats.ok_2xx += 1
            elif 400 <= status < 500:
                stats.errors_4xx += 1
            elif status >= 500:
                stats.errors_5xx += 1
            stats.latencies.append(latency_ms)
            self._recent.append(
                {
                    "ts": _now_iso(),
                    "request_id": request_id,
                    "method": method,
                    "route": route,
                    "status": status,
                    "latency_ms": round(latency_ms, 1),
                }
            )

    def record_genai(self, latency_ms: float, ok: bool, provider: str) -> None:
        """Registra una llamada al LLM (medida aparte, dentro de /analyze)."""
        with self._lock:
            self._genai_calls += 1
            if not ok:
                self._genai_failures += 1
            self._genai_latencies.append(latency_ms)
            self._genai_provider = provider

    def snapshot(self) -> dict:
        """Foto de todas las métricas, lista para serializar a JSON."""
        with self._lock:
            requests = {}
            total = total_errors = 0
            for key, stats in sorted(self._routes.items()):
                errors = stats.errors_4xx + stats.errors_5xx
                p50, p95 = percentiles(stats.latencies)
                requests[key] = {
                    "count": stats.count,
                    "errors_4xx": stats.errors_4xx,
                    "errors_5xx": stats.errors_5xx,
                    "error_rate": round(errors / stats.count, 4) if stats.count else 0.0,
                    "p50_ms": p50,
                    "p95_ms": p95,
                }
                total += stats.count
                total_errors += errors

            genai_p50, genai_p95 = percentiles(self._genai_latencies)
            return {
                "uptime_s": round(time.monotonic() - self._started_monotonic, 1),
                "started_at": self.started_at,
                "requests": requests,
                "totals": {
                    "count": total,
                    "error_rate": round(total_errors / total, 4) if total else 0.0,
                },
                "genai": {
                    "provider": self._genai_provider,
                    "calls": self._genai_calls,
                    "failures": self._genai_failures,
                    "p50_ms": genai_p50,
                    "p95_ms": genai_p95,
                },
                "recent": list(reversed(self._recent)),
            }
