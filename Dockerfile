# Imagen única para la API (CMD por defecto) y el dashboard (otro comando).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# El código va antes del pip install: requirements.txt incluye `-e .`.
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# data/processed está en .gitignore: se regenera aquí a partir de data/raw.
# No se entrena: se usa el modelo del repo (models/).
RUN python -m football_shots.cli ingest && python -m football_shots.cli score

# Usuario sin privilegios.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8000 8501

# Solo comprueba la API (el servicio del dashboard lo desactiva en docker-compose).
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\", \"8000\")}/health', timeout=4)"

# Render inyecta $PORT; en local se usa el 8000.
CMD ["sh", "-c", "uvicorn football_shots.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
