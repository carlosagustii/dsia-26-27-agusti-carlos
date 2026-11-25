#!/bin/sh
# Arranque del dashboard Streamlit en contenedores.
# Escucha en $PORT (Render lo inyecta) o en el 8501 en local.
set -e
exec streamlit run dashboard/app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true
