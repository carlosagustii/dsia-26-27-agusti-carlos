#!/usr/bin/env bash
# Smoke test de la API desplegada.
#   scripts/smoke_test.sh [BASE_URL]          (por defecto http://localhost:8000)
#   SMOKE_ANALYZE=1 scripts/smoke_test.sh URL (también prueba /analyze: gasta una llamada al LLM)
# Sale con código distinto de 0 si algún check falla. Solo necesita curl y python3.
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
BASE_URL="${BASE_URL%/}"
FAILED=0

ok() { echo "✅ $1"; }
ko() { echo "❌ $1"; FAILED=1; }

# Hace la petición y deja el cuerpo en $BODY y el código HTTP en $STATUS.
call() {
  local method="$1" path="$2" data="${3:-}"
  local tmp
  tmp="$(mktemp)"
  if [[ -n "$data" ]]; then
    STATUS="$(curl -sS -o "$tmp" -w '%{http_code}' -X "$method" -H 'Content-Type: application/json' \
      --max-time 60 -d "$data" "$BASE_URL$path" || true)"
  else
    STATUS="$(curl -sS -o "$tmp" -w '%{http_code}' -X "$method" --max-time 30 "$BASE_URL$path" || true)"
  fi
  BODY="$(cat "$tmp")"
  rm -f "$tmp"
}

# Evalúa una expresión de Python sobre el JSON de $BODY (variable `d`).
check_json() {
  python3 -c "import json, sys; d = json.loads(sys.argv[1]); sys.exit(0 if ($2) else 1)" "$BODY" 2>/dev/null
}

echo "Smoke test contra $BASE_URL"

call GET /health
if [[ "$STATUS" == "200" ]] && check_json "$BODY" "d['status'] == 'ok'"; then
  ok "/health → 200 status=ok"
else
  ko "/health → $STATUS $BODY"
fi

call POST /predict '{"x": 0.8952, "y": 0.5, "minute": 45, "situation": "OpenPlay", "shot_type": "RightFoot", "last_action": "Pass"}'
if [[ "$STATUS" == "200" ]] && check_json "$BODY" "0 < d['xg'] < 1"; then
  ok "/predict → 200 xg=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['xg'])" "$BODY")"
else
  ko "/predict → $STATUS $BODY"
fi

call GET "/players?season=2020"
if [[ "$STATUS" == "200" ]] && check_json "$BODY" "isinstance(d, list) and len(d) > 0"; then
  ok "/players?season=2020 → 200 con $(python3 -c "import json,sys; print(len(json.loads(sys.argv[1])))" "$BODY") jugadores"
else
  ko "/players?season=2020 → $STATUS"
fi

if [[ "${SMOKE_ANALYZE:-0}" == "1" ]]; then
  call POST /analyze '{"player_id": 2518, "season": 2020}'
  if [[ "$STATUS" == "200" ]] && check_json "$BODY" "len(d['report']) > 0"; then
    ok "/analyze → 200 provider=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['provider'])" "$BODY")"
  else
    ko "/analyze → $STATUS $BODY"
  fi
fi

if [[ "$FAILED" == "0" ]]; then
  echo "OK"
else
  echo "FALLO"
fi
exit "$FAILED"
