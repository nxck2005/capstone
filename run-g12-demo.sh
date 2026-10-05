#!/usr/bin/env bash
# One-command launcher for the G-12 edition of the exhibition demo (demo_g12/):
# the same dashboard as ./run-demo.sh, with the chart showing the final test results.
#
#   ./run-g12-demo.sh                  start on port 8001 and open the browser
#   ./run-g12-demo.sh --port 8081      use another port
#   ./run-g12-demo.sh --no-browser     start the server only
#
# The first run builds the page if it is missing (needs Node/npm). Stop with Ctrl+C.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND="$ROOT/demo_g12/frontend"
PORT="${DEMO_PORT:-8001}"
OPEN_BROWSER=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --no-browser) OPEN_BROWSER=0; shift ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
  esac
done

# Python: an explicit DEMO_PYTHON wins, then the original demo's venv, then the project venv.
if [[ -n "${DEMO_PYTHON:-}" ]]; then
  PY="$DEMO_PYTHON"
elif [[ -x "$ROOT/demo/.venv-demo/bin/python" ]]; then
  PY="$ROOT/demo/.venv-demo/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  echo "No Python environment found. Follow the install steps in demo/README.md." >&2
  exit 1
fi
if ! "$PY" -c 'import fastapi, uvicorn' 2>/dev/null; then
  echo "$PY is missing the demo server packages." >&2
  echo "Run: $PY -m pip install -r demo/backend/requirements.txt" >&2
  exit 1
fi

if [[ ! -f "$FRONTEND/dist/index.html" ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "The dashboard is not built and npm is not installed. Install Node/npm, then rerun." >&2
    exit 1
  fi
  echo "First run: building the dashboard."
  (cd "$FRONTEND" && { [[ -d node_modules ]] || npm ci --silent; } && npm run build --silent)
fi

# Weights are shared with demo/ and optional: without them live predictions show as unavailable.
if ! "$PY" "$ROOT/demo/scripts/provision_assets.py" --verify >/dev/null 2>&1; then
  echo "Warning: model weights are missing or unverified, so live predictions will show as unavailable." >&2
  echo "         See 'Frozen weights' in demo/README.md." >&2
fi

if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :$PORT" | grep -q LISTEN; then
  echo "Port $PORT is already in use. Is the demo already running? Try: ./run-g12-demo.sh --port 8081" >&2
  exit 1
fi

URL="http://127.0.0.1:$PORT"
open_when_ready() {
  for _ in $(seq 1 60); do
    if curl -s -o /dev/null "$URL/" 2>/dev/null; then
      if command -v wslview >/dev/null 2>&1; then wslview "$URL"
      elif command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$URL"
      elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
      else echo "Open $URL in your browser."
      fi
      return
    fi
    sleep 1
  done
  echo "The server did not answer within 60 s; open $URL manually." >&2
}
if [[ "$OPEN_BROWSER" == 1 ]]; then
  open_when_ready >/dev/null 2>&1 &
fi

echo "Starting the G-12 demo at $URL (Ctrl+C to stop)"
cd "$ROOT"
exec "$PY" -m uvicorn demo_g12.scripts.serve:app --host 127.0.0.1 --port "$PORT" --app-dir "$ROOT"
