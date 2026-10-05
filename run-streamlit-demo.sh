#!/usr/bin/env bash
# One-command launcher for the Streamlit demo (demo_streamlit/).
#
#   ./run-streamlit-demo.sh                start on port 8501 and open the browser
#   ./run-streamlit-demo.sh --port 8502    use another port
#   ./run-streamlit-demo.sh --no-browser   start the server only
#
# The first run creates demo_streamlit/.venv from the hashed CPU lock (needs
# internet, a few minutes); every later run works offline. Stop with Ctrl+C.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$ROOT/demo_streamlit"
VENV="$APP/.venv"
PORT=8501
OPEN_BROWSER=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --no-browser) OPEN_BROWSER=0; shift ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
  esac
done

PY="${DEMO_PYTHON:-$VENV/bin/python}"
if ! "$PY" -c 'import streamlit, fastapi, torch' 2>/dev/null; then
  if [[ -n "${DEMO_PYTHON:-}" ]]; then
    echo "$PY lacks the demo packages. Run: $PY -m pip install -r demo_streamlit/requirements.txt" >&2
    exit 1
  fi
  echo "First run: creating $VENV (needs internet, a few minutes)."
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --require-hashes -r "$ROOT/requirements-cpu.lock"
  "$VENV/bin/python" -m pip install --quiet -r "$APP/requirements.txt"
  PY="$VENV/bin/python"
fi

# Weights are optional: without them the page still runs and marks live outputs unavailable.
if ! "$PY" "$ROOT/demo/scripts/provision_assets.py" --verify >/dev/null 2>&1; then
  echo "Warning: model weights are missing or unverified, so live transmissions will show as unavailable." >&2
  echo "         Provision them with demo/scripts/provision_assets.py (see demo/README.md)." >&2
fi

if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :$PORT" | grep -q LISTEN; then
  echo "Port $PORT is already in use. Is the demo already running? Try: $0 --port 8502" >&2
  exit 1
fi

URL="http://127.0.0.1:$PORT"
open_when_ready() {
  for _ in $(seq 1 90); do
    if curl -s -o /dev/null "$URL/_stcore/health" 2>/dev/null; then
      if command -v wslview >/dev/null 2>&1; then wslview "$URL"
      elif command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$URL"
      elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
      elif command -v open >/dev/null 2>&1; then open "$URL"
      else echo "Open $URL in your browser."
      fi
      return
    fi
    sleep 1
  done
  echo "The server did not answer within 90 s; open $URL manually." >&2
}
if [[ "$OPEN_BROWSER" == 1 ]]; then
  open_when_ready >/dev/null 2>&1 &
fi

echo "Starting the Streamlit demo at $URL (Ctrl+C to stop)"
cd "$APP"  # Streamlit reads .streamlit/config.toml from the working directory.
exec "$PY" -m streamlit run app.py --server.port "$PORT"
