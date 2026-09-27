#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -f "$ROOT/demo/frontend/dist/index.html" ]]; then
  echo 'Frontend not built. Run: cd demo/frontend && npm ci && npm run build' >&2
  exit 1
fi
PY="${DEMO_PYTHON:-$ROOT/demo/.venv-demo/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "Python environment not found at $PY. See demo/README.md (or set DEMO_PYTHON)." >&2
  exit 1
fi
cd "$ROOT"
echo "Opening offline demo at http://127.0.0.1:${DEMO_PORT:-8000}"
exec "$PY" -m uvicorn demo.scripts.serve:app --host 127.0.0.1 --port "${DEMO_PORT:-8000}" --app-dir "$ROOT"
