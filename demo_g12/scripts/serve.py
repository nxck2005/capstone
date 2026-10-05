"""Single localhost process: G-12 adapter first, built offline UI second."""

from pathlib import Path

from fastapi.staticfiles import StaticFiles

from demo_g12.backend.app import app

DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if not DIST.is_dir():
    raise RuntimeError("Build the offline UI first: cd demo_g12/frontend && npm ci && npm run build")

# Mount last so /api/* always reaches the backend. No CDN, remote calls or proxy.
app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
