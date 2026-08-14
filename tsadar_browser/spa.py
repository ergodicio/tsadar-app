"""Serving the built single-page app alongside ``/api``.

In development the SPA runs on Vite's own server and proxies ``/api`` here, so
none of this is used. In the deployed image both halves are served from one
origin by one process, which is why the container needs no proxy and no CORS.

Two details that a naive static mount gets wrong:

- **Client-side routes must return ``index.html``.** ``/runs/abc123`` is a route
  the router resolves in the browser, not a file on disk, so a plain static mount
  would 404 it -- meaning every shared deep link would break, which is much of
  the point of leaving Streamlit.
- **Unknown ``/api`` paths must stay JSON 404s.** If the catch-all served
  ``index.html`` for those too, a typo'd endpoint would return HTML with a 200
  and the client would fail while parsing rather than on the status code.
- **The two halves of the bundle need opposite caching.** ``assets/`` is
  content-hashed and cached for a year; ``index.html`` must never be cached or a
  new image's assets are never fetched.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

logger = logging.getLogger(__name__)

#: Vite emits content-hashed filenames here, so they can be cached hard.
ASSETS_DIRNAME = "assets"

IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
#: index.html must not be cached, or a new image's assets are never fetched.
NO_CACHE = "no-cache, no-store, must-revalidate"


class ImmutableStaticFiles(StaticFiles):
    """Static files served with a year-long immutable cache.

    Safe only because Vite content-hashes these filenames: the bytes behind
    ``/assets/index-BZKVenSd.js`` never change, and a new build produces a new
    name. A plain ``StaticFiles`` mount serves them with an ETag and no
    ``Cache-Control``, so every visit revalidates every asset -- correct, but it
    pays a round trip per file for content that provably cannot change.
    """

    def file_response(self, *args: object, **kwargs: object) -> Response:
        # Positional signature has changed across starlette versions, so pass
        # through rather than restating it.
        response = super().file_response(*args, **kwargs)  # type: ignore[arg-type]
        response.headers["Cache-Control"] = IMMUTABLE_CACHE
        return response


def _safe_child(root: Path, relative: str) -> Path | None:
    """Resolve ``relative`` under ``root``, or None if it escapes or is absent."""
    if not relative:
        return None
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        return None
    return candidate


def mount_spa(app: FastAPI, static_dir: Path) -> bool:
    """Serve the built SPA from ``static_dir``. Returns False if there is nothing to serve.

    Called after the API routers are registered so ``/api`` always wins.
    """
    index = static_dir / "index.html"
    if not index.is_file():
        logger.info("no SPA bundle at %s; serving the API only", static_dir)
        return False

    assets = static_dir / ASSETS_DIRNAME
    if assets.is_dir():
        app.mount(
            f"/{ASSETS_DIRNAME}",
            ImmutableStaticFiles(directory=assets),
            name=ASSETS_DIRNAME,
        )

    @app.get("/{spa_path:path}", include_in_schema=False)
    async def serve_spa(spa_path: str) -> FileResponse:
        # Never shadow the API: an unknown /api path is a JSON 404, not the app
        # shell with a 200.
        if spa_path == "api" or spa_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")

        # A real file (favicon, robots.txt) is served as itself...
        existing = _safe_child(static_dir, spa_path)
        if existing is not None:
            return FileResponse(existing)

        # ...anything else is a client-side route, so hand back the shell and let
        # the router decide -- including its own 404 page.
        return FileResponse(index, headers={"Cache-Control": NO_CACHE})

    logger.info("serving the SPA from %s", static_dir)
    return True
