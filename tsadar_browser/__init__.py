"""Backend for the Thomson scattering analysis browser.

A read-only FastAPI layer over the MLflow tracking server. Run it with::

    uvicorn tsadar_browser.app:app --reload
"""

__all__ = ["create_app"]


def create_app():  # pragma: no cover - thin re-export to avoid importing FastAPI at package import
    from .app import create_app as _create_app

    return _create_app()
