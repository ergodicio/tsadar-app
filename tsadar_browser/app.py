"""FastAPI application for the Thomson analysis browser.

The OpenAPI schema this app produces is the frontend contract -- the TypeScript
client is generated from it -- so route signatures and response models are
public API. See ``docs/browser.md``.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import datasets, experiments, health, runs
from .settings import get_settings
from .spa import mount_spa

logger = logging.getLogger(__name__)

DESCRIPTION = """
Read layer over the MLflow tracking server for browsing Thomson scattering
analysis runs. MLflow remains the source of truth; this service holds no
database of its own and never writes to it.
"""


def create_app() -> FastAPI:
    settings = get_settings()
    settings.apply_to_environment()

    app = FastAPI(
        title="TSADAR analysis browser API",
        description=DESCRIPTION,
        version="0.1.0",
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url=None,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET"],
            allow_headers=["*"],
        )

    app.include_router(health.router, prefix="/api")
    app.include_router(experiments.router, prefix="/api")
    app.include_router(runs.router, prefix="/api")
    app.include_router(datasets.router, prefix="/api")

    # After the routers, so /api always wins over the SPA catch-all.
    if settings.static_dir is not None:
        mount_spa(app, settings.static_dir)

    logger.info("browser API configured against %s", settings.mlflow_tracking_uri)
    if not settings.mlflow_tracking_username:
        # Not fatal -- /api/health will report `degraded` and say why -- but this
        # is the single most likely misconfiguration, so it should be visible in
        # the task logs rather than only in a health response.
        logger.warning("MLFLOW_TRACKING_USERNAME is unset; MLflow requests will likely be rejected")

    return app


app = create_app()
