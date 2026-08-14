"""Configuration for the Thomson analysis browser backend.

Everything is env-driven so the container needs no config file. The MLflow
credential variables keep their canonical names (``MLFLOW_TRACKING_URI`` and
friends) because the mlflow client reads them straight out of ``os.environ``;
:meth:`Settings.apply_to_environment` pushes values loaded from a ``.env`` file
back out so the client sees them too.
"""

import os
import tempfile
from functools import lru_cache
from pathlib import Path

from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

DEFAULT_TRACKING_URI = "https://continuum.ergodic.io/experiments"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    mlflow_tracking_uri: str = DEFAULT_TRACKING_URI
    mlflow_tracking_username: str | None = None
    mlflow_tracking_password: str | None = None

    cache_dir: Path = Field(default_factory=lambda: Path(tempfile.gettempdir()) / "tsadar-browser-cache")
    cache_max_gb: float = 10.0

    # Where the built SPA lives. Unset in development, where Vite serves it and
    # proxies /api here; set to /app/static in the deployed image.
    static_dir: Path | None = None

    # The Vite dev server runs on a different origin than uvicorn; in the
    # deployed image the SPA is served same-origin so this stays empty.
    # NoDecode opts out of pydantic-settings' JSON parsing so the validator
    # below can accept the comma-separated form a task definition would use.
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept ``a,b`` as well as a real list, and treat empty as 'no CORS'."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    # search_runs page size ceiling. MLflow itself caps at 50000, but the run
    # browser table has no use for pages that large.
    max_page_size: int = 200

    # MLflow's own defaults are a 120s timeout with 7 retries and a backoff
    # factor of 2, so one unreachable-server call can block for minutes. That is
    # unusable behind a load balancer health check and unpleasant in a UI, so the
    # browser bounds it hard: a failure should surface as 'degraded' quickly.
    mlflow_http_timeout: int = 15
    mlflow_http_max_retries: int = 2
    mlflow_http_backoff_factor: int = 1

    # How long a successful/failed reachability probe is trusted, so a burst of
    # health checks does not turn into a burst of MLflow requests.
    health_probe_ttl_s: float = 5.0

    @property
    def cache_max_bytes(self) -> int:
        return int(self.cache_max_gb * 1024**3)

    @property
    def mlflow_ui_base(self) -> str:
        """Base URL for deep links back into the MLflow UI."""
        return self.mlflow_tracking_uri.rstrip("/")

    def apply_to_environment(self) -> None:
        """Export MLflow settings so the mlflow client picks them up.

        The client reads all of these from ``os.environ`` directly, which is why
        they are pushed back out here rather than passed as arguments.
        """
        os.environ["MLFLOW_TRACKING_URI"] = self.mlflow_tracking_uri
        for name, value in (
            ("MLFLOW_TRACKING_USERNAME", self.mlflow_tracking_username),
            ("MLFLOW_TRACKING_PASSWORD", self.mlflow_tracking_password),
        ):
            if value:
                os.environ[name] = value

        # Only set if the operator has not overridden them explicitly.
        for name, value in (
            ("MLFLOW_HTTP_REQUEST_TIMEOUT", self.mlflow_http_timeout),
            ("MLFLOW_HTTP_REQUEST_MAX_RETRIES", self.mlflow_http_max_retries),
            ("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", self.mlflow_http_backoff_factor),
        ):
            os.environ.setdefault(name, str(value))


@lru_cache
def get_settings() -> Settings:
    return Settings()
