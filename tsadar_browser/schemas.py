"""Response models for ``/api``.

These models *are* the frontend contract: the generated TypeScript client comes
from the OpenAPI schema FastAPI derives from them, so field names and
optionality here are load-bearing. Prefer adding an optional field over
changing the meaning of an existing one.
"""

from typing import Any

from pydantic import BaseModel, Field


class CacheStats(BaseModel):
    directory: str
    entries: int
    bytes: int
    max_bytes: int


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when MLflow is reachable, 'degraded' otherwise")
    mlflow_tracking_uri: str
    mlflow_reachable: bool
    mlflow_error: str | None = Field(default=None, description="Why the tracking server could not be reached")
    cache: CacheStats


class Experiment(BaseModel):
    experiment_id: str
    name: str
    artifact_location: str | None = None
    lifecycle_stage: str | None = None
    creation_time: int | None = Field(default=None, description="Unix epoch milliseconds")
    last_update_time: int | None = Field(default=None, description="Unix epoch milliseconds")
    tags: dict[str, str] = {}


class ExperimentList(BaseModel):
    experiments: list[Experiment]


class RunSummary(BaseModel):
    """One row of the run browser table."""

    run_id: str
    run_name: str | None = None
    experiment_id: str
    experiment_name: str | None = None
    status: str | None = Field(default=None, description="MLflow lifecycle status: RUNNING/FINISHED/FAILED/KILLED")
    stage: str | None = Field(
        default=None,
        description="tsadar's own progress tag ('preprocessing', 'minimizing', ... 'completed'), when logged",
    )
    shot: str | None = Field(default=None, description="Shot number, from the data.shotnum param")
    final_loss: float | None = None
    loss_key: str | None = Field(default=None, description="Which metric final_loss was read from")
    start_time: int | None = Field(default=None, description="Unix epoch milliseconds")
    end_time: int | None = Field(default=None, description="Unix epoch milliseconds")
    duration_s: float | None = Field(default=None, description="Computed from start/end time; null while running")
    user: str | None = None


class RunPage(BaseModel):
    runs: list[RunSummary]
    page_size: int
    next_page_token: str | None = Field(
        default=None,
        description="Opaque cursor for the next page. MLflow paginates by token, not offset; null means last page.",
    )


class MetricSummary(BaseModel):
    key: str
    value: float
    step: int | None = None
    timestamp: int | None = None


class MetricPoint(BaseModel):
    step: int
    value: float
    timestamp: int | None = None


class MetricHistory(BaseModel):
    key: str
    points: list[MetricPoint]


class ArtifactEntry(BaseModel):
    path: str
    is_dir: bool
    size: int | None = None


class RunDetail(RunSummary):
    artifact_uri: str | None = None
    mlflow_run_url: str | None = Field(default=None, description="Deep link to this run in the MLflow UI")

    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Logged params unflattened back into the nested config tree",
    )
    config_flat: dict[str, str] = Field(
        default_factory=dict, description="The raw flattened params exactly as MLflow stores them"
    )
    config_unflatten_error: str | None = Field(
        default=None,
        description="Set when params could not be unflattened (colliding dotted keys); config falls back to empty",
    )

    tags: dict[str, str] = {}
    metrics: list[MetricSummary] = []
    artifacts: list[ArtifactEntry] = []
    manifest: dict[str, Any] | None = Field(
        default=None, description="Parsed manifest.json when the run logged one (ergodicio/tsadar#116)"
    )


class ErrorResponse(BaseModel):
    detail: str
