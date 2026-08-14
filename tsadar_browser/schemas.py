"""Response models for ``/api``.

These models *are* the frontend contract: the generated TypeScript client comes
from the OpenAPI schema FastAPI derives from them, so field names and
optionality here are load-bearing. Prefer adding an optional field over
changing the meaning of an existing one.
"""

from enum import Enum
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
    spectype: str | None = Field(
        default=None,
        description=(
            "Spectrum type as logged ('temporal', 'imaging', 'angular', 'angular_full'). "
            "A HINT, NOT GROUND TRUTH: log_mlflow runs before fitter.fit, and loadData "
            "overwrites spectype from the data file during prepare, so a deck saying "
            "'temporal' run against angular data logs 'temporal'. Confirm from artifact "
            "shape (binary/fit_and_data.nc is angular; ele_/ion_fit_and_data.nc is 1D) "
            "before rendering anything axis-dependent."
        ),
    )
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


# -- analysis datasets (issue #30) --------------------------------------------
#
# Values are plain JSON numbers. Everything is downsampled to a pixel budget
# server-side, so the payload stays small enough that a binary format would buy
# little and cost the codegen contract. Non-finite values (gaps in a fit) are
# serialized as null, since JSON has no NaN and a bare NaN makes browser
# JSON.parse throw.


class DatasetKind(str, Enum):
    """What kind of Thomson run the artifacts say this is."""

    one_d = "one_d"
    angular = "angular"
    unknown = "unknown"


class UnavailableReason(str, Enum):
    """Machine-readable reason a dataset view cannot be served.

    Distinguishing these matters: showing "no data found" for an angular run
    reads as a bug, when the truth is that the view is deliberately out of scope.
    """

    angular_not_supported = "angular_not_supported"
    dataset_missing = "dataset_missing"
    dataset_unreadable = "dataset_unreadable"
    unexpected_schema = "unexpected_schema"
    field_unavailable = "field_unavailable"
    index_out_of_range = "index_out_of_range"


class DatasetUnavailableBody(BaseModel):
    """The reason payload itself."""

    reason: UnavailableReason
    detail: str


class DatasetUnavailableResponse(BaseModel):
    """Error body for the dataset endpoints, carrying a reason code.

    Nested under ``detail`` because that is what actually goes over the wire:
    FastAPI wraps whatever an ``HTTPException`` carries in a ``detail`` key. A
    flat declaration would make the generated client read ``err.reason`` and get
    ``undefined`` on exactly the angular-vs-missing distinction these endpoints
    exist to make legible.
    """

    detail: DatasetUnavailableBody


class SpectrumInfo(BaseModel):
    which: str = Field(description="'ele' or 'ion'")
    path: str = Field(description="Artifact path this spectrum was read from")
    x_label: str = Field(description="Lineout axis name, e.g. 'Time (ps)' or 'Radius (\\mum)'")
    y_label: str = Field(description="Spectral axis name, normally 'Wavelength'")
    lineout_count: int
    wavelength_count: int
    fields: list[str] = Field(description="Renderable fields; 'residual' is derived as data - fit")


class DatasetAvailability(BaseModel):
    """What the interactive views can render for a run, and why not when they can't.

    The run detail view calls this first to choose a layout, so it answers for
    every run -- including angular and pre-contract ones -- instead of erroring.
    """

    kind: DatasetKind
    supported: bool = Field(description="True when at least one spectrum can be served")
    reason: UnavailableReason | None = None
    message: str | None = Field(default=None, description="Human-readable explanation, safe to show")
    spectra: list[SpectrumInfo] = []
    profiles_available: bool = False
    sigmas_available: bool = Field(
        default=False, description="Whether uncertainties exist; calc_sigmas is off by default"
    )
    unavailable_fields: dict[str, str] = Field(
        default_factory=dict,
        description="Fields a client might expect but which cannot be served, mapped to why",
    )


class Spectrogram(BaseModel):
    which: str
    field: str = Field(description="'data', 'fit', or the derived 'residual'")
    x_label: str
    y_label: str
    x: list[float | None] = Field(description="Lineout axis coordinates")
    y: list[float | None] = Field(description="Wavelength coordinates")
    values: list[list[float | None]] = Field(
        description="Shape (len(y), len(x)) -- row-major by wavelength, directly usable as a Plotly heatmap z"
    )
    full_shape: list[int] = Field(description="[lineouts, wavelengths] before downsampling")
    returned_shape: list[int] = Field(description="[lineouts, wavelengths] as returned")
    downsample_factors: dict[str, int] = Field(
        description="Block-averaging factors applied per axis. The lineout axis is spared where possible."
    )
    downsample_method: str = Field(description="'mean' when downsampled, 'none' when full resolution")


class Lineout(BaseModel):
    which: str
    index: int = Field(description="Resolved lineout index; negative inputs are normalized")
    lineout_count: int
    x_label: str
    x_value: float = Field(description="Position of this lineout on the lineout axis")
    y_label: str
    wavelength: list[float | None]
    data: list[float | None] = Field(description="Measured spectrum")
    fit: list[float | None] = Field(description="Fitted spectrum")
    residual: list[float | None] = Field(description="Derived as data - fit")
    components: dict[str, list[float | None]] = Field(
        default_factory=dict, description="Always empty today; see components_unavailable"
    )
    components_unavailable: str | None = Field(
        default=None, description="Why IRF/noise components are absent, when they are"
    )


class ProfileSeries(BaseModel):
    name: str = Field(description="Column name from learned_parameters.csv, e.g. 'Te_electron'")
    values: list[float | None]
    sigma: list[float | None] | None = Field(
        default=None, description="Per-point uncertainty when the run computed sigmas"
    )


class Profiles(BaseModel):
    x_label: str
    x: list[float | None] = Field(description="Lineout axis coordinates")
    lineout_pixels: list[int] | None = Field(
        default=None, description="Detector pixel index per lineout, when logged"
    )
    series: list[ProfileSeries]
    sigmas_available: bool
