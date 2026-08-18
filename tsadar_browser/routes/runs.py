"""Run listing, run detail, metric history, and artifact passthrough."""

import logging
import mimetypes
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from mlflow.exceptions import MlflowException

from ..cache import UnsafeArtifactPath
from ..deps import get_gateway
from ..gateway import SHOT_PARAM, InvalidQuery, MlflowGateway, NotThomson
from ..schemas import MetricHistory, RunDetail, RunPage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["runs"])

#: Content types the stdlib does not know but the browser needs to serve.
EXTRA_CONTENT_TYPES = {
    ".nc": "application/x-netcdf",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
}


def _not_thomson(exc: NotThomson) -> HTTPException:
    """Translate an out-of-scope experiment or run into a 404 with a reason code.

    404 rather than 403: the resource exists on the tracking server but not in
    *this* browser, which covers Thomson analysis only. That is the same
    distinction the dataset endpoints draw, and it keeps one code path in the
    client -- the body nests under ``detail`` exactly like theirs, so the reason
    reads as ``err.detail.reason``.

    Note the neighbouring 400 for an experiment that does not exist at all: a
    name nothing matches is a bad query value, whereas a real experiment holding
    another project's runs is a resource this browser does not serve.
    """
    return HTTPException(status_code=404, detail={"reason": "not_thomson", "detail": str(exc)})


def _reraise(exc: MlflowException, what: str) -> HTTPException:
    """Translate an MLflow error into the closest HTTP status.

    MLflow already maps its error codes to HTTP statuses, so defer to that
    rather than matching on error-code strings. Anything that is not clearly the
    caller's fault becomes a 502: the failure is upstream, not here.
    """
    status = exc.get_http_status_code()
    if status == 404:
        return HTTPException(status_code=404, detail=f"{what} not found: {exc}")
    if status in (400, 403):
        return HTTPException(status_code=status, detail=str(exc))
    return HTTPException(status_code=502, detail=f"MLflow tracking server error: {exc}")


@router.get("/runs", response_model=RunPage, summary="Search runs")
def list_runs(
    gateway: Annotated[MlflowGateway, Depends(get_gateway)],
    experiment: Annotated[str | None, Query(description="Experiment name (or id)")] = None,
    shot: Annotated[str | None, Query(description=f'Shot number, matched against params."{SHOT_PARAM}"')] = None,
    status: Annotated[
        str | None, Query(description="MLflow lifecycle status: RUNNING, FINISHED, FAILED, KILLED")
    ] = None,
    stage: Annotated[str | None, Query(description="tsadar progress tag, e.g. 'minimizing', 'completed'")] = None,
    user: Annotated[str | None, Query(description="Submitting user (mlflow.user tag)")] = None,
    q: Annotated[
        str | None,
        Query(
            description=(
                "Free-text substring match on run name. '%' and '_' act as SQL LIKE "
                "wildcards (any-sequence and any-character) rather than literals; "
                "MLflow's filter grammar has no ESCAPE clause to make them literal."
            )
        ),
    ] = None,
    sort: Annotated[
        str | None,
        Query(
            description=(
                "Sort key; prefix with '-' for descending. One of created, name, status, shot, loss. "
                "'loss' orders on the 'overall loss' metric specifically -- unlike final_loss it "
                "cannot fall back to 'min loss'/'epoch loss', so runs lacking 'overall loss' sort "
                "as null; check loss_key when displaying the value. Duration is not sortable: it is "
                "computed from timestamps and MLflow cannot order by it."
            )
        ),
    ] = None,
    page_size: Annotated[int, Query(ge=1, description="Rows per page")] = 50,
    page_token: Annotated[
        str | None, Query(description="Cursor from a previous response's next_page_token")
    ] = None,
) -> RunPage:
    try:
        return gateway.search_runs(
            experiment=experiment,
            shot=shot,
            status=status,
            stage=stage,
            user=user,
            q=q,
            sort=sort,
            page_size=min(page_size, gateway.settings.max_page_size),
            page_token=page_token,
        )
    except NotThomson as exc:
        raise _not_thomson(exc) from exc
    except InvalidQuery as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MlflowException as exc:
        raise _reraise(exc, "experiment") from exc


@router.get("/runs/{run_id}", response_model=RunDetail, summary="Run detail with config tree and artifact listing")
def get_run(run_id: str, gateway: Annotated[MlflowGateway, Depends(get_gateway)]) -> RunDetail:
    try:
        return gateway.get_run(run_id)
    except NotThomson as exc:
        raise _not_thomson(exc) from exc
    except MlflowException as exc:
        raise _reraise(exc, "run") from exc


@router.get(
    "/runs/{run_id}/metrics/{key:path}",
    response_model=MetricHistory,
    summary="Full metric history (loss curves)",
)
def get_metric_history(
    run_id: str, key: str, gateway: Annotated[MlflowGateway, Depends(get_gateway)]
) -> MetricHistory:
    """Metric keys contain spaces (``overall loss``), so they arrive URL-encoded."""
    try:
        history = gateway.get_metric_history(run_id, key)
    except MlflowException as exc:
        raise _reraise(exc, "run") from exc

    if not history.points:
        raise HTTPException(status_code=404, detail=f"no metric history for key {key!r} on run {run_id}")
    return history


@router.get(
    "/runs/{run_id}/artifacts/{artifact_path:path}",
    summary="Stream an artifact",
    response_class=FileResponse,
    responses={200: {"content": {"application/octet-stream": {}}, "description": "Artifact bytes"}},
)
def get_artifact(
    run_id: str, artifact_path: str, gateway: Annotated[MlflowGateway, Depends(get_gateway)]
) -> FileResponse:
    """Serve artifact bytes through the API so the frontend needs no S3 credentials."""
    try:
        local = gateway.download_artifact(run_id, artifact_path)
    except NotThomson as exc:
        raise _not_thomson(exc) from exc
    except UnsafeArtifactPath as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MlflowException as exc:
        raise _reraise(exc, "artifact") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"artifact not found: {artifact_path}") from exc
    except OSError as exc:
        logger.warning("could not fetch artifact %s/%s: %s", run_id, artifact_path, exc)
        raise HTTPException(status_code=502, detail=f"could not fetch artifact: {exc}") from exc

    suffix = local.suffix.lower()
    media_type = EXTRA_CONTENT_TYPES.get(suffix) or mimetypes.guess_type(local.name)[0] or "application/octet-stream"
    return FileResponse(local, media_type=media_type, filename=local.name)
