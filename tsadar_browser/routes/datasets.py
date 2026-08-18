"""Interactive plot endpoints, served from the ``binary/*.nc`` datasets.

Restricted to 1D (time- and space-resolved) Thomson; angular runs are refused
with a distinguishable reason so the client can fall back to the plot gallery
rather than reporting an error. See issue #37.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from mlflow.exceptions import MlflowException

from ..datasets import DEFAULT_MAX_PX, SPECTROGRAM_FIELDS, DatasetService, DatasetUnavailable
from ..deps import get_dataset_service
from ..schemas import (
    DatasetAvailability,
    DatasetUnavailableResponse,
    Lineout,
    Profiles,
    Spectrogram,
)
from ..gateway import NotThomson
from .runs import _not_thomson, _reraise

logger = logging.getLogger(__name__)

router = APIRouter(tags=["datasets"])

#: Documented on every dataset route so the generated client knows the error
#: body carries a reason code, not just a string.
UNAVAILABLE_RESPONSES = {
    400: {"model": DatasetUnavailableResponse, "description": "Invalid field, spectrum or index"},
    404: {"model": DatasetUnavailableResponse, "description": "Dataset missing or unreadable"},
    409: {"model": DatasetUnavailableResponse, "description": "Recognized but unsupported, e.g. an angular run"},
}


def _unavailable(exc: DatasetUnavailable) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={"reason": exc.reason.value, "detail": exc.message},
    )


@router.get(
    "/runs/{run_id}/datasets",
    response_model=DatasetAvailability,
    summary="What interactive views this run supports",
)
def get_availability(
    run_id: str, service: Annotated[DatasetService, Depends(get_dataset_service)]
) -> DatasetAvailability:
    """Probe before rendering.

    Always answers -- an angular or pre-contract run gets ``supported: false``
    plus a reason rather than an error, so the run detail view can pick a layout
    in one call.
    """
    try:
        return service.describe(run_id)
    except NotThomson as exc:
        raise _not_thomson(exc) from exc
    except MlflowException as exc:
        raise _reraise(exc, "run") from exc


@router.get(
    "/runs/{run_id}/spectrogram",
    response_model=Spectrogram,
    responses=UNAVAILABLE_RESPONSES,
    summary="2D spectrogram, downsampled server-side",
)
def get_spectrogram(
    run_id: str,
    service: Annotated[DatasetService, Depends(get_dataset_service)],
    which: Annotated[str, Query(description="'ele' or 'ion'")] = "ele",
    field: Annotated[
        str, Query(description=f"One of {list(SPECTROGRAM_FIELDS)}; 'residual' is derived as data - fit")
    ] = "data",
    max_px: Annotated[
        int,
        Query(
            ge=1,
            description=(
                "Pixel budget. The array is block-averaged to at most this many points; "
                "the lineout axis is spared where possible, so wavelength is reduced first."
            ),
        ),
    ] = DEFAULT_MAX_PX,
) -> Spectrogram:
    try:
        return service.spectrogram(run_id, which=which, field=field, max_px=max_px)
    except NotThomson as exc:
        raise _not_thomson(exc) from exc
    except DatasetUnavailable as exc:
        raise _unavailable(exc) from exc
    except NotThomson as exc:
        raise _not_thomson(exc) from exc
    except MlflowException as exc:
        raise _reraise(exc, "run") from exc


@router.get(
    "/runs/{run_id}/lineout",
    response_model=Lineout,
    responses=UNAVAILABLE_RESPONSES,
    summary="Measured vs fitted spectrum at one lineout",
)
def get_lineout(
    run_id: str,
    service: Annotated[DatasetService, Depends(get_dataset_service)],
    which: Annotated[str, Query(description="'ele' or 'ion'")] = "ele",
    index: Annotated[int, Query(description="Lineout index; negative counts from the end")] = 0,
) -> Lineout:
    """The interactive replacement for the pre-rendered lineout PNGs.

    IRF and noise components are not in the netCDF datasets, so they are reported
    as unavailable rather than fabricated -- see ``components_unavailable``.
    """
    try:
        return service.lineout(run_id, which=which, index=index)
    except NotThomson as exc:
        raise _not_thomson(exc) from exc
    except DatasetUnavailable as exc:
        raise _unavailable(exc) from exc
    except NotThomson as exc:
        raise _not_thomson(exc) from exc
    except MlflowException as exc:
        raise _reraise(exc, "run") from exc


@router.get(
    "/runs/{run_id}/profiles",
    response_model=Profiles,
    responses=UNAVAILABLE_RESPONSES,
    summary="Fitted parameters vs lineout, with uncertainties where available",
)
def get_profiles(
    run_id: str, service: Annotated[DatasetService, Depends(get_dataset_service)]
) -> Profiles:
    try:
        return service.profiles(run_id)
    except NotThomson as exc:
        raise _not_thomson(exc) from exc
    except DatasetUnavailable as exc:
        raise _unavailable(exc) from exc
    except NotThomson as exc:
        raise _not_thomson(exc) from exc
    except MlflowException as exc:
        raise _reraise(exc, "run") from exc
