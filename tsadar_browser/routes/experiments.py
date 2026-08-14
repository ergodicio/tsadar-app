"""Experiment listing."""

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_gateway
from ..gateway import MlflowGateway
from ..schemas import ExperimentList

router = APIRouter(tags=["experiments"])


@router.get("/experiments", response_model=ExperimentList, summary="List active experiments")
def list_experiments(gateway: MlflowGateway = Depends(get_gateway)) -> ExperimentList:
    try:
        return ExperimentList(experiments=gateway.list_experiments())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"MLflow tracking server error: {exc}") from exc
