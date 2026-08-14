"""Health endpoint.

Doubles as the ALB target-group check, so it must answer 200 even when MLflow
is unreachable -- a degraded browser that can explain itself is more useful than
a task the load balancer keeps killing. ``status`` carries the real verdict.
"""

from fastapi import APIRouter, Depends

from ..deps import get_gateway
from ..gateway import MlflowGateway, MlflowUnavailable
from ..schemas import CacheStats, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and MLflow reachability")
def health(gateway: MlflowGateway = Depends(get_gateway)) -> HealthResponse:
    # Read both off the gateway so a single override in tests (and a single
    # construction path in production) governs the whole request.
    settings, cache = gateway.settings, gateway.cache

    reachable, error = True, None
    try:
        gateway.ping()
    except MlflowUnavailable as exc:
        reachable, error = False, str(exc)

    entries, total_bytes = cache.stats()
    return HealthResponse(
        status="ok" if reachable else "degraded",
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
        mlflow_reachable=reachable,
        mlflow_error=error,
        cache=CacheStats(
            directory=str(cache.root),
            entries=entries,
            bytes=total_bytes,
            max_bytes=cache.max_bytes,
        ),
    )
