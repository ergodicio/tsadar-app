"""Shared singletons for the API routes."""

from functools import lru_cache

from .cache import ArtifactCache
from .datasets import DatasetService
from .gateway import MlflowGateway
from .settings import Settings, get_settings


@lru_cache
def get_cache() -> ArtifactCache:
    settings = get_settings()
    return ArtifactCache(root=settings.cache_dir, max_bytes=settings.cache_max_bytes)


@lru_cache
def get_gateway() -> MlflowGateway:
    settings: Settings = get_settings()
    settings.apply_to_environment()
    return MlflowGateway(settings=settings, cache=get_cache())


@lru_cache
def get_dataset_service() -> DatasetService:
    return DatasetService(gateway=get_gateway())
