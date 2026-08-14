"""Fixtures for the browser backend tests.

The tests run against a fake MLflow client rather than the live tracking server,
so CI needs no credentials and no network. The fake mimics only the handful of
``MlflowClient`` methods :mod:`tsadar_browser.gateway` actually calls.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from tsadar_browser.app import create_app
from tsadar_browser.cache import ArtifactCache
from tsadar_browser.deps import get_gateway
from tsadar_browser.gateway import MlflowGateway
from tsadar_browser.settings import Settings


def missing(message: str) -> MlflowException:
    """Build a genuine 'does not exist' MlflowException.

    MlflowException ignores a *string* error_code (it silently becomes
    INTERNAL_ERROR), so the protobuf enum is the only way to produce an
    exception that maps to a 404 the way the real server's does.
    """
    return MlflowException(message, error_code=RESOURCE_DOES_NOT_EXIST)


# A realistic slice of what tsadar logs: flattened with a dot reducer, every
# value stringified, and a shot number under data.shotnum.
SAMPLE_PARAMS = {
    "data.shotnum": "101675",
    "data.lineouts.start": "800",
    "data.lineouts.end": "940",
    "data.lineouts.type.ps": "ps",
    "mlflow.experiment": "inverse-thomson-scattering",
    "other.extraoptions.load_ion_spec": "False",
    "other.extraoptions.spectype": "temporal",
    "other.refit_thresh": "5.0",
    "other.lamrangE": "[400, 700]",
    "parameters.electron.Te.val": "0.5",
    "parameters.electron.Te.active": "True",
    "parameters.general.Va.angle": "None",
}


def make_run(
    run_id="run-abc",
    experiment_id="1",
    run_name="test-run",
    status="FINISHED",
    params=None,
    metrics=None,
    tags=None,
    start_time=1_700_000_000_000,
    end_time=1_700_000_123_000,
):
    tags = {"mlflow.user": "archis", "mlflow.runName": run_name, "status": "completed", **(tags or {})}
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id,
            experiment_id=experiment_id,
            run_name=run_name,
            status=status,
            start_time=start_time,
            end_time=end_time,
            artifact_uri=f"s3://public-ergodic-continuum/{experiment_id}/{run_id}/artifacts",
        ),
        data=SimpleNamespace(
            params=dict(SAMPLE_PARAMS if params is None else params),
            metrics=dict({"overall loss": 12.5, "fit_time": 42.0} if metrics is None else metrics),
            tags=tags,
        ),
    )


class FakePagedList(list):
    def __init__(self, items, token=None):
        super().__init__(items)
        self.token = token


class FakeMlflowClient:
    """Minimal stand-in for ``MlflowClient``."""

    def __init__(self, runs=None, experiments=None, artifacts=None, artifact_files=None, fail=False):
        self.runs = {run.info.run_id: run for run in (runs or [make_run()])}
        self.experiments = experiments or [
            SimpleNamespace(
                experiment_id="1",
                name="inverse-thomson-scattering",
                artifact_location="s3://public-ergodic-continuum/1",
                lifecycle_stage="active",
                creation_time=1_700_000_000_000,
                last_update_time=1_700_000_000_000,
                tags={},
            )
        ]
        # {run_id: {dir_path: [FileInfo-alikes]}}
        self.artifacts = artifacts or {}
        # {run_id: {artifact_path: bytes}}
        self.artifact_files = artifact_files or {}
        self.fail = fail
        self.search_calls: list[dict] = []
        self.metric_history: dict[tuple[str, str], list] = {}

    def _guard(self):
        if self.fail:
            raise MlflowException("connection refused")

    def search_experiments(self, view_type=None, max_results=None):
        self._guard()
        return self.experiments

    def get_experiment(self, experiment_id):
        self._guard()
        for exp in self.experiments:
            if exp.experiment_id == experiment_id:
                return exp
        raise missing(f"no experiment {experiment_id}")

    def get_experiment_by_name(self, name):
        self._guard()
        return next((exp for exp in self.experiments if exp.name == name), None)

    def search_runs(self, experiment_ids, filter_string="", run_view_type=None, max_results=50, order_by=None, page_token=None):
        self._guard()
        self.search_calls.append(
            {
                "experiment_ids": experiment_ids,
                "filter_string": filter_string,
                "max_results": max_results,
                "order_by": order_by,
                "page_token": page_token,
            }
        )
        return FakePagedList(list(self.runs.values())[:max_results], token="next-cursor")

    def get_run(self, run_id):
        self._guard()
        if run_id not in self.runs:
            raise missing(f"no run {run_id}")
        return self.runs[run_id]

    def get_metric_history(self, run_id, key):
        self._guard()
        self.get_run(run_id)
        return self.metric_history.get((run_id, key), [])

    def list_artifacts(self, run_id, path=""):
        self._guard()
        return self.artifacts.get(run_id, {}).get(path, [])

    def download_artifacts(self, run_id, path, dst_path):
        self._guard()
        payload = self.artifact_files.get(run_id, {}).get(path)
        if payload is None:
            raise missing(f"no artifact {path}")
        target = Path(dst_path) / Path(path).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)


def file_info(path, is_dir=False, file_size=None):
    return SimpleNamespace(path=path, is_dir=is_dir, file_size=file_size)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        mlflow_tracking_uri="https://continuum.ergodic.io/experiments",
        cache_dir=tmp_path / "cache",
        cache_max_gb=0.001,
        cors_origins=[],
    )


@pytest.fixture
def cache(settings) -> ArtifactCache:
    return ArtifactCache(root=settings.cache_dir, max_bytes=settings.cache_max_bytes)


@pytest.fixture
def fake_client() -> FakeMlflowClient:
    return FakeMlflowClient()


@pytest.fixture
def gateway(settings, cache, fake_client) -> MlflowGateway:
    return MlflowGateway(settings=settings, cache=cache, client=fake_client)


@pytest.fixture
def client(gateway) -> TestClient:
    """A TestClient whose routes all resolve through the fake-backed gateway.

    Overriding the gateway alone is enough: routes read settings and the
    artifact cache off it rather than reaching for module-level singletons.
    """
    app = create_app()
    app.dependency_overrides[get_gateway] = lambda: gateway
    return TestClient(app)


@pytest.fixture
def manifest_bytes() -> bytes:
    return json.dumps({"schema_version": 1, "datasets": ["binary/ele_fit_and_data.nc"]}).encode()
