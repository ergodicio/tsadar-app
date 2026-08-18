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
from tsadar_browser.s3 import S3ArtifactReader
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
        # Records MLflow-side artifact fetches, so a test can prove a download
        # went straight to S3 rather than falling back through the tracking server.
        self.download_calls: list[tuple[str, str]] = []
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
        self.download_calls.append((run_id, path))
        payload = self.artifact_files.get(run_id, {}).get(path)
        if payload is None:
            raise missing(f"no artifact {path}")
        target = Path(dst_path) / Path(path).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)


def file_info(path, is_dir=False, file_size=None):
    return SimpleNamespace(path=path, is_dir=is_dir, file_size=file_size)


class FakeClientError(Exception):
    """A botocore ``ClientError`` lookalike.

    ``S3ArtifactReader._translate`` matches on ``response["Error"]["Code"]``
    rather than the exception class, precisely because botocore raises the same
    class for a missing key and for access denied. So the fake only has to carry
    that shape.
    """

    def __init__(self, code: str, message: str = "boom"):
        super().__init__(f"{code}: {message}")
        self.response = {"Error": {"Code": code, "Message": message}}


class FakeS3Client:
    """Serves objects out of ``FakeMlflowClient.artifact_files``.

    Injected as the *boto3 client* rather than replacing S3ArtifactReader, so the
    real key construction and error translation run in tests -- that is the code
    production takes, since every run's artifact_uri on this tracking server is
    an ``s3://`` URI.
    """

    def __init__(self, fake_client: "FakeMlflowClient"):
        self.fake_client = fake_client
        self.downloads: list[tuple[str, str]] = []
        self.error_code: str | None = None

    def download_file(self, bucket, key, target):
        self.downloads.append((bucket, key))
        if self.error_code:
            raise FakeClientError(self.error_code)

        # Keys look like "<experiment_id>/<run_id>/artifacts/<artifact_path>",
        # which is what the fake run's artifact_uri prefix plus object_key builds.
        marker = "/artifacts/"
        if marker not in key:
            raise FakeClientError("NoSuchKey", key)
        prefix, _, artifact_path = key.partition(marker)
        run_id = prefix.split("/")[-1]

        payload = self.fake_client.artifact_files.get(run_id, {}).get(artifact_path)
        if payload is None:
            raise FakeClientError("NoSuchKey", key)

        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(payload)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        mlflow_tracking_uri="https://continuum.ergodic.io/experiments",
        cache_dir=tmp_path / "cache",
        cache_max_gb=0.001,
        cors_origins=[],
        # Pin the Thomson scope to the fake's one experiment. An explicit
        # allowlist means ThomsonRegistry never starts a background discovery,
        # which matters for more than speed: discovery issues its own
        # search_runs calls, and a thread landing mid-test would both append to
        # FakeMlflowClient.search_calls (breaking any assertion on the *last*
        # call) and make the scope depend on timing. Discovery itself is tested
        # directly in test_thomson.py against a purpose-built fake.
        thomson_experiments=["inverse-thomson-scattering"],
    )


@pytest.fixture
def cache(settings) -> ArtifactCache:
    return ArtifactCache(root=settings.cache_dir, max_bytes=settings.cache_max_bytes)


@pytest.fixture
def fake_client() -> FakeMlflowClient:
    return FakeMlflowClient()


@pytest.fixture
def fake_s3(fake_client) -> FakeS3Client:
    return FakeS3Client(fake_client)


@pytest.fixture
def gateway(settings, cache, fake_client, fake_s3) -> MlflowGateway:
    return MlflowGateway(
        settings=settings,
        cache=cache,
        client=fake_client,
        s3_reader=S3ArtifactReader(client=fake_s3),
    )


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


# -- dataset fixtures (issue #30) ---------------------------------------------


def install_artifacts(fake_client: FakeMlflowClient, run_id: str, files: dict[str, bytes]) -> None:
    """Register artifact bytes and the directory listing that exposes them.

    The gateway lists artifacts to classify a run before reading anything, so a
    test that only registers bytes would look like a run with no artifacts.
    """
    fake_client.artifact_files.setdefault(run_id, {}).update(files)

    listing: dict[str, list] = {}
    for path in files:
        parent, _, _ = path.rpartition("/")
        listing.setdefault(parent, []).append(file_info(path, file_size=len(files[path])))
        if parent and not any(entry.path == parent for entry in listing.get("", [])):
            listing.setdefault("", []).append(file_info(parent, is_dir=True))
    fake_client.artifacts[run_id] = listing


@pytest.fixture
def dataset_service(gateway):
    from tsadar_browser.datasets import DatasetService

    return DatasetService(gateway=gateway)


@pytest.fixture
def dataset_client(gateway, dataset_service) -> TestClient:
    from tsadar_browser.deps import get_dataset_service

    app = create_app()
    app.dependency_overrides[get_gateway] = lambda: gateway
    app.dependency_overrides[get_dataset_service] = lambda: dataset_service
    return TestClient(app)


@pytest.fixture
def one_d_run(fake_client, tmp_path) -> str:
    """A temporal run with ele + ion spectra, learned parameters and sigmas."""
    from .fixtures import learned_parameters_csv, sigmas_netcdf, write_spectrum

    run_id = "run-abc"
    install_artifacts(
        fake_client,
        run_id,
        {
            "binary/ele_fit_and_data.nc": write_spectrum(tmp_path / "ele", "ele_fit_and_data.nc"),
            "binary/ion_fit_and_data.nc": write_spectrum(tmp_path / "ion", "ion_fit_and_data.nc"),
            "csv/learned_parameters.csv": learned_parameters_csv(),
            "sigmas.nc": sigmas_netcdf(tmp_path / "sig"),
            "plots/fit_and_data.png": b"\x89PNG\r\n\x1a\n",
        },
    )
    return run_id


@pytest.fixture
def angular_run(fake_client, tmp_path) -> str:
    """An angular run: same variables and dimensionality, angular x axis."""
    from .fixtures import ANGULAR_AXIS, learned_parameters_csv, write_spectrum

    run_id = "run-angular"
    fake_client.runs[run_id] = make_run(run_id=run_id, params={**SAMPLE_PARAMS, "other.extraoptions.spectype": "angular_full"})
    install_artifacts(
        fake_client,
        run_id,
        {
            "binary/fit_and_data.nc": write_spectrum(
                tmp_path / "ang", "fit_and_data.nc", x_label=ANGULAR_AXIS
            ),
            "csv/learned_parameters.csv": learned_parameters_csv(include_axis=False),
            "plots/fit_and_data.png": b"\x89PNG\r\n\x1a\n",
        },
    )
    return run_id


@pytest.fixture
def bare_run(fake_client) -> str:
    """A pre-contract run: PNGs only, no datasets."""
    run_id = "run-old"
    fake_client.runs[run_id] = make_run(run_id=run_id)
    install_artifacts(fake_client, run_id, {"plots/fit_and_data.png": b"\x89PNG\r\n\x1a\n"})
    return run_id
