"""Tests for fetching artifact bytes straight from S3.

Artifacts are read with boto3 rather than through MLflow's artifact repository,
which takes a tracking-server round trip off every artifact request. These tests
cover the URI/key handling, the error translation the routes depend on, and that
the gateway actually picks the S3 path (and falls back when it should).
"""

import pytest

from tsadar_browser.cache import ArtifactCache
from tsadar_browser.gateway import MlflowGateway
from tsadar_browser.s3 import S3ArtifactReader, object_key, parse_s3_uri

from .conftest import FakeS3Client, make_run


class TestParseS3Uri:
    def test_splits_bucket_and_prefix(self):
        assert parse_s3_uri("s3://public-ergodic-continuum/298/run-abc/artifacts") == (
            "public-ergodic-continuum",
            "298/run-abc/artifacts",
        )

    def test_bucket_root_has_an_empty_prefix(self):
        assert parse_s3_uri("s3://bucket") == ("bucket", "")

    @pytest.mark.parametrize(
        "uri",
        [
            None,
            "",
            "/mlruns/298/run-abc/artifacts",
            "file:///Users/archis/mlruns/298/run-abc/artifacts",
            "gs://bucket/prefix",
        ],
    )
    def test_non_s3_stores_return_none(self, uri):
        """None is the caller's signal to fall back to MLflow, not an error.

        A local ``file://`` mlruns is the normal development setup, so it must not
        raise on the way past.
        """
        assert parse_s3_uri(uri) is None


class TestObjectKey:
    def test_joins_prefix_and_artifact_path(self):
        assert object_key("298/run-abc/artifacts", "binary/ele_fit_and_data.nc") == (
            "298/run-abc/artifacts/binary/ele_fit_and_data.nc"
        )

    def test_empty_prefix_leaves_the_path_alone(self):
        assert object_key("", "manifest.json") == "manifest.json"


class TestReader:
    def test_downloads_to_the_destination(self, fake_client, fake_s3, tmp_path):
        fake_client.artifact_files["run-abc"] = {"plots/fit.png": b"\x89PNG"}
        reader = S3ArtifactReader(client=fake_s3)

        local = reader.download(
            "s3://public-ergodic-continuum/1/run-abc/artifacts", "plots/fit.png", tmp_path / "scratch"
        )

        assert local.read_bytes() == b"\x89PNG"
        assert local.name == "fit.png"

    def test_builds_the_key_from_the_run_prefix(self, fake_client, fake_s3, tmp_path):
        fake_client.artifact_files["run-abc"] = {"binary/ele_fit_and_data.nc": b"cdf"}
        reader = S3ArtifactReader(client=fake_s3)

        reader.download(
            "s3://public-ergodic-continuum/1/run-abc/artifacts",
            "binary/ele_fit_and_data.nc",
            tmp_path / "scratch",
        )

        assert fake_s3.downloads == [
            ("public-ergodic-continuum", "1/run-abc/artifacts/binary/ele_fit_and_data.nc")
        ]

    def test_a_non_s3_uri_is_a_programming_error(self, fake_s3, tmp_path):
        reader = S3ArtifactReader(client=fake_s3)
        with pytest.raises(ValueError, match="not an S3 artifact URI"):
            reader.download("file:///mlruns/1/run-abc/artifacts", "plots/fit.png", tmp_path)

    def test_missing_object_is_a_file_not_found(self, fake_client, fake_s3, tmp_path):
        """So the artifact route answers 404 rather than 502."""
        reader = S3ArtifactReader(client=fake_s3)
        with pytest.raises(FileNotFoundError):
            reader.download("s3://bucket/1/run-abc/artifacts", "plots/absent.png", tmp_path)

    @pytest.mark.parametrize("code", ["404", "NoSuchKey", "NoSuchBucket", "NotFound"])
    def test_all_missing_codes_map_to_file_not_found(self, fake_s3, tmp_path, code):
        fake_s3.error_code = code
        reader = S3ArtifactReader(client=fake_s3)
        with pytest.raises(FileNotFoundError):
            reader.download("s3://bucket/1/run-abc/artifacts", "plots/fit.png", tmp_path)

    @pytest.mark.parametrize("code", ["AccessDenied", "SlowDown", "InternalError"])
    def test_other_codes_are_an_os_error(self, fake_s3, tmp_path, code):
        """AccessDenied is an upstream problem, so it must not read as 'no such file'.

        Collapsing it into a 404 would present a credentials misconfiguration as a
        run with no artifacts, which is the hardest possible thing to debug.
        """
        fake_s3.error_code = code
        reader = S3ArtifactReader(client=fake_s3)
        with pytest.raises(OSError) as raised:
            reader.download("s3://bucket/1/run-abc/artifacts", "plots/fit.png", tmp_path)
        assert not isinstance(raised.value, FileNotFoundError)
        assert code in str(raised.value)

    def test_an_untyped_failure_is_an_os_error(self, fake_s3, tmp_path):
        class Boom(FakeS3Client):
            def download_file(self, bucket, key, target):
                raise RuntimeError("connection reset")

        reader = S3ArtifactReader(client=Boom(None))
        with pytest.raises(OSError, match="connection reset"):
            reader.download("s3://bucket/1/run-abc/artifacts", "plots/fit.png", tmp_path)

    def test_the_boto3_client_is_built_lazily(self):
        """Constructing the reader must not need credentials.

        /api/health has to answer on a task with no AWS access at all, and the app
        builds a gateway (and therefore a reader) before any request arrives.
        """
        reader = S3ArtifactReader()
        assert reader._client is None


class TestGatewayTransport:
    def test_s3_is_used_and_mlflow_is_not(self, gateway, fake_client, fake_s3):
        fake_client.artifact_files["run-abc"] = {"plots/fit.png": b"\x89PNG"}

        local = gateway.download_artifact("run-abc", "plots/fit.png")

        assert local.read_bytes() == b"\x89PNG"
        assert fake_s3.downloads == [("public-ergodic-continuum", "1/run-abc/artifacts/plots/fit.png")]
        assert fake_client.download_calls == []

    def test_a_file_store_falls_back_to_mlflow(self, settings, cache, fake_client, fake_s3):
        """A local mlruns directory has no S3 to talk to."""
        run = make_run(run_id="run-local")
        run.info.artifact_uri = "file:///Users/archis/mlruns/1/run-local/artifacts"
        fake_client.runs["run-local"] = run
        fake_client.artifact_files["run-local"] = {"plots/fit.png": b"local"}

        gateway = MlflowGateway(
            settings=settings, cache=cache, client=fake_client, s3_reader=S3ArtifactReader(client=fake_s3)
        )
        local = gateway.download_artifact("run-local", "plots/fit.png")

        assert local.read_bytes() == b"local"
        assert fake_client.download_calls == [("run-local", "plots/fit.png")]
        assert fake_s3.downloads == []

    def test_the_setting_can_force_everything_through_mlflow(self, tmp_path, fake_client, fake_s3):
        from tsadar_browser.settings import Settings

        settings = Settings(
            cache_dir=tmp_path / "cache",
            cache_max_gb=0.001,
            cors_origins=[],
            thomson_experiments=["inverse-thomson-scattering"],
            artifact_s3_direct=False,
        )
        gateway = MlflowGateway(
            settings=settings,
            cache=ArtifactCache(root=settings.cache_dir, max_bytes=settings.cache_max_bytes),
            client=fake_client,
            s3_reader=S3ArtifactReader(client=fake_s3),
        )
        fake_client.artifact_files["run-abc"] = {"plots/fit.png": b"\x89PNG"}

        gateway.download_artifact("run-abc", "plots/fit.png")

        assert fake_s3.downloads == []
        assert fake_client.download_calls == [("run-abc", "plots/fit.png")]

    def test_a_missing_object_is_a_404_through_the_route(self, client, fake_client):
        response = client.get("/api/runs/run-abc/artifacts/plots/absent.png")
        assert response.status_code == 404

    def test_access_denied_is_a_502_through_the_route(self, client, fake_client, fake_s3):
        """A credentials problem must not masquerade as a missing artifact."""
        fake_client.artifact_files["run-abc"] = {"plots/fit.png": b"\x89PNG"}
        fake_s3.error_code = "AccessDenied"

        response = client.get("/api/runs/run-abc/artifacts/plots/fit.png")

        assert response.status_code == 502

    def test_a_cached_artifact_is_not_fetched_twice(self, gateway, fake_client, fake_s3):
        """The disk cache still short-circuits S3, same as it did MLflow."""
        fake_client.artifact_files["run-abc"] = {"plots/fit.png": b"\x89PNG"}

        gateway.download_artifact("run-abc", "plots/fit.png")
        gateway.download_artifact("run-abc", "plots/fit.png")

        assert len(fake_s3.downloads) == 1

    def test_a_running_run_is_refetched(self, settings, cache, fake_client, fake_s3):
        """A live run's artifacts may still be rewritten, so they are not cached."""
        run = make_run(run_id="run-live", status="RUNNING")
        fake_client.runs["run-live"] = run
        fake_client.artifact_files["run-live"] = {"plots/fit.png": b"\x89PNG"}

        gateway = MlflowGateway(
            settings=settings, cache=cache, client=fake_client, s3_reader=S3ArtifactReader(client=fake_s3)
        )
        gateway.download_artifact("run-live", "plots/fit.png")
        gateway.download_artifact("run-live", "plots/fit.png")

        assert len(fake_s3.downloads) == 2
