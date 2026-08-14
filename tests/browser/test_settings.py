"""Settings parsing and the environment handoff to the mlflow client."""

import os

from tsadar_browser.settings import Settings


class TestCorsOrigins:
    def test_accepts_comma_separated(self, monkeypatch):
        """A task definition writes CORS_ORIGINS=a,b -- not a JSON array."""
        monkeypatch.setenv("CORS_ORIGINS", "http://a.example, http://b.example")
        assert Settings().cors_origins == ["http://a.example", "http://b.example"]

    def test_empty_means_no_cors(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "")
        assert Settings().cors_origins == []

    def test_accepts_a_real_list(self):
        assert Settings(cors_origins=["http://a"]).cors_origins == ["http://a"]


class TestCacheSizing:
    def test_gb_converted_to_bytes(self):
        assert Settings(cache_max_gb=2).cache_max_bytes == 2 * 1024**3


class TestEnvironmentHandoff:
    def test_exports_credentials_for_the_mlflow_client(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
        settings = Settings(
            mlflow_tracking_uri="https://example.invalid/experiments",
            mlflow_tracking_username="user",
            mlflow_tracking_password="secret",
        )
        settings.apply_to_environment()
        assert os.environ["MLFLOW_TRACKING_URI"] == "https://example.invalid/experiments"
        assert os.environ["MLFLOW_TRACKING_USERNAME"] == "user"
        assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "secret"

    def test_bounds_mlflow_http_retries(self, monkeypatch):
        """MLflow defaults to 120s x 7 retries, which would hang the health check."""
        for name in (
            "MLFLOW_HTTP_REQUEST_TIMEOUT",
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES",
            "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR",
        ):
            monkeypatch.delenv(name, raising=False)

        Settings().apply_to_environment()
        assert int(os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"]) == 15
        assert int(os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]) == 2

    def test_does_not_override_an_explicit_operator_setting(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "99")
        Settings().apply_to_environment()
        assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "99"

    def test_missing_credentials_are_not_exported_as_empty(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
        Settings(mlflow_tracking_username=None).apply_to_environment()
        assert "MLFLOW_TRACKING_USERNAME" not in os.environ

    def test_ui_base_strips_trailing_slash(self):
        settings = Settings(mlflow_tracking_uri="https://example.invalid/experiments/")
        assert settings.mlflow_ui_base == "https://example.invalid/experiments"
