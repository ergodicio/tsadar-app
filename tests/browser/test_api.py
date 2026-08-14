"""Endpoint tests -- the contract the generated TypeScript client depends on."""

from types import SimpleNamespace

from .conftest import file_info, make_run


class TestHealth:
    def test_ok_when_mlflow_reachable(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["mlflow_reachable"] is True
        assert body["mlflow_error"] is None
        assert body["cache"]["max_bytes"] > 0

    def test_degraded_but_still_200_when_mlflow_is_down(self, client, fake_client):
        """The ALB health check must not kill a task just because MLflow is down."""
        fake_client.fail = True
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["mlflow_reachable"] is False
        assert "connection refused" in body["mlflow_error"]

    def test_probe_result_is_reused_within_its_ttl(self, client, gateway, fake_client):
        """A polling load balancer must not mean one MLflow request per poll."""
        assert client.get("/api/health").json()["status"] == "ok"
        before = len(fake_client.search_calls)

        fake_client.fail = True
        assert client.get("/api/health").json()["status"] == "ok", "cached probe should still be trusted"
        assert len(fake_client.search_calls) == before

    def test_probe_is_rechecked_once_the_ttl_expires(self, client, gateway, fake_client):
        assert client.get("/api/health").json()["status"] == "ok"
        gateway.settings.health_probe_ttl_s = 0
        fake_client.fail = True
        assert client.get("/api/health").json()["status"] == "degraded"


class TestExperiments:
    def test_lists_experiments(self, client):
        response = client.get("/api/experiments")
        assert response.status_code == 200
        experiments = response.json()["experiments"]
        assert [exp["name"] for exp in experiments] == ["inverse-thomson-scattering"]

    def test_bad_gateway_when_mlflow_is_down(self, client, fake_client):
        fake_client.fail = True
        assert client.get("/api/experiments").status_code == 502


class TestListRuns:
    def test_returns_rows_and_a_cursor(self, client):
        response = client.get("/api/runs")
        assert response.status_code == 200
        body = response.json()
        assert body["next_page_token"] == "next-cursor"
        row = body["runs"][0]
        assert row["run_id"] == "run-abc"
        assert row["shot"] == "101675"
        assert row["experiment_name"] == "inverse-thomson-scattering"
        assert row["final_loss"] == 12.5
        assert row["duration_s"] == 123.0
        assert row["user"] == "archis"

    def test_filters_reach_mlflow_as_a_filter_string(self, client, fake_client):
        client.get("/api/runs?shot=101675&status=FINISHED&user=archis")
        assert fake_client.search_calls[-1]["filter_string"] == (
            "params.\"data.shotnum\" = '101675' and "
            "attributes.status = 'FINISHED' and "
            "tags.\"mlflow.user\" = 'archis'"
        )

    def test_experiment_name_is_resolved_to_an_id(self, client, fake_client):
        client.get("/api/runs?experiment=inverse-thomson-scattering")
        assert fake_client.search_calls[-1]["experiment_ids"] == ["1"]

    def test_unknown_experiment_is_a_400(self, client):
        response = client.get("/api/runs?experiment=does-not-exist")
        assert response.status_code == 400
        assert "unknown experiment" in response.json()["detail"]

    def test_sort_is_translated(self, client, fake_client):
        client.get("/api/runs?sort=-loss")
        assert fake_client.search_calls[-1]["order_by"] == ['metrics."overall loss" DESC']

    def test_unsortable_field_is_a_400(self, client):
        assert client.get("/api/runs?sort=nonsense").status_code == 400

    def test_hostile_filter_value_is_a_400_not_a_query(self, client):
        assert client.get("/api/runs?user=o'brien").status_code == 400

    def test_page_size_is_capped(self, client, fake_client):
        client.get("/api/runs?page_size=100000")
        assert fake_client.search_calls[-1]["max_results"] == 200

    def test_page_token_is_forwarded(self, client, fake_client):
        client.get("/api/runs?page_token=abc123")
        assert fake_client.search_calls[-1]["page_token"] == "abc123"


class TestRunDetail:
    def test_returns_config_tree_and_artifacts(self, client, fake_client):
        fake_client.artifacts["run-abc"] = {
            "": [file_info("plots", is_dir=True), file_info("config.yaml", file_size=120)],
            "plots": [file_info("plots/fit_and_data.png", file_size=2048)],
        }
        response = client.get("/api/runs/run-abc")
        assert response.status_code == 200
        body = response.json()
        assert body["config"]["data"]["shotnum"] == 101675
        assert body["config_flat"]["data.shotnum"] == "101675"
        assert [a["path"] for a in body["artifacts"]] == ["plots", "plots/fit_and_data.png", "config.yaml"]
        assert body["mlflow_run_url"].endswith("/#/experiments/1/runs/run-abc")

    def test_missing_run_is_a_404(self, client):
        response = client.get("/api/runs/nope")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_manifest_is_parsed_when_present(self, client, fake_client, manifest_bytes):
        fake_client.artifacts["run-abc"] = {"": [file_info("manifest.json", file_size=len(manifest_bytes))]}
        fake_client.artifact_files["run-abc"] = {"manifest.json": manifest_bytes}
        body = client.get("/api/runs/run-abc").json()
        assert body["manifest"] == {"schema_version": 1, "datasets": ["binary/ele_fit_and_data.nc"]}

    def test_manifest_is_null_when_absent(self, client, fake_client):
        fake_client.artifacts["run-abc"] = {"": [file_info("config.yaml", file_size=12)]}
        assert client.get("/api/runs/run-abc").json()["manifest"] is None


class TestMetricHistory:
    def test_returns_points_sorted_by_step(self, client, fake_client):
        fake_client.metric_history[("run-abc", "epoch loss")] = [
            SimpleNamespace(step=2, value=5.0, timestamp=300),
            SimpleNamespace(step=0, value=9.0, timestamp=100),
            SimpleNamespace(step=1, value=7.0, timestamp=200),
        ]
        response = client.get("/api/runs/run-abc/metrics/epoch%20loss")
        assert response.status_code == 200
        body = response.json()
        assert body["key"] == "epoch loss"
        assert [p["step"] for p in body["points"]] == [0, 1, 2]
        assert [p["value"] for p in body["points"]] == [9.0, 7.0, 5.0]

    def test_metric_keys_with_spaces_survive_url_encoding(self, client, fake_client):
        """tsadar's loss metrics are named 'overall loss', 'min loss', ..."""
        fake_client.metric_history[("run-abc", "overall loss")] = [
            SimpleNamespace(step=0, value=12.5, timestamp=100)
        ]
        assert client.get("/api/runs/run-abc/metrics/overall%20loss").status_code == 200

    def test_unknown_metric_is_a_404(self, client):
        assert client.get("/api/runs/run-abc/metrics/no-such-metric").status_code == 404

    def test_missing_run_is_a_404(self, client):
        assert client.get("/api/runs/nope/metrics/epoch%20loss").status_code == 404


class TestArtifactPassthrough:
    def test_streams_a_png_with_its_content_type(self, client, fake_client):
        payload = b"\x89PNG\r\n\x1a\n" + b"0" * 64
        fake_client.artifact_files["run-abc"] = {"plots/fit_and_data.png": payload}

        response = client.get("/api/runs/run-abc/artifacts/plots/fit_and_data.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content == payload

    def test_netcdf_gets_a_useful_content_type(self, client, fake_client):
        fake_client.artifact_files["run-abc"] = {"binary/ele_fit_and_data.nc": b"CDF\x01"}
        response = client.get("/api/runs/run-abc/artifacts/binary/ele_fit_and_data.nc")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-netcdf"

    def test_traversal_is_rejected(self, client):
        response = client.get("/api/runs/run-abc/artifacts/../../etc/passwd")
        assert response.status_code in (400, 404)
        if response.status_code == 400:
            assert "'..'" in response.json()["detail"]

    def test_missing_artifact_is_a_404(self, client, fake_client):
        fake_client.artifact_files["run-abc"] = {}
        assert client.get("/api/runs/run-abc/artifacts/plots/absent.png").status_code == 404

    def test_second_request_is_served_from_cache(self, client, fake_client, gateway):
        fake_client.artifact_files["run-abc"] = {"config.yaml": b"data:\n  shotnum: 101675\n"}

        assert client.get("/api/runs/run-abc/artifacts/config.yaml").status_code == 200
        cached = gateway.cache.get("run-abc", "config.yaml")
        assert cached is not None and cached.read_bytes().startswith(b"data:")

        # Break the client: a cache hit must not need it.
        fake_client.artifact_files["run-abc"] = {}
        assert client.get("/api/runs/run-abc/artifacts/config.yaml").status_code == 200

    def test_running_run_artifacts_are_not_cached(self, client, fake_client):
        fake_client.runs["run-live"] = make_run(run_id="run-live", status="RUNNING", end_time=None)
        fake_client.artifact_files["run-live"] = {"config.yaml": b"first"}
        assert client.get("/api/runs/run-live/artifacts/config.yaml").content == b"first"

        fake_client.artifact_files["run-live"] = {"config.yaml": b"second"}
        assert client.get("/api/runs/run-live/artifacts/config.yaml").content == b"second"


class TestOpenApi:
    def test_schema_is_served_and_covers_the_read_layer(self, client):
        """The frontend client is generated from this; #29 requires it stay honest.

        A subset assertion rather than equality: later issues legitimately add
        endpoints, and this test is about the read layer staying present, not
        about the API never growing.
        """
        response = client.get("/api/openapi.json")
        assert response.status_code == 200
        paths = response.json()["paths"]
        assert {
            "/api/health",
            "/api/experiments",
            "/api/runs",
            "/api/runs/{run_id}",
            "/api/runs/{run_id}/metrics/{key}",
            "/api/runs/{run_id}/artifacts/{artifact_path}",
        } <= set(paths)
