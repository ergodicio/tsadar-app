"""Gateway unit tests: filter translation, sorting, and the config round-trip."""

import pytest

from tsadar_browser.gateway import InvalidQuery, MlflowGateway

from .conftest import make_run


class TestBuildFilter:
    def test_empty_when_no_filters(self):
        assert MlflowGateway.build_filter() == ""

    def test_shot_targets_the_flattened_param(self):
        assert MlflowGateway.build_filter(shot="101675") == "params.\"data.shotnum\" = '101675'"

    def test_combines_clauses_with_and(self):
        built = MlflowGateway.build_filter(shot="101675", status="finished", user="archis")
        assert built == (
            "params.\"data.shotnum\" = '101675' and "
            "attributes.status = 'FINISHED' and "
            "tags.\"mlflow.user\" = 'archis'"
        )

    def test_stage_uses_the_tsadar_status_tag(self):
        assert MlflowGateway.build_filter(stage="minimizing") == "tags.\"status\" = 'minimizing'"

    def test_free_text_becomes_a_like_on_run_name(self):
        assert MlflowGateway.build_filter(q="scan") == "attributes.run_name LIKE '%scan%'"

    def test_unknown_status_rejected(self):
        with pytest.raises(InvalidQuery):
            MlflowGateway.build_filter(status="not-a-status")

    @pytest.mark.parametrize("hostile", ["o'brien", 'say "hi"', "back\\slash"])
    def test_quotes_and_backslashes_rejected(self, hostile):
        """MLflow's filter grammar cannot escape these, so they must not be interpolated."""
        with pytest.raises(InvalidQuery):
            MlflowGateway.build_filter(user=hostile)


class TestBuildOrderBy:
    def test_defaults_to_newest_first(self):
        assert MlflowGateway.build_order_by(None) == ["attributes.start_time DESC"]

    def test_ascending_and_descending(self):
        assert MlflowGateway.build_order_by("name") == ["attributes.run_name ASC"]
        assert MlflowGateway.build_order_by("-name") == ["attributes.run_name DESC"]

    def test_loss_sorts_on_the_real_metric_name(self):
        assert MlflowGateway.build_order_by("loss") == ['metrics."overall loss" ASC']

    def test_unknown_sort_key_rejected(self):
        with pytest.raises(InvalidQuery):
            MlflowGateway.build_order_by("; drop table runs")


class TestConfigRoundTrip:
    def test_params_unflatten_into_a_nested_tree(self, gateway):
        detail = gateway.get_run("run-abc")
        assert detail.config["data"]["shotnum"] == 101675
        assert detail.config["data"]["lineouts"]["start"] == 800
        assert detail.config["data"]["lineouts"]["type"]["ps"] == "ps"
        assert detail.config["parameters"]["electron"]["Te"]["val"] == 0.5

    def test_types_are_recovered_not_left_as_strings(self, gateway):
        config = gateway.get_run("run-abc").config
        assert config["other"]["extraoptions"]["load_ion_spec"] is False
        assert config["parameters"]["electron"]["Te"]["active"] is True
        assert config["other"]["refit_thresh"] == 5.0
        assert config["other"]["lamrangE"] == [400, 700]

    def test_none_string_becomes_none(self, gateway):
        """MLflow stores Python None as the literal 'None', which is not YAML null."""
        assert gateway.get_run("run-abc").config["parameters"]["general"]["Va"]["angle"] is None

    def test_flat_params_are_preserved_alongside_the_tree(self, gateway):
        detail = gateway.get_run("run-abc")
        assert detail.config_flat["data.shotnum"] == "101675"
        assert detail.config_unflatten_error is None

    def test_colliding_dotted_keys_report_an_error_instead_of_guessing(self, settings, cache, fake_client):
        fake_client.runs["run-bad"] = make_run(
            run_id="run-bad", params={"a.b": "1", "a.b.c": "2"}
        )
        gateway = MlflowGateway(settings=settings, cache=cache, client=fake_client)

        detail = gateway.get_run("run-bad")
        assert detail.config == {}
        assert detail.config_unflatten_error
        assert detail.config_flat == {"a.b": "1", "a.b.c": "2"}


class TestSummaries:
    def test_duration_computed_from_timestamps(self, gateway):
        assert gateway.get_run("run-abc").duration_s == pytest.approx(123.0)

    def test_duration_is_null_while_running(self, settings, cache, fake_client):
        fake_client.runs["run-live"] = make_run(run_id="run-live", status="RUNNING", end_time=None)
        gateway = MlflowGateway(settings=settings, cache=cache, client=fake_client)
        assert gateway.get_run("run-live").duration_s is None

    def test_loss_key_is_reported_alongside_the_value(self, gateway):
        detail = gateway.get_run("run-abc")
        assert detail.loss_key == "overall loss"
        assert detail.final_loss == 12.5

    def test_falls_back_through_the_loss_key_preference_order(self, settings, cache, fake_client):
        fake_client.runs["run-min"] = make_run(run_id="run-min", metrics={"min loss": 3.0})
        gateway = MlflowGateway(settings=settings, cache=cache, client=fake_client)
        detail = gateway.get_run("run-min")
        assert (detail.loss_key, detail.final_loss) == ("min loss", 3.0)

    def test_no_loss_metric_leaves_both_null(self, settings, cache, fake_client):
        fake_client.runs["run-noloss"] = make_run(run_id="run-noloss", metrics={"fit_time": 1.0})
        gateway = MlflowGateway(settings=settings, cache=cache, client=fake_client)
        detail = gateway.get_run("run-noloss")
        assert detail.loss_key is None and detail.final_loss is None

    def test_spectype_is_reported(self, settings, cache, fake_client):
        from .conftest import SAMPLE_PARAMS

        fake_client.runs["run-temporal"] = make_run(
            run_id="run-temporal",
            params={**SAMPLE_PARAMS, "other.extraoptions.spectype": "temporal"},
        )
        gateway = MlflowGateway(settings=settings, cache=cache, client=fake_client)
        assert gateway.get_run("run-temporal").spectype == "temporal"

    def test_angular_spectype_is_reported_not_suppressed(self, settings, cache, fake_client):
        """Angular runs stay listed (issue #37); only interactive views are out of scope."""
        from .conftest import SAMPLE_PARAMS

        fake_client.runs["run-ang"] = make_run(
            run_id="run-ang",
            params={**SAMPLE_PARAMS, "other.extraoptions.spectype": "angular_full"},
        )
        gateway = MlflowGateway(settings=settings, cache=cache, client=fake_client)
        assert gateway.get_run("run-ang").spectype == "angular_full"

    def test_spectype_is_null_when_not_logged(self, settings, cache, fake_client):
        fake_client.runs["run-bare"] = make_run(run_id="run-bare", params={"data.shotnum": "1"})
        gateway = MlflowGateway(settings=settings, cache=cache, client=fake_client)
        assert gateway.get_run("run-bare").spectype is None

    def test_stage_tag_is_surfaced_separately_from_lifecycle_status(self, gateway):
        detail = gateway.get_run("run-abc")
        assert detail.status == "FINISHED"
        assert detail.stage == "completed"

    def test_run_url_deep_links_into_the_mlflow_ui(self, gateway):
        detail = gateway.get_run("run-abc")
        assert detail.mlflow_run_url == (
            "https://continuum.ergodic.io/experiments/#/experiments/1/runs/run-abc"
        )


class TestArtifactListing:
    def test_listing_recurses_into_directories(self, settings, cache, fake_client):
        from .conftest import file_info

        fake_client.artifacts["run-abc"] = {
            "": [file_info("binary", is_dir=True), file_info("config.yaml", file_size=120)],
            "binary": [file_info("binary/ele_fit_and_data.nc", file_size=4096)],
        }
        gateway = MlflowGateway(settings=settings, cache=cache, client=fake_client)

        # Depth-first: each directory is followed immediately by its contents,
        # so the flat list still reads as a tree.
        paths = [entry.path for entry in gateway.list_artifacts("run-abc")]
        assert paths == ["binary", "binary/ele_fit_and_data.nc", "config.yaml"]

        by_path = {entry.path: entry for entry in gateway.list_artifacts("run-abc")}
        assert by_path["binary"].is_dir is True
        assert by_path["binary/ele_fit_and_data.nc"].size == 4096

    def test_unreadable_artifact_store_does_not_raise(self, settings, cache, fake_client):
        """The run page must still render if S3 is unreachable."""
        fake_client.fail = True
        gateway = MlflowGateway(settings=settings, cache=cache, client=fake_client)
        assert gateway.list_artifacts("run-abc") == []
