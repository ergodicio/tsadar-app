"""Tests for Thomson experiment scoping.

The tracking server is shared with every other Ergodic project, so the browser
restricts every run query to experiments that actually hold Thomson runs. These
tests cover how that set is decided (:mod:`tsadar_browser.thomson`) and that the
gateway and routes actually honour it.

The fake here is deliberately stricter than ``conftest``'s: it implements MLflow's
param-existence filter and terminates pagination, because discovery depends on
both.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tsadar_browser.app import create_app
from tsadar_browser.cache import ArtifactCache
from tsadar_browser.deps import get_gateway
from tsadar_browser.gateway import MlflowGateway, NotThomson
from tsadar_browser.settings import Settings
from tsadar_browser.thomson import (
    SEED_EXPERIMENTS,
    THOMSON_PARAM_MARKERS,
    ThomsonRegistry,
    discover_experiment_names,
    experiment_is_thomson,
)

from .conftest import make_run


def experiment(experiment_id: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(
        experiment_id=experiment_id,
        name=name,
        artifact_location=f"s3://public-ergodic-continuum/{experiment_id}",
        lifecycle_stage="active",
        creation_time=1_700_000_000_000,
        last_update_time=1_700_000_000_000,
        tags={},
    )


class _Page(list):
    def __init__(self, items, token=None):
        super().__init__(items)
        self.token = token


class DiscoveryFake:
    """A fake that honours MLflow's param-existence filter and ends its pages.

    ``search_runs`` is given runs keyed by experiment id. A filter of the form
    ``params.`key` != 'sentinel'`` is matched the way the real server matches it:
    only runs that logged ``key`` at all come back.
    """

    def __init__(self, experiments, runs_by_experiment, page_size_limit=None):
        self.experiments = experiments
        self.runs_by_experiment = runs_by_experiment
        self.page_size_limit = page_size_limit
        self.search_run_calls: list[dict] = []
        self.failures = 0

    def search_experiments(self, view_type=None, max_results=None):
        return self.experiments

    def get_experiment(self, experiment_id):
        for exp in self.experiments:
            if exp.experiment_id == experiment_id:
                return exp
        raise KeyError(experiment_id)

    def get_experiment_by_name(self, name):
        return next((exp for exp in self.experiments if exp.name == name), None)

    def get_run(self, run_id):
        for runs in self.runs_by_experiment.values():
            for run in runs:
                if run.info.run_id == run_id:
                    return run
        raise KeyError(run_id)

    def list_artifacts(self, run_id, path=""):
        # No artifacts: the dataset probe then answers 'supported: false' with a
        # reason, which is still a 200 and is all these tests need from it.
        return []

    @staticmethod
    def _marker(filter_string: str) -> str | None:
        if not filter_string or "!=" not in filter_string:
            return None
        return filter_string.split("`")[1]

    def search_runs(
        self,
        experiment_ids,
        filter_string="",
        run_view_type=None,
        max_results=50,
        order_by=None,
        page_token=None,
    ):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("tracking server unreachable")

        self.search_run_calls.append({"experiment_ids": list(experiment_ids), "filter_string": filter_string})

        marker = self._marker(filter_string)
        matching = [
            run
            for experiment_id in experiment_ids
            for run in self.runs_by_experiment.get(experiment_id, [])
            if marker is None or marker in (run.data.params or {})
        ]

        # Real pagination: a token only when there is genuinely more.
        limit = self.page_size_limit or max_results
        offset = int(page_token or 0)
        window = matching[offset : offset + limit]
        next_offset = offset + len(window)
        token = str(next_offset) if next_offset < len(matching) else None
        return _Page(window, token=token)


THOMSON_RUN_PARAMS = {"data.shotnum": "101675", "other.extraoptions.spectype": "temporal"}
LEGACY_RUN_PARAMS = {"D.extraoptions.spectype": "temporal", "specCurvature": "1.0"}
ADEPT_RUN_PARAMS = {"solver": "vlasov1d", "grid.nx": "1024", "mlflow.experiment": "vp-turbulence"}


@pytest.fixture
def mixed_server() -> DiscoveryFake:
    """Two Thomson experiments (one modern, one legacy) and two ADEPT ones."""
    experiments = [
        experiment("1", "shot_day_3_27_24"),
        experiment("2", "inverse-thomson-scattering"),
        experiment("3", "vp-turbulence"),
        experiment("4", "lagradept-peak-power-scan"),
    ]
    runs = {
        "1": [make_run(run_id="ts-1", experiment_id="1", params=dict(THOMSON_RUN_PARAMS))],
        "2": [make_run(run_id="ts-legacy", experiment_id="2", params=dict(LEGACY_RUN_PARAMS))],
        "3": [make_run(run_id="adept-1", experiment_id="3", params=dict(ADEPT_RUN_PARAMS))],
        "4": [make_run(run_id="adept-2", experiment_id="4", params=dict(ADEPT_RUN_PARAMS))],
    }
    return DiscoveryFake(experiments, runs)


def build_settings(tmp_path, **overrides) -> Settings:
    defaults = {
        "mlflow_tracking_uri": "https://continuum.ergodic.io/experiments",
        "cache_dir": tmp_path / "cache",
        "cache_max_gb": 0.001,
        "cors_origins": [],
    }
    defaults.update(overrides)
    return Settings(**defaults)


def build_gateway(tmp_path, client, **overrides) -> MlflowGateway:
    settings = build_settings(tmp_path, **overrides)
    cache = ArtifactCache(root=settings.cache_dir, max_bytes=settings.cache_max_bytes)
    return MlflowGateway(settings=settings, cache=cache, client=client)


class TestDiscovery:
    def test_finds_experiments_holding_thomson_runs(self, mixed_server):
        assert discover_experiment_names(mixed_server) == {"shot_day_3_27_24", "inverse-thomson-scattering"}

    def test_legacy_only_experiment_is_found(self, mixed_server):
        """The whole reason the legacy markers exist.

        ``inverse-thomson-scattering`` here holds *only* a pre-refactor run with
        no ``data.shotnum``. The modern marker alone would miss it, and every run
        in it would vanish from the browser.
        """
        found = discover_experiment_names(mixed_server)
        assert "inverse-thomson-scattering" in found

    def test_queries_one_filter_per_marker(self, mixed_server):
        discover_experiment_names(mixed_server)
        markers = {call["filter_string"].split("`")[1] for call in mixed_server.search_run_calls}
        assert markers == set(THOMSON_PARAM_MARKERS)

    def test_pages_until_exhausted(self):
        """Discovery must not stop at the first page, or it misses experiments.

        The Thomson runs here are ordered so the only run from experiment 9 is on
        the last page; a single-page scan would classify it as non-Thomson.
        """
        experiments = [experiment(str(i), f"exp-{i}") for i in range(1, 10)]
        runs = {
            str(i): [make_run(run_id=f"r{i}", experiment_id=str(i), params=dict(THOMSON_RUN_PARAMS))]
            for i in range(1, 10)
        }
        fake = DiscoveryFake(experiments, runs, page_size_limit=2)

        found = discover_experiment_names(fake)
        assert found == {f"exp-{i}" for i in range(1, 10)}

    def test_no_experiments_means_no_names(self):
        assert discover_experiment_names(DiscoveryFake([], {})) == set()


class TestRegistryNames:
    def test_seed_is_used_before_discovery_completes(self, tmp_path, mixed_server):
        registry = ThomsonRegistry(client=mixed_server, settings=build_settings(tmp_path))
        names, source = registry.names()
        assert source == "seed"
        assert names == tuple(sorted(SEED_EXPERIMENTS))

    def test_discovery_replaces_the_seed(self, tmp_path, mixed_server):
        registry = ThomsonRegistry(client=mixed_server, settings=build_settings(tmp_path))
        registry.refresh_now()
        names, source = registry.names()
        assert source == "discovered"
        assert names == ("inverse-thomson-scattering", "shot_day_3_27_24")

    def test_explicit_allowlist_wins_and_skips_discovery(self, tmp_path, mixed_server):
        registry = ThomsonRegistry(
            client=mixed_server,
            settings=build_settings(tmp_path, thomson_experiments=["shot_day_3_27_24"]),
        )
        names, source = registry.names()
        assert (names, source) == (("shot_day_3_27_24",), "configured")
        assert registry.stale() is False
        registry.resolve(mixed_server.experiments)
        assert mixed_server.search_run_calls == []

    def test_extra_is_added_and_exclude_is_removed(self, tmp_path, mixed_server):
        registry = ThomsonRegistry(
            client=mixed_server,
            settings=build_settings(
                tmp_path,
                thomson_experiments_extra=["hand-added"],
                thomson_experiments_exclude=["shot_day_3_27_24"],
            ),
        )
        registry.refresh_now()
        names, _ = registry.names()
        assert names == ("hand-added", "inverse-thomson-scattering")

    def test_exclude_applies_to_an_explicit_allowlist_too(self, tmp_path, mixed_server):
        registry = ThomsonRegistry(
            client=mixed_server,
            settings=build_settings(
                tmp_path,
                thomson_experiments=["a", "b"],
                thomson_experiments_exclude=["b"],
            ),
        )
        assert registry.names()[0] == ("a",)

    def test_comma_separated_env_form_is_accepted(self, tmp_path):
        settings = build_settings(tmp_path, thomson_experiments="one, two ,three")
        assert settings.thomson_experiments == ["one", "two", "three"]

    def test_failed_discovery_keeps_the_previous_answer(self, tmp_path, mixed_server):
        """An unreachable server must never widen the browser back out.

        Falling back to 'everything' on failure would turn a blip into a page of
        Vlasov runs, which is the bug this module exists to prevent.
        """
        registry = ThomsonRegistry(client=mixed_server, settings=build_settings(tmp_path))
        registry.refresh_now()
        discovered = registry.names()[0]

        mixed_server.failures = len(THOMSON_PARAM_MARKERS)
        registry.refresh_now()

        assert registry.names()[0] == discovered
        assert "unreachable" in (registry.resolve(mixed_server.experiments).error or "")

    def test_first_discovery_failure_falls_back_to_the_seed(self, tmp_path, mixed_server):
        mixed_server.failures = len(THOMSON_PARAM_MARKERS)
        registry = ThomsonRegistry(client=mixed_server, settings=build_settings(tmp_path))
        registry.refresh_now()
        names, source = registry.names()
        assert source == "seed"
        assert names == tuple(sorted(SEED_EXPERIMENTS))


class TestRegistryResolve:
    def test_resolves_names_to_ids_and_ignores_unknown_names(self, tmp_path, mixed_server):
        registry = ThomsonRegistry(
            client=mixed_server,
            settings=build_settings(tmp_path, thomson_experiments=["shot_day_3_27_24", "not-on-this-server"]),
        )
        scope = registry.resolve(mixed_server.experiments)
        assert scope.experiment_ids == frozenset({"1"})
        assert scope.names == ("shot_day_3_27_24",)
        assert scope.scoped is True

    def test_unresolvable_scope_allows_everything(self, tmp_path, mixed_server):
        """Fail open, and say so, rather than serving an empty browser."""
        registry = ThomsonRegistry(
            client=mixed_server, settings=build_settings(tmp_path, thomson_experiments=["nothing-matches"])
        )
        scope = registry.resolve(mixed_server.experiments)
        assert scope.scoped is False
        assert scope.allows("3") is True

    def test_scoped_registry_rejects_out_of_scope_ids(self, tmp_path, mixed_server):
        registry = ThomsonRegistry(
            client=mixed_server, settings=build_settings(tmp_path, thomson_experiments=["shot_day_3_27_24"])
        )
        scope = registry.resolve(mixed_server.experiments)
        assert scope.allows("1") is True
        assert scope.allows("3") is False

    def test_stale_scope_triggers_one_background_refresh(self, tmp_path, mixed_server):
        clock = iter([0.0] * 200)
        registry = ThomsonRegistry(
            client=mixed_server,
            settings=build_settings(tmp_path, thomson_registry_ttl_s=0.0),
            clock=lambda: next(clock, 0.0),
        )
        assert registry.stale() is True
        registry.resolve(mixed_server.experiments)
        # The refresh runs on a daemon thread; joining it is what makes the
        # assertion deterministic rather than a race on scheduling.
        for thread in __import__("threading").enumerate():
            if thread.name == "thomson-discovery":
                thread.join(timeout=5)
        assert registry.names()[1] == "discovered"


class TestMarkerHelper:
    def test_recognises_modern_and_legacy_params(self):
        assert experiment_is_thomson(THOMSON_RUN_PARAMS) is True
        assert experiment_is_thomson(LEGACY_RUN_PARAMS) is True

    def test_rejects_another_project(self):
        assert experiment_is_thomson(ADEPT_RUN_PARAMS) is False

    def test_rejects_a_run_with_no_params(self):
        assert experiment_is_thomson({}) is False


class TestGatewayScoping:
    def test_run_search_covers_only_thomson_experiments(self, tmp_path, mixed_server):
        gateway = build_gateway(tmp_path, mixed_server)
        gateway.thomson.refresh_now()
        mixed_server.search_run_calls.clear()

        gateway.search_runs()

        assert mixed_server.search_run_calls[-1]["experiment_ids"] == ["1", "2"]

    def test_experiment_listing_hides_other_projects(self, tmp_path, mixed_server):
        gateway = build_gateway(tmp_path, mixed_server)
        gateway.thomson.refresh_now()

        names = [exp.name for exp in gateway.list_experiments()]
        assert names == ["shot_day_3_27_24", "inverse-thomson-scattering"]

    def test_named_non_thomson_experiment_is_refused(self, tmp_path, mixed_server):
        gateway = build_gateway(tmp_path, mixed_server)
        gateway.thomson.refresh_now()

        with pytest.raises(NotThomson, match="not a Thomson analysis experiment"):
            gateway.search_runs(experiment="vp-turbulence")

    def test_named_thomson_experiment_is_allowed(self, tmp_path, mixed_server):
        gateway = build_gateway(tmp_path, mixed_server)
        gateway.thomson.refresh_now()

        gateway.search_runs(experiment="shot_day_3_27_24")
        assert mixed_server.search_run_calls[-1]["experiment_ids"] == ["1"]

    def test_run_detail_refuses_another_project_s_run(self, tmp_path, mixed_server):
        gateway = build_gateway(tmp_path, mixed_server)
        gateway.thomson.refresh_now()

        with pytest.raises(NotThomson, match="not a Thomson analysis run"):
            gateway.get_run("adept-1")

    def test_marker_bearing_run_survives_an_unclassified_experiment(self, tmp_path, mixed_server):
        """A shot day created since the last discovery must not 404.

        Scope is refreshed hourly; a run logged into a brand-new experiment in
        between still carries its Thomson params, and that is enough.
        """
        gateway = build_gateway(tmp_path, mixed_server)
        gateway.thomson.refresh_now()

        mixed_server.experiments.append(experiment("9", "shot_day_brand_new"))
        fresh = make_run(run_id="ts-fresh", experiment_id="9", params=dict(THOMSON_RUN_PARAMS))
        mixed_server.runs_by_experiment["9"] = [fresh]
        gateway._experiments_cache = None

        assert gateway.get_run("ts-fresh").run_id == "ts-fresh"

    def test_artifacts_of_another_project_s_run_are_refused(self, tmp_path, mixed_server):
        """Scope is enforced on the bytes route too, not only on the detail route."""
        gateway = build_gateway(tmp_path, mixed_server)
        gateway.thomson.refresh_now()

        with pytest.raises(NotThomson):
            gateway.download_artifact("adept-1", "plots/fit_and_data.png")


class TestScopedApi:
    @pytest.fixture
    def scoped_client(self, tmp_path, mixed_server) -> TestClient:
        from tsadar_browser.datasets import DatasetService
        from tsadar_browser.deps import get_dataset_service

        gateway = build_gateway(tmp_path, mixed_server)
        gateway.thomson.refresh_now()
        app = create_app()
        app.dependency_overrides[get_gateway] = lambda: gateway
        # Overriding the gateway alone is not enough: the dataset routes depend on
        # get_dataset_service, whose default singleton builds its own gateway
        # against the real tracking server.
        app.dependency_overrides[get_dataset_service] = lambda: DatasetService(gateway=gateway)
        return TestClient(app)

    def test_experiments_endpoint_is_thomson_only(self, scoped_client):
        names = [exp["name"] for exp in scoped_client.get("/api/experiments").json()["experiments"]]
        assert names == ["shot_day_3_27_24", "inverse-thomson-scattering"]

    def test_non_thomson_experiment_filter_is_a_404_with_a_reason(self, scoped_client):
        response = scoped_client.get("/api/runs?experiment=vp-turbulence")
        assert response.status_code == 404
        assert response.json()["detail"]["reason"] == "not_thomson"

    def test_non_thomson_run_detail_is_a_404_with_a_reason(self, scoped_client):
        response = scoped_client.get("/api/runs/adept-1")
        assert response.status_code == 404
        assert response.json()["detail"]["reason"] == "not_thomson"

    def test_thomson_run_detail_still_works(self, scoped_client):
        assert scoped_client.get("/api/runs/ts-1").status_code == 200

    def test_the_dataset_probe_is_scoped_too(self, scoped_client):
        """The probe answers for every run *in scope*, not for every run anywhere.

        It classifies from an artifact listing rather than fetching bytes, so it
        is the one dataset path that does not inherit download_artifact's guard.
        Regression: it returned 200 with supported:false for an ADEPT run while
        /api/runs/{id} next to it returned 404.
        """
        response = scoped_client.get("/api/runs/adept-1/datasets")
        assert response.status_code == 404
        assert response.json()["detail"]["reason"] == "not_thomson"

    def test_the_dataset_probe_still_answers_for_thomson_runs(self, scoped_client):
        assert scoped_client.get("/api/runs/ts-1/datasets").status_code == 200

    @pytest.mark.parametrize("path", ["spectrogram", "lineout?which=ele&index=0", "profiles"])
    def test_every_dataset_route_refuses_a_non_thomson_run(self, scoped_client, path):
        response = scoped_client.get(f"/api/runs/adept-1/{path}")
        assert response.status_code == 404
        assert response.json()["detail"]["reason"] == "not_thomson"

    def test_health_reports_the_scope(self, scoped_client):
        thomson = scoped_client.get("/api/health").json()["thomson"]
        assert thomson["scoped"] is True
        assert thomson["experiment_count"] == 2
        assert thomson["source"] == "discovered"
        assert sorted(thomson["experiments"]) == ["inverse-thomson-scattering", "shot_day_3_27_24"]
