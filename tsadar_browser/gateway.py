"""Read-only gateway over the MLflow tracking server.

MLflow is the database; this module is the only place that talks to it. It
translates browser-shaped queries into MLflow ``search_runs`` filter strings and
turns MLflow's entities into the response models in :mod:`.schemas`.

Nothing here writes to MLflow.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import flatten_dict
import yaml
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from .cache import ArtifactCache, sanitize_artifact_path
from .schemas import (
    ArtifactEntry,
    Experiment,
    MetricHistory,
    MetricPoint,
    MetricSummary,
    RunDetail,
    RunPage,
    RunSummary,
)
from .s3 import S3ArtifactReader, parse_s3_uri
from .settings import Settings
from .thomson import ThomsonRegistry, ThomsonScope, experiment_is_thomson

logger = logging.getLogger(__name__)

#: MLflow statuses after which a run's artifacts can never change.
TERMINAL_STATUSES = frozenset({"FINISHED", "FAILED", "KILLED"})

#: How long the active-experiment list is reused before re-listing. Experiments
#: are created a few times a week; Thomson scope is checked on every run and
#: artifact request, so this keeps the check off the wire.
EXPERIMENT_LIST_TTL_S = 30.0

#: Param key holding the shot number. Flattened with a dot reducer by
#: ``tsadar.utils.misc.log_mlflow``, so ``data.shotnum`` in the config tree.
#: Switches to a tag once ergodicio/tsadar#115 lands.
SHOT_PARAM = "data.shotnum"

#: Param key holding the spectrum type. Reported but *not* trusted -- see
#: :attr:`schemas.RunSummary.spectype`.
SPECTYPE_PARAM = "other.extraoptions.spectype"

#: Spectrum types the browser supports interactive views for. Angular Thomson
#: needs its own diagnostics and is out of scope (see issue #37); those runs are
#: still listed and still get their PNG gallery.
ONE_D_SPECTYPES = frozenset({"temporal", "imaging"})

#: tsadar's own progress tag, distinct from the MLflow lifecycle status.
STAGE_TAG = "status"

#: Metric holding the headline loss, best first. tsadar logs several loss
#: metrics and the names contain spaces (see tsadar/inverse/fitter.py).
LOSS_KEYS = ("overall loss", "min loss", "epoch loss", "loss")

#: Sort keys the API accepts, mapped to MLflow ``order_by`` expressions.
#: Allowlisted rather than interpolated: these land in a query string.
#:
#: Note ``loss`` sorts specifically on ``overall loss`` -- MLflow can only order
#: by a named metric, so it cannot follow the same per-run fallback that
#: :data:`LOSS_KEYS` gives ``final_loss``. A run that logged only ``min loss``
#: therefore sorts as though it had no loss at all, even though the table shows
#: a value. ``loss_key`` in the response is what makes that visible, so clients
#: displaying a loss column should surface it.
SORTABLE = {
    "created": "attributes.start_time",
    "start_time": "attributes.start_time",
    "end_time": "attributes.end_time",
    "name": "attributes.run_name",
    "status": "attributes.status",
    "shot": f'params."{SHOT_PARAM}"',
    "loss": f'metrics."{LOSS_KEYS[0]}"',
}


class MlflowUnavailable(RuntimeError):
    """The tracking server could not be reached."""


class InvalidQuery(ValueError):
    """A caller-supplied filter or sort value was rejected."""


def _quote(value: str) -> str:
    """Render a value as an MLflow filter string literal.

    MLflow's filter grammar has no escape sequence for a single quote inside a
    quoted literal, so a value containing one is rejected rather than silently
    mangled into a different query.
    """
    if "'" in value or '"' in value:
        raise InvalidQuery(f"filter values must not contain quotes: {value!r}")
    if "\\" in value:
        raise InvalidQuery(f"filter values must not contain backslashes: {value!r}")
    return f"'{value}'"


def _coerce_param(raw: str) -> Any:
    """Best-effort reversal of MLflow's stringification of a config value.

    Params originate as YAML, so YAML is the right parser to get numbers,
    booleans and lists back. ``str(None)`` is handled separately because
    ``"None"`` is not a YAML null token.
    """
    if raw == "None":
        return None
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


class NotThomson(ValueError):
    """The caller asked for an experiment or run that is not Thomson analysis."""


class MlflowGateway:
    def __init__(
        self,
        settings: Settings,
        cache: ArtifactCache,
        client: MlflowClient | None = None,
        registry: ThomsonRegistry | None = None,
        s3_reader: S3ArtifactReader | None = None,
    ):
        self.settings = settings
        self.cache = cache
        self._client = client or MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        self._experiment_names: dict[str, str] = {}
        self._experiments_cache: tuple[float, list[Any]] | None = None
        self._probe: tuple[float, str | None] | None = None
        self.thomson = registry or ThomsonRegistry(client=self._client, settings=settings)
        self._s3 = s3_reader or S3ArtifactReader()

    # -- experiments ----------------------------------------------------------

    def _all_experiments(self) -> list[Any]:
        """Every active experiment, Thomson or not, with the name cache refreshed.

        The unfiltered list is what scope resolution needs (it maps Thomson
        *names* onto ids) and what row rendering needs, so it is fetched once and
        narrowed by the callers that want only Thomson experiments.

        Memoized for :data:`EXPERIMENT_LIST_TTL_S` because scope is now checked on
        the artifact path too, which the lineout scrubber hits once per step; a
        round trip per step to re-list 386 experiments that change weekly would be
        the whole cost of the guard. A new experiment appears one TTL late, which
        is nothing against the registry's own hour.
        """
        now = time.monotonic()
        if self._experiments_cache is not None:
            fetched_at, cached = self._experiments_cache
            if now - fetched_at < EXPERIMENT_LIST_TTL_S:
                return cached

        experiments = list(self._client.search_experiments(view_type=ViewType.ACTIVE_ONLY))
        self._experiment_names = {exp.experiment_id: exp.name for exp in experiments}
        self._experiments_cache = (now, experiments)
        return experiments

    def thomson_scope(self) -> ThomsonScope:
        """The current Thomson experiment scope, resolved against the live list."""
        return self.thomson.resolve(self._all_experiments())

    def list_experiments(self) -> list[Experiment]:
        """Only experiments holding Thomson runs.

        This is a Thomson browser, so the experiment picker must not offer the
        350-odd ADEPT experiments that share this tracking server. When the scope
        cannot be resolved at all the full list is returned rather than an empty
        one -- ``/api/health`` reports that as ``scoped: false``.
        """
        experiments = self._all_experiments()
        scope = self.thomson.resolve(experiments)
        return [
            Experiment(
                experiment_id=exp.experiment_id,
                name=exp.name,
                artifact_location=exp.artifact_location,
                lifecycle_stage=exp.lifecycle_stage,
                creation_time=exp.creation_time,
                last_update_time=exp.last_update_time,
                tags=dict(exp.tags or {}),
            )
            for exp in experiments
            if scope.allows(exp.experiment_id)
        ]

    def _experiment_name(self, experiment_id: str) -> str | None:
        if experiment_id not in self._experiment_names:
            try:
                experiment = self._client.get_experiment(experiment_id)
            except Exception:  # noqa: BLE001 - a missing experiment must not fail the row
                logger.debug("could not resolve experiment %s", experiment_id)
                return None
            self._experiment_names[experiment_id] = experiment.name
        return self._experiment_names.get(experiment_id)

    def _resolve_experiment_ids(self, experiment: str | None) -> list[str]:
        """Resolve an experiment name (or id) to the ids to search.

        With no experiment given this searches **every Thomson experiment** rather
        than every active one: the tracking server is shared, and the runs a
        physicist wants are a ~9% slice of it (see :mod:`.thomson`). MLflow takes
        the id list natively, so cursor pagination, filters and sorting are
        unaffected -- and scanning 35 experiments instead of 386 is faster.

        A named experiment that is not Thomson is rejected rather than searched,
        so a stale bookmark says why instead of quietly returning ADEPT runs.
        """
        experiments = self._all_experiments()
        scope = self.thomson.resolve(experiments)

        if experiment:
            found = next((exp for exp in experiments if exp.name == experiment), None)
            if found is None:
                # Tolerate an id being passed where a name is expected.
                found = next((exp for exp in experiments if exp.experiment_id == experiment), None)
            if found is None:
                try:
                    found = self._client.get_experiment(experiment)
                except Exception as exc:  # noqa: BLE001
                    raise InvalidQuery(f"unknown experiment: {experiment!r}") from exc
            if not scope.allows(found.experiment_id):
                raise NotThomson(
                    f"{experiment!r} is not a Thomson analysis experiment. This browser covers "
                    f"Thomson scattering runs only; {found.name!r} holds runs from another project."
                )
            return [found.experiment_id]

        if not scope.scoped:
            # Scope could not be resolved; searching everything is the honest
            # fallback and /api/health reports it. See ThomsonScope.allows.
            return [exp.experiment_id for exp in experiments]

        return sorted(scope.experiment_ids)

    # -- runs -----------------------------------------------------------------

    @staticmethod
    def build_filter(
        shot: str | None = None,
        status: str | None = None,
        user: str | None = None,
        stage: str | None = None,
        q: str | None = None,
    ) -> str:
        """Translate browser filters into an MLflow filter string."""
        clauses: list[str] = []

        if shot:
            clauses.append(f'params."{SHOT_PARAM}" = {_quote(shot)}')
        if status:
            normalized = status.upper()
            if normalized not in TERMINAL_STATUSES | {"RUNNING", "SCHEDULED"}:
                raise InvalidQuery(f"unknown status: {status!r}")
            clauses.append(f"attributes.status = {_quote(normalized)}")
        if user:
            clauses.append(f'tags."mlflow.user" = {_quote(user)}')
        if stage:
            clauses.append(f'tags."{STAGE_TAG}" = {_quote(stage)}')
        if q:
            # `%` and `_` inside q stay live SQL LIKE wildcards. MLflow's filter
            # grammar offers no ESCAPE clause and _quote already rejects the
            # backslash an escape would need, so this is documented as the
            # behavior rather than silently half-escaped.
            clauses.append(f"attributes.run_name LIKE {_quote(f'%{q}%')}")

        return " and ".join(clauses)

    @staticmethod
    def build_order_by(sort: str | None) -> list[str]:
        """Translate a ``field`` / ``-field`` sort key into MLflow order_by."""
        if not sort:
            return ["attributes.start_time DESC"]

        descending = sort.startswith("-")
        field = sort.lstrip("-")
        if field not in SORTABLE:
            raise InvalidQuery(f"cannot sort by {field!r}; sortable fields are {sorted(SORTABLE)}")
        return [f"{SORTABLE[field]} {'DESC' if descending else 'ASC'}"]

    def search_runs(
        self,
        experiment: str | None = None,
        shot: str | None = None,
        status: str | None = None,
        user: str | None = None,
        stage: str | None = None,
        q: str | None = None,
        sort: str | None = None,
        page_size: int = 50,
        page_token: str | None = None,
    ) -> RunPage:
        experiment_ids = self._resolve_experiment_ids(experiment)
        if not experiment_ids:
            return RunPage(runs=[], page_size=page_size, next_page_token=None)

        paged = self._client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=self.build_filter(shot=shot, status=status, user=user, stage=stage, q=q),
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=page_size,
            order_by=self.build_order_by(sort),
            page_token=page_token,
        )

        return RunPage(
            runs=[self._summarize(run) for run in paged],
            page_size=page_size,
            next_page_token=getattr(paged, "token", None),
        )

    def _summarize(self, run: Any) -> RunSummary:
        info, data = run.info, run.data
        metrics = dict(data.metrics or {})
        params = dict(data.params or {})
        tags = dict(data.tags or {})

        loss_key = next((key for key in LOSS_KEYS if key in metrics), None)

        duration_s = None
        if info.start_time and info.end_time:
            duration_s = round((info.end_time - info.start_time) / 1000.0, 2)

        return RunSummary(
            run_id=info.run_id,
            run_name=getattr(info, "run_name", None) or tags.get("mlflow.runName"),
            experiment_id=info.experiment_id,
            experiment_name=self._experiment_name(info.experiment_id),
            status=info.status,
            stage=tags.get(STAGE_TAG),
            shot=params.get(SHOT_PARAM),
            spectype=params.get(SPECTYPE_PARAM),
            final_loss=metrics.get(loss_key) if loss_key else None,
            loss_key=loss_key,
            start_time=info.start_time,
            end_time=info.end_time,
            duration_s=duration_s,
            user=tags.get("mlflow.user"),
        )

    def _require_thomson_run(self, run: Any) -> None:
        """Reject a run that is not Thomson analysis.

        Scope is by experiment, but a run carrying a Thomson marker is admitted
        even from an experiment the registry has not classified yet: a shot day
        created since the last discovery pass would otherwise 404 for up to one
        TTL, which reads as data loss rather than as a stale index.
        """
        scope = self.thomson.resolve(self._all_experiments())
        if scope.allows(run.info.experiment_id):
            return
        if experiment_is_thomson(dict(run.data.params or {})):
            logger.info(
                "run %s is in unclassified experiment %s but carries a Thomson marker; allowing",
                run.info.run_id,
                run.info.experiment_id,
            )
            return
        name = self._experiment_name(run.info.experiment_id) or run.info.experiment_id
        raise NotThomson(
            f"run {run.info.run_id} is not a Thomson analysis run. This browser covers Thomson "
            f"scattering runs only; it belongs to {name!r}, which holds runs from another project."
        )

    def require_thomson(self, run_id: str) -> None:
        """Raise :class:`NotThomson` unless this run is in scope.

        For paths that reach a run without going through :meth:`get_run` or
        :meth:`download_artifact` -- notably the dataset capability probe, which
        answers from an artifact *listing* and so never fetches bytes.
        """
        self._require_thomson_run(self._client.get_run(run_id))

    def get_run(self, run_id: str) -> RunDetail:
        run = self._client.get_run(run_id)
        self._require_thomson_run(run)
        summary = self._summarize(run)
        params = dict(run.data.params or {})

        config, unflatten_error = self._unflatten_params(params)
        artifacts = self.list_artifacts(run_id)

        return RunDetail(
            **summary.model_dump(),
            artifact_uri=run.info.artifact_uri,
            mlflow_run_url=self.run_url(run.info.experiment_id, run_id),
            config=config,
            config_flat=params,
            config_unflatten_error=unflatten_error,
            tags=dict(run.data.tags or {}),
            metrics=[
                MetricSummary(key=key, value=value) for key, value in sorted((run.data.metrics or {}).items())
            ],
            artifacts=artifacts,
            manifest=self.read_manifest(run_id, artifacts),
        )

    @staticmethod
    def _unflatten_params(params: dict[str, str]) -> tuple[dict[str, Any], str | None]:
        """Rebuild the nested config tree from dot-flattened params.

        Mirrors ``tsadar.utils.misc.log_mlflow``, which flattens with
        ``reducer="dot"``. Colliding keys (a scalar at ``a.b`` alongside
        ``a.b.c``) make the tree ill-defined; report that rather than guessing.
        """
        if not params:
            return {}, None
        coerced = {key: _coerce_param(value) for key, value in params.items()}
        try:
            return flatten_dict.unflatten(coerced, splitter="dot"), None
        except Exception as exc:  # noqa: BLE001 - flatten_dict raises bare ValueError/TypeError
            logger.warning("could not unflatten params into a config tree: %s", exc)
            return {}, str(exc)

    def run_url(self, experiment_id: str, run_id: str) -> str:
        return f"{self.settings.mlflow_ui_base}/#/experiments/{experiment_id}/runs/{run_id}"

    def get_metric_history(self, run_id: str, key: str) -> MetricHistory:
        history = self._client.get_metric_history(run_id, key)
        return MetricHistory(
            key=key,
            points=[
                MetricPoint(step=metric.step, value=metric.value, timestamp=metric.timestamp)
                for metric in sorted(history, key=lambda metric: (metric.step, metric.timestamp))
            ],
        )

    # -- artifacts ------------------------------------------------------------

    def list_artifacts(self, run_id: str, path: str = "") -> list[ArtifactEntry]:
        """Recursively list a run's artifacts, flattened to file entries plus dirs."""
        entries: list[ArtifactEntry] = []
        try:
            listing = self._client.list_artifacts(run_id, path)
        except Exception as exc:  # noqa: BLE001 - an unreachable artifact store must not 500 the run page
            logger.warning("could not list artifacts for run %s at %r: %s", run_id, path, exc)
            return entries

        for item in listing:
            entries.append(ArtifactEntry(path=item.path, is_dir=item.is_dir, size=item.file_size))
            if item.is_dir:
                entries.extend(self.list_artifacts(run_id, item.path))
        return entries

    def read_manifest(self, run_id: str, artifacts: list[ArtifactEntry] | None = None) -> dict[str, Any] | None:
        """Parse ``manifest.json`` if the run logged one (ergodicio/tsadar#116)."""
        if artifacts is not None and not any(entry.path == "manifest.json" for entry in artifacts):
            return None
        try:
            local = self.download_artifact(run_id, "manifest.json")
            return json.loads(local.read_text())
        except Exception as exc:  # noqa: BLE001 - absent or malformed manifest is expected pre-contract
            logger.debug("no readable manifest.json for run %s: %s", run_id, exc)
            return None

    def download_artifact(self, run_id: str, artifact_path: str, cacheable: bool | None = None) -> Path:
        """Return a local path to an artifact, using the disk cache.

        The frontend never sees S3: this is the only route to artifact bytes,
        which is also why scope is enforced here and not only on the detail
        route -- otherwise a non-Thomson run's artifacts stay reachable by URL.

        One ``get_run`` serves three purposes -- the scope check, the
        terminal-status check that decides cacheability, and the ``artifact_uri``
        that locates the bytes in S3 -- so none of them costs an extra round trip.

        Bytes come **straight from S3** rather than through MLflow's artifact
        repository whenever the store is S3; see :mod:`.s3` for why. A non-S3
        store (a local ``file://`` mlruns) falls back to the MLflow client.
        """
        safe_path = sanitize_artifact_path(artifact_path)

        run = None
        try:
            run = self._client.get_run(run_id)
        except Exception:  # noqa: BLE001
            # An unreadable run leaves scope unverified. Tolerated rather than
            # refused, matching what this method did before scoping existed: the
            # fetch below has no artifact_uri and no cacheability either, so it
            # falls through to MLflow and fails there. The one thing that can
            # still be served is an artifact already in the cache during an
            # MLflow outage, which is acceptable -- Thomson scope is a product
            # boundary over a public bucket, not an access control.
            logger.warning(
                "could not read run %s before fetching %s; scope unverified", run_id, safe_path
            )
        else:
            self._require_thomson_run(run)
            if cacheable is None:
                cacheable = run.info.status in TERMINAL_STATUSES
        if cacheable is None:
            cacheable = False

        artifact_uri = getattr(run.info, "artifact_uri", None) if run is not None else None
        s3_uri = parse_s3_uri(artifact_uri) if self.settings.artifact_s3_direct else None

        def download(scratch: Path) -> Path:
            if s3_uri is not None:
                return self._s3.download(artifact_uri, safe_path, scratch)
            local = self._client.download_artifacts(run_id, safe_path, str(scratch))
            return Path(local)

        return self.cache.fetch(run_id, safe_path, download, cacheable=cacheable)

    def ping(self) -> None:
        """Raise :class:`MlflowUnavailable` if the tracking server is unreachable.

        The result is remembered for ``health_probe_ttl_s`` so a load balancer
        polling ``/api/health`` does not translate into one MLflow request per
        poll. Request timeouts and retries are bounded by
        :meth:`Settings.apply_to_environment`, so a hard failure surfaces in
        seconds rather than minutes.
        """
        now = time.monotonic()
        if self._probe is not None:
            checked_at, error = self._probe
            if now - checked_at < self.settings.health_probe_ttl_s:
                if error is not None:
                    raise MlflowUnavailable(error)
                return

        try:
            self._client.search_experiments(max_results=1)
        except Exception as exc:  # noqa: BLE001
            self._probe = (now, str(exc))
            raise MlflowUnavailable(str(exc)) from exc

        self._probe = (now, None)
