"""Which MLflow experiments hold Thomson analysis runs.

The tracking server is shared by every Ergodic project. At the time of writing it
has 386 active experiments, of which **35 hold Thomson runs**; the rest are ADEPT
and friends (``lagradept-*``, ``vp-turbulence``, ``vlasov*``, ``tpd-*``,
``osiris_*``, ``warpx-*``). Searching all of them is not a mild annoyance: the
1500 most recently started runs on the server contain *zero* Thomson runs, so an
unscoped browser opens on a page of Vlasov runs and never shows a shot day at
all.

Scoping therefore happens at the **experiment** level. That is a deliberate
choice over the two alternatives, both of which were measured against the live
server and rejected:

*Per-run param matching does not work.* The obvious marker is
``params."data.shotnum"``, present in every current tsadar deck (all four of
``configs/*/defaults.yaml``) and on 4700 runs. But 345 runs in
``inverse-thomson-scattering`` predate that config layout and carry one of
several older schemas instead -- a ``D.*`` deck with flat ``parameters.amp1.*``,
and an older fully flat deck of ``Te`` / ``Ti`` / ``specCurvature`` / ``fitprops``
-- and **no single param key covers all of them**. The union cannot be expressed
as one query either, because MLflow's filter grammar has **no OR**: the live
server rejects ``a != x OR b != y`` with ``INVALID_PARAMETER_VALUE``. Filtering
runs on ``data.shotnum`` alone would silently drop 345 genuine Thomson runs,
which is the exact failure mode ``docs/browser.md`` already refuses for
``spectype``.

*A separate database is not needed.* Thomson and non-Thomson experiments are
**disjoint** -- no experiment mixes them. So the whole filter reduces to a set of
experiment ids handed to ``search_runs``, which MLflow accepts natively. Cursor
pagination, filter strings and sort keys all keep working untouched, and queries
get *faster* because 35 experiments are scanned instead of 386. MLflow stays the
source of truth and nothing has to be exported anywhere.

How the set is known, in priority order:

1. ``THOMSON_EXPERIMENTS`` -- an explicit operator allowlist. Set it and
   discovery is skipped entirely.
2. Otherwise :data:`SEED_EXPERIMENTS` (a snapshot, see below) plus whatever
   :func:`discover_experiment_names` finds, refreshed in the background on a TTL.

Either way ``THOMSON_EXPERIMENTS_EXTRA`` is added and
``THOMSON_EXPERIMENTS_EXCLUDE`` is removed, so an operator can correct the
verdict without a deploy.

The seed matters because discovery takes ~50 seconds: a cold container must serve
correct results on its first request rather than blocking or showing everything.
The seed is a *snapshot, not the truth* -- background discovery supersedes it
within one TTL, which is what makes future shot days appear on their own.
"""

import logging
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from mlflow.entities import ViewType

logger = logging.getLogger(__name__)

#: Param keys whose presence on a run proves it came from a tsadar deck.
#:
#: Only ever used to classify an *experiment*, never to filter runs within one,
#: so this does not need to cover every run -- one matching run is enough to
#: mark the experiment, and every run in it is then shown.
#:
#: ``data.shotnum`` alone identifies all 35 experiments today; the three legacy
#: keys resolve to ``inverse-thomson-scattering``, which ``data.shotnum`` already
#: covers via its 1053 modern runs. They are kept as insurance against an
#: experiment holding *only* pre-refactor runs, which the modern key would miss
#: entirely. Each is verified filterable against the live server.
THOMSON_PARAM_MARKERS: tuple[str, ...] = (
    "data.shotnum",
    "D.extraoptions.spectype",
    "specCurvature",
    "lineoutloc.type",
)

#: Experiments known to hold Thomson runs, as discovered on 2026-08-18.
#:
#: A seed so a cold container is correct immediately, **not** a list to maintain:
#: background discovery replaces it within one TTL, so a new shot day needs no
#: code change. Stale entries are harmless -- a name that no longer exists simply
#: does not resolve to an id.
SEED_EXPERIMENTS: tuple[str, ...] = (
    "Kinshock19A",
    "Kinshock19A-92532",
    "Kinshock19A-92533",
    "Kinshock19A-92534",
    "Kinshock19A-92535",
    "Kinshock19A-92536",
    "Kinshock19A-92537",
    "Kinshock19A-H2",
    "Kinshock26A-117827",
    "Kinshock26A-117828",
    "Kinshock26A-117830",
    "Kinshock26A-117837",
    "Kinshock26A-117838",
    "Kinshock26A-117839",
    "arts-adam",
    "arts-fp",
    "arts-smoothing",
    "flux_cbet",
    "iKinshock19A-92534",
    "inverse-thomson-scattering",
    "inverse-thomson-scattering-d2",
    "julie-inverse-thomson-scattering",
    "llnl-inverse-thomson-scattering",
    "princeton_ts",
    "shot_day_10_22_25",
    "shot_day_3_27_24",
    "ts-batch_size-scaling",
    "ts-bo",
    "ts-bo-sym",
    "ts-figs",
    "ts-lle",
    "ts-lle-kyle",
    "tsadar-test",
    "tsadar-tests",
    "tsadar-tests-2",
)

#: Runs per discovery page. Discovery reads only ``experiment_id`` off each run,
#: so the aim is the fewest requests rather than the smallest payload: 4700
#: marker-bearing runs come back in 5 pages instead of 94.
DISCOVERY_PAGE_SIZE = 1000

#: Hard ceiling on pages per marker, so a server that keeps handing back cursors
#: cannot turn a background refresh into an unbounded loop.
DISCOVERY_MAX_PAGES = 200


class _Client(Protocol):
    """The two ``MlflowClient`` methods discovery needs."""

    def search_experiments(self, *args: Any, **kwargs: Any) -> Any: ...

    def search_runs(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class ThomsonScope:
    """The resolved verdict for one request.

    ``experiment_ids`` empty means *do not scope* -- see
    :meth:`ThomsonRegistry.resolve`.
    """

    experiment_ids: frozenset[str]
    names: tuple[str, ...]
    source: str
    discovered_at: float | None
    stale: bool
    error: str | None

    @property
    def scoped(self) -> bool:
        return bool(self.experiment_ids)

    def allows(self, experiment_id: str) -> bool:
        """Whether a run in this experiment is in scope.

        An unscoped registry allows everything rather than nothing: a browser
        that 404s every run because discovery failed is worse than one that
        shows too much and says so through ``/api/health``.
        """
        return not self.scoped or experiment_id in self.experiment_ids


def discover_experiment_names(client: _Client) -> set[str]:
    """Names of every active experiment holding at least one Thomson run.

    One paged ``search_runs`` per marker across all active experiments. Costs 8
    requests and ~50 seconds against the production server, which is why callers
    run it in the background rather than on a request.
    """
    experiments = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
    names = {exp.experiment_id: exp.name for exp in experiments}
    if not names:
        return set()

    found: set[str] = set()
    for marker in THOMSON_PARAM_MARKERS:
        # `!= sentinel` is how an existence test is spelled in MLflow's grammar:
        # a param filter only matches runs that logged the key at all, so every
        # run carrying it passes a comparison against a value nothing holds.
        filter_string = f"params.`{marker}` != '__tsadar_browser_no_such_value__'"
        token = None
        for _ in range(DISCOVERY_MAX_PAGES):
            page = client.search_runs(
                experiment_ids=list(names),
                filter_string=filter_string,
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=DISCOVERY_PAGE_SIZE,
                order_by=["attributes.start_time DESC"],
                page_token=token,
            )
            for run in page:
                name = names.get(run.info.experiment_id)
                if name is not None:
                    found.add(name)
            token = getattr(page, "token", None)
            if not token:
                break
        else:
            logger.warning("Thomson discovery hit the page ceiling for marker %r", marker)

    return found


class ThomsonRegistry:
    """The set of Thomson experiment names, refreshed in the background.

    Reads never block on MLflow: :meth:`resolve` answers from the seed (or the
    last successful discovery) and kicks off a refresh only when the current
    answer has aged past the TTL.
    """

    def __init__(self, client: _Client, settings: Any, *, clock: Any = time.monotonic):
        self._client = client
        self._settings = settings
        self._clock = clock
        self._lock = threading.Lock()

        self._discovered: set[str] | None = None
        self._discovered_at: float | None = None
        self._error: str | None = None
        self._refreshing = False

    # -- configuration --------------------------------------------------------

    @property
    def _configured(self) -> tuple[str, ...]:
        return tuple(getattr(self._settings, "thomson_experiments", ()) or ())

    @property
    def _extra(self) -> tuple[str, ...]:
        return tuple(getattr(self._settings, "thomson_experiments_extra", ()) or ())

    @property
    def _excluded(self) -> frozenset[str]:
        return frozenset(getattr(self._settings, "thomson_experiments_exclude", ()) or ())

    @property
    def _ttl_s(self) -> float:
        return float(getattr(self._settings, "thomson_registry_ttl_s", 3600.0))

    # -- current best answer --------------------------------------------------

    def names(self) -> tuple[tuple[str, ...], str]:
        """The names currently considered Thomson, and where they came from."""
        with self._lock:
            discovered, source = self._discovered, "discovered"

        if self._configured:
            # An explicit allowlist is the operator's final word: discovery is
            # not consulted, and not started.
            base, source = set(self._configured), "configured"
        elif discovered is not None:
            base = set(discovered)
        else:
            base, source = set(SEED_EXPERIMENTS), "seed"

        base |= set(self._extra)
        base -= self._excluded
        return tuple(sorted(base)), source

    def stale(self) -> bool:
        if self._configured:
            return False
        with self._lock:
            if self._discovered_at is None:
                return True
            return (self._clock() - self._discovered_at) >= self._ttl_s

    def resolve(self, experiments: Sequence[Any]) -> ThomsonScope:
        """Map the Thomson names onto ids from an already-fetched experiment list.

        Takes the live list rather than fetching it so a caller that already has
        one (every ``/api/runs``) does not pay a second round trip. Resolution is
        by **name**: ids are server-specific, names are what a physicist and an
        operator both recognise.
        """
        if not self._configured:
            self._maybe_refresh()

        wanted, source = self.names()
        by_name = {exp.name: exp.experiment_id for exp in experiments}
        ids = frozenset(by_name[name] for name in wanted if name in by_name)

        with self._lock:
            discovered_at, error = self._discovered_at, self._error

        if not ids:
            # Nothing resolved: an empty tracking server, a wholly unrecognised
            # one, or an exclude list that removed everything. Report it rather
            # than silently serving an empty browser.
            logger.warning(
                "no Thomson experiments resolved from %d active experiments (source=%s); "
                "run scoping is disabled and every run will be listed",
                len(experiments),
                source,
            )

        return ThomsonScope(
            experiment_ids=ids,
            names=tuple(name for name in wanted if name in by_name),
            source=source,
            discovered_at=discovered_at,
            stale=self.stale(),
            error=error,
        )

    # -- refresh --------------------------------------------------------------

    def _maybe_refresh(self) -> None:
        """Start a background discovery if the current answer has gone stale."""
        if not self.stale():
            return
        with self._lock:
            if self._refreshing:
                return
            self._refreshing = True

        thread = threading.Thread(target=self._refresh_in_background, name="thomson-discovery", daemon=True)
        thread.start()

    def _refresh_in_background(self) -> None:
        try:
            self.refresh_now()
        finally:
            with self._lock:
                self._refreshing = False

    def refresh_now(self) -> tuple[str, ...]:
        """Run discovery synchronously and adopt the result.

        A failure is recorded and surfaced through ``/api/health``, leaving the
        previous answer (or the seed) in place -- an unreachable tracking server
        must not widen the browser back out to every experiment.
        """
        try:
            found = discover_experiment_names(self._client)
        except Exception as exc:  # noqa: BLE001 - any upstream failure keeps the old answer
            logger.warning("Thomson experiment discovery failed: %s", exc)
            with self._lock:
                self._error = str(exc)
                # Back off for one TTL rather than retrying on every request.
                self._discovered_at = self._clock() if self._discovered is None else self._discovered_at
            return self.names()[0]

        with self._lock:
            self._discovered = found
            self._discovered_at = self._clock()
            self._error = None

        logger.info("Thomson discovery found %d experiments", len(found))
        return self.names()[0]


def experiment_is_thomson(params: Iterable[str]) -> bool:
    """Whether a run's param keys carry a Thomson marker.

    The in-process counterpart to the MLflow filter in
    :func:`discover_experiment_names`, for callers that already hold a run.
    """
    keys = set(params)
    return any(marker in keys for marker in THOMSON_PARAM_MARKERS)
