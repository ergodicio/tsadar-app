/**
 * The run browser: a filterable, sortable table over `GET /api/runs`.
 *
 * All filter and sort state lives in the URL so any view can be pasted into
 * Slack. The page cursor does not -- see `lib/urlState.ts`.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";

import { api, type Experiment, type ThomsonScope } from "../api/client";
import { Filters } from "../components/Filters";
import { RunTable } from "../components/RunTable";
import { EmptyState, ErrorState, LoadingState } from "../components/StateViews";
import { useRuns } from "../hooks/useRuns";
import {
  filtersFromSearch,
  hasActiveFilters,
  searchFromFilters,
  toggleSort,
  type RunFilters,
} from "../lib/urlState";

export function RunBrowser() {
  const [search, setSearch] = useSearchParams();
  const navigate = useNavigate();

  const filters = useMemo(() => filtersFromSearch(search), [search]);
  const { runs, loading, loadingMore, error, loadMoreError, hasMore, loadMore, reload } =
    useRuns(filters);

  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [scope, setScope] = useState<ThomsonScope | null>(null);
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());

  useEffect(() => {
    const controller = new AbortController();
    api
      .experiments(controller.signal)
      // A failed experiment list only costs the dropdown its options; the table
      // still works, so this is not surfaced as a page error.
      .then(setExperiments)
      .catch(() => setExperiments([]));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    // The Thomson restriction is otherwise invisible, and invisible filtering is
    // how someone concludes their run was lost. Health is the only endpoint that
    // reports it; a failure just leaves the note off.
    api
      .health(controller.signal)
      .then((health) => setScope(health.thomson ?? null))
      .catch(() => setScope(null));
    return () => controller.abort();
  }, []);

  const applyFilters = useCallback(
    (next: RunFilters) => setSearch(searchFromFilters(next), { replace: true }),
    [setSearch],
  );

  const onSort = useCallback(
    (field: string) => applyFilters({ ...filters, sort: toggleSort(filters.sort, field) }),
    [applyFilters, filters],
  );

  const onToggleSelected = useCallback((runId: string) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return next;
    });
  }, []);

  const openRun = useCallback((runId: string) => navigate(`/runs/${runId}`), [navigate]);

  const compareHref = `/compare?runs=${[...selected].join(",")}`;

  return (
    <section className="browser">
      <header className="browser__header">
        <h1>Thomson runs</h1>
        <p className="browser__hint">
          Filters and sort are in the URL, so this view can be shared as a link.
        </p>
        {scope?.scoped && (
          <p className="browser__scope" title={(scope.experiments ?? []).join("\n")}>
            Showing Thomson scattering analysis runs from {scope.experiment_count} experiment
            {scope.experiment_count === 1 ? "" : "s"}. Runs from other projects on this tracking
            server are not listed.
          </p>
        )}
        {scope && !scope.scoped && (
          // Fail-open: the backend could not work out which experiments are
          // Thomson, so the table is showing everything. Say so rather than
          // letting a page of Vlasov runs look like the intended contents.
          <p className="browser__scope browser__scope--warning" role="alert">
            Thomson experiments could not be identified
            {scope.error ? `: ${scope.error}` : ""}. Every experiment on the tracking server is
            being listed, so runs from other projects may appear.
          </p>
        )}
      </header>

      <Filters
        filters={filters}
        experiments={experiments}
        onChange={applyFilters}
        onClear={() => applyFilters({})}
      />

      {selected.size > 0 && (
        <div className="browser__selection" role="status">
          <span>
            {selected.size} run{selected.size === 1 ? "" : "s"} selected
          </span>
          {/* The compare view itself arrives with #33; the URL contract is fixed
              now so selection is not wasted work. A client-side Link rather than
              an anchor: a full document reload would discard the accumulated
              pages and the very selection this link is carrying. */}
          <Link className="button" to={compareHref}>
            Compare
          </Link>
          <button type="button" className="button" onClick={() => setSelected(new Set())}>
            Clear selection
          </button>
        </div>
      )}

      {error && <ErrorState message={error} onRetry={reload} />}
      {!error && loading && <LoadingState />}
      {!error && !loading && runs.length === 0 && (
        <EmptyState filtered={hasActiveFilters(filters)} onClear={() => applyFilters({})} />
      )}

      {!error && !loading && runs.length > 0 && (
        <>
          <RunTable
            runs={runs}
            sort={filters.sort}
            onSort={onSort}
            onOpen={openRun}
            selected={selected}
            onToggleSelected={onToggleSelected}
          />
          <footer className="browser__footer">
            <span>
              {runs.length} run{runs.length === 1 ? "" : "s"} loaded
            </span>
            {loadMoreError && (
              // Inline, not a full-page error: the pages already loaded stay put
              // because there is no cursor to resume from.
              <span className="browser__footer-error" role="alert">
                {loadMoreError}
              </span>
            )}
            {hasMore && (
              // Cursor pagination: MLflow has no offsets, so pages accumulate
              // rather than being jumped to.
              <button type="button" className="button" onClick={loadMore} disabled={loadingMore}>
                {loadingMore ? "Loading…" : "Load more"}
              </button>
            )}
          </footer>
        </>
      )}
    </section>
  );
}
