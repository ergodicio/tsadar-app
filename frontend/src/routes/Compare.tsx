/**
 * Multi-run comparison.
 *
 * `/compare?runs=a,b,c` fully encodes the comparison, so it can be pasted into
 * Slack. That also means the run ids are arbitrary user input rather than a
 * selection made in the run browser, so every incompatibility has to be handled
 * here rather than assumed away upstream.
 */

import { useCallback, useMemo } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { CompareConfig } from "../components/CompareConfig";
import { CompareLoss } from "../components/CompareLoss";
import { CompareProfiles } from "../components/CompareProfiles";
import { CompareSummary } from "../components/CompareSummary";
import { ErrorState, LoadingState } from "../components/StateViews";
import { useComparison } from "../hooks/useComparison";
import { mixedAxisWarning, parseRunIds, runLabel } from "../lib/compare";

export function Compare() {
  const [search, setSearch] = useSearchParams();
  const runIds = useMemo(() => parseRunIds(search.get("runs")), [search]);

  const { runs, failures, loading, reload } = useComparison(runIds);

  const removeRun = useCallback(
    (runId: string) => {
      const next = new URLSearchParams(search);
      const remaining = runIds.filter((candidate) => candidate !== runId);
      if (remaining.length) next.set("runs", remaining.join(","));
      else next.delete("runs");
      setSearch(next, { replace: true });
    },
    [runIds, search, setSearch],
  );

  if (runIds.length === 0) {
    return (
      <section className="state">
        <p className="state__title">No runs selected</p>
        <p className="state__detail">
          Pick runs in the run browser and choose Compare, or pass them directly as
          <code> /compare?runs=a,b,c</code>.
        </p>
        <Link className="button" to="/runs">
          Go to runs
        </Link>
      </section>
    );
  }

  if (loading) return <LoadingState />;

  if (runs.length === 0) {
    return (
      <ErrorState
        message={
          failures.length
            ? `None of the selected runs could be loaded. ${failures[0]?.message ?? ""}`
            : "None of the selected runs could be loaded."
        }
        onRetry={reload}
      />
    );
  }

  const excluded = runs.filter((run) => run.excluded !== null);
  const mixedAxes = mixedAxisWarning(runs);

  return (
    <article className="detail">
      <header className="runheader">
        <div className="runheader__title">
          <Link to="/runs" className="runheader__back">
            ← Runs
          </Link>
          <h1>Comparing {runs.length} runs</h1>
        </div>
      </header>

      {failures.length > 0 && (
        <p className="panel__status panel__status--error" role="alert">
          Could not load: {failures.map((failure) => `${failure.runId} (${failure.message})`).join("; ")}
        </p>
      )}

      {excluded.length > 0 && (
        // Excluded-and-why, not silently dropped: a run missing from the overlay
        // with no explanation looks like a bug.
        <div className="panel__status panel__status--notice" role="status">
          <p>Not included in the overlays:</p>
          <ul className="compare__excluded">
            {excluded.map((run) => (
              <li key={run.runId}>
                <strong>{runLabel(run)}</strong> — {run.excluded}
              </li>
            ))}
          </ul>
        </div>
      )}

      {mixedAxes && (
        <p className="panel__status panel__status--notice" role="status">
          {mixedAxes}
        </p>
      )}

      <CompareSummary runs={runs} onRemove={removeRun} />
      <CompareProfiles runs={runs} />
      <CompareLoss runs={runs} />
      <CompareConfig runs={runs} />
    </article>
  );
}
