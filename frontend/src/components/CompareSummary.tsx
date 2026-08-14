/**
 * Side-by-side summary table.
 *
 * `loss_key` is shown next to each final loss rather than assumed: runs log
 * different loss metrics, and a column that silently mixed `overall loss` with
 * `min loss` would invite exactly the wrong conclusion from a comparison.
 */

import { memo, useMemo } from "react";
import { Link } from "react-router-dom";

import { formatDuration, formatLoss, formatTimestamp, spectypeLabel } from "../lib/format";
import { runLabel, type ComparisonRun } from "../lib/compare";

interface CompareSummaryProps {
  runs: ComparisonRun[];
  onRemove: (runId: string) => void;
}

function CompareSummaryImpl({ runs, onRemove }: CompareSummaryProps) {
  const lossKeys = useMemo(
    () => [...new Set(runs.map((run) => run.detail.loss_key).filter(Boolean))],
    [runs],
  );
  const mixedLossMetrics = lossKeys.length > 1;

  return (
    <section className="panel" aria-labelledby="summary-heading">
      <header className="panel__header">
        <h2 id="summary-heading">Runs</h2>
        <span className="panel__meta">{runs.length} selected</span>
      </header>

      <div className="comparetable__scroll">
        <table className="comparetable">
          <thead>
            <tr>
              <th scope="col">Field</th>
              {runs.map((run) => (
                <th key={run.runId} scope="col">
                  <Link to={`/runs/${run.runId}`}>{runLabel(run)}</Link>
                  <button
                    type="button"
                    className="comparetable__remove"
                    aria-label={`Remove ${runLabel(run)} from the comparison`}
                    onClick={() => onRemove(run.runId)}
                  >
                    ×
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">Shot</th>
              {runs.map((run) => (
                <td key={run.runId}>{run.detail.shot ?? "—"}</td>
              ))}
            </tr>
            <tr>
              <th scope="row">Type</th>
              {runs.map((run) => (
                <td key={run.runId}>{spectypeLabel(run.detail.spectype)}</td>
              ))}
            </tr>
            <tr>
              <th scope="row">Status</th>
              {runs.map((run) => (
                <td key={run.runId}>
                  {run.detail.status ?? "—"}
                  {run.detail.stage && <span className="comparetable__stage"> / {run.detail.stage}</span>}
                </td>
              ))}
            </tr>
            <tr>
              <th scope="row">Final loss</th>
              {runs.map((run) => (
                <td key={run.runId}>
                  {formatLoss(run.detail.final_loss)}
                  {run.detail.loss_key && (
                    <span className="comparetable__losskey"> ({run.detail.loss_key})</span>
                  )}
                </td>
              ))}
            </tr>
            <tr>
              <th scope="row">Runtime</th>
              {runs.map((run) => (
                <td key={run.runId}>{formatDuration(run.detail.duration_s)}</td>
              ))}
            </tr>
            <tr>
              <th scope="row">Started</th>
              {runs.map((run) => (
                <td key={run.runId}>{formatTimestamp(run.detail.start_time)}</td>
              ))}
            </tr>
            <tr>
              <th scope="row">User</th>
              {runs.map((run) => (
                <td key={run.runId}>{run.detail.user ?? "—"}</td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>

      {mixedLossMetrics && (
        <p className="panel__note">
          These runs report <strong>different loss metrics</strong> ({lossKeys.join(", ")}), so the
          final-loss row is not directly comparable across all of them.
        </p>
      )}
    </section>
  );
}

/** `onRemove` is stable (the route memoizes it), so this only re-renders when the
 *  comparison itself changes rather than on every parent render. */
export const CompareSummary = memo(CompareSummaryImpl);
