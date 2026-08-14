/**
 * Run identity and status.
 *
 * Status and stage are shown as two separate badges on purpose: MLflow's
 * lifecycle status and tsadar's own progress tag answer different questions, and
 * a `FAILED` run whose stage is stuck at `minimizing` tells you where it died.
 * Collapsing them into one badge would throw that away.
 */

import { Link } from "react-router-dom";

import type { RunDetail } from "../api/client";
import { formatDuration, formatLoss, formatTimestamp, isAngular, spectypeLabel } from "../lib/format";

export function RunHeader({ run }: { run: RunDetail }) {
  const facts: Array<[string, string]> = [
    ["Shot", run.shot ?? "—"],
    ["Experiment", run.experiment_name ?? run.experiment_id],
    ["Type", spectypeLabel(run.spectype)],
    ["Submitted by", run.user ?? "—"],
    ["Started", formatTimestamp(run.start_time)],
    ["Duration", formatDuration(run.duration_s)],
    [run.loss_key ? `Final loss (${run.loss_key})` : "Final loss", formatLoss(run.final_loss)],
  ];

  return (
    <header className="runheader">
      <div className="runheader__title">
        <Link to="/runs" className="runheader__back">
          ← Runs
        </Link>
        <h1>{run.run_name ?? run.run_id}</h1>

        <span className={`badge status--${(run.status ?? "unknown").toLowerCase()}`}>
          {run.status ?? "unknown"}
        </span>
        {run.stage && (
          <span className="badge badge--muted" title="tsadar's own progress tag">
            {run.stage}
          </span>
        )}
        {isAngular(run.spectype) && (
          <span className="badge badge--muted" title="Interactive views are limited to 1D Thomson">
            angular
          </span>
        )}

        {run.mlflow_run_url && (
          <a className="button runheader__mlflow" href={run.mlflow_run_url} target="_blank" rel="noreferrer">
            Open in MLflow
          </a>
        )}
      </div>

      <dl className="runheader__facts">
        {facts.map(([label, value]) => (
          <div key={label} className="runheader__fact">
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>

      <p className="runheader__runid">
        <code>{run.run_id}</code>
      </p>
    </header>
  );
}
