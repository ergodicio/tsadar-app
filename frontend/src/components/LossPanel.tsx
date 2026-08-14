/**
 * Loss history.
 *
 * There is no metric called `loss`. tsadar logs `overall loss`, `min loss`,
 * `epoch loss` and `batch loss` -- names with **spaces** in them -- so this picks
 * from what the run actually reported rather than requesting a fixed key that
 * would 404 on every run.
 *
 * `epoch loss` is the one with a real history worth plotting (it is logged per
 * step); `overall loss` is typically a single summary point. So the panel prefers
 * whichever key has the most points and lets you switch.
 */

import { useEffect, useMemo, useState } from "react";

import { ApiError, api, type MetricHistory, type RunDetail } from "../api/client";
import { Plot } from "./Plot";

/** Metric keys that are loss curves, in rough order of usefulness as a series. */
const LOSS_KEY_PATTERN = /loss/i;

export function lossKeys(run: RunDetail): string[] {
  const keys = run.metrics.map((metric) => metric.key).filter((key) => LOSS_KEY_PATTERN.test(key));
  // Per-step histories first: "epoch loss" and "batch loss" are logged with a
  // step, while "overall loss" and "min loss" are summaries.
  const rank = (key: string) => (key.includes("epoch") ? 0 : key.includes("batch") ? 1 : 2);
  return keys.sort((left, right) => rank(left) - rank(right) || left.localeCompare(right));
}

interface LossPanelProps {
  runId: string;
  run: RunDetail;
}

export function LossPanel({ runId, run }: LossPanelProps) {
  const keys = useMemo(() => lossKeys(run), [run]);
  const [selected, setSelected] = useState<string | null>(keys[0] ?? null);
  const [history, setHistory] = useState<MetricHistory | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    setError(null);

    api
      .metricHistory(runId, selected, controller.signal)
      .then(setHistory)
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setHistory(null);
        setError(cause instanceof ApiError ? cause.message : "Could not load the loss history.");
      });

    return () => controller.abort();
  }, [runId, selected]);

  if (keys.length === 0) {
    return (
      <section className="panel">
        <h2>Loss</h2>
        <p className="panel__status">This run logged no loss metrics.</p>
      </section>
    );
  }

  const traces = history
    ? [
        {
          type: "scatter",
          mode: history.points.length > 1 ? "lines+markers" : "markers",
          x: history.points.map((point) => point.step),
          y: history.points.map((point) => point.value),
          name: history.key,
        },
      ]
    : [];

  return (
    <section className="panel" aria-labelledby="loss-heading">
      <header className="panel__header">
        <h2 id="loss-heading">Loss</h2>
        <label className="control">
          <span>Metric</span>
          <select value={selected ?? ""} onChange={(event) => setSelected(event.target.value)}>
            {keys.map((key) => (
              <option key={key} value={key}>
                {key}
              </option>
            ))}
          </select>
        </label>
      </header>

      {error && (
        <p className="panel__status panel__status--error" role="alert">
          {error}
        </p>
      )}

      {history && !error && (
        <>
          <Plot
            data={traces}
            layout={{
              xaxis: { title: "Step" },
              yaxis: { title: history.key },
              margin: { t: 12, r: 16, b: 40, l: 62 },
            }}
            height={240}
            ariaLabel={`${history.key} history`}
          />
          {history.points.length === 1 && (
            <p className="panel__note">
              A single point: <code>{history.key}</code> is logged once as a summary rather than per
              step. Try <code>epoch loss</code> for a curve.
            </p>
          )}
        </>
      )}
    </section>
  );
}
