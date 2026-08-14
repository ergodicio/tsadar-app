/**
 * Overlaid loss curves.
 *
 * Each run may have logged different loss metric names, so the panel offers the
 * union and fetches whichever key a run actually has. A run without the selected
 * key is listed as absent rather than silently missing from the legend.
 */

import { useEffect, useMemo, useState } from "react";

import { api, type MetricHistory } from "../api/client";
import { runLabel, type ComparisonRun } from "../lib/compare";
import { Plot } from "./Plot";

const LOSS_PATTERN = /loss/i;

/** Loss metric keys across all runs, per-step ones first. */
export function lossKeyOptions(runs: ComparisonRun[]): string[] {
  const keys = new Set<string>();
  for (const run of runs) {
    for (const metric of run.detail.metrics) {
      if (LOSS_PATTERN.test(metric.key)) keys.add(metric.key);
    }
  }
  const rank = (key: string) => (key.includes("epoch") ? 0 : key.includes("batch") ? 1 : 2);
  return [...keys].sort((left, right) => rank(left) - rank(right) || left.localeCompare(right));
}

export function CompareLoss({ runs }: { runs: ComparisonRun[] }) {
  const options = useMemo(() => lossKeyOptions(runs), [runs]);
  const [selected, setSelected] = useState<string | null>(options[0] ?? null);
  const [histories, setHistories] = useState<Map<string, MetricHistory>>(new Map());
  const [missing, setMissing] = useState<string[]>([]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();

    const havingKey = runs.filter((run) =>
      run.detail.metrics.some((metric) => metric.key === selected),
    );

    setMissing(
      runs.filter((run) => !havingKey.includes(run)).map((run) => runLabel(run)),
    );

    Promise.all(
      havingKey.map((run) =>
        api
          .metricHistory(run.runId, selected, controller.signal)
          .then((history) => [run.runId, history] as const)
          .catch(() => null),
      ),
    ).then((results) => {
      if (controller.signal.aborted) return;
      setHistories(new Map(results.filter((entry): entry is [string, MetricHistory] => entry !== null)));
    });

    return () => controller.abort();
  }, [runs, selected]);

  if (options.length === 0) {
    return (
      <section className="panel">
        <h2>Loss curves</h2>
        <p className="panel__status">None of these runs logged a loss metric.</p>
      </section>
    );
  }

  const traces = runs
    .map((run) => {
      const history = histories.get(run.runId);
      if (!history) return null;
      return {
        type: "scatter",
        mode: history.points.length > 1 ? "lines" : "markers",
        name: runLabel(run),
        x: history.points.map((point) => point.step),
        y: history.points.map((point) => point.value),
      };
    })
    .filter((trace): trace is NonNullable<typeof trace> => trace !== null);

  return (
    <section className="panel" aria-labelledby="compare-loss-heading">
      <header className="panel__header">
        <h2 id="compare-loss-heading">Loss curves</h2>
        <label className="control">
          <span>Metric</span>
          <select value={selected ?? ""} onChange={(event) => setSelected(event.target.value)}>
            {options.map((key) => (
              <option key={key} value={key}>
                {key}
              </option>
            ))}
          </select>
        </label>
      </header>

      <Plot
        data={traces}
        layout={{
          xaxis: { title: "Step" },
          yaxis: { title: selected ?? "loss" },
          margin: { t: 12, r: 16, b: 40, l: 62 },
          showlegend: true,
        }}
        height={280}
        ariaLabel={`${selected} across runs`}
      />

      {missing.length > 0 && (
        <p className="panel__note">
          Not logged by: {missing.join(", ")}. Runs log different loss metrics, so try another key.
        </p>
      )}
    </section>
  );
}
