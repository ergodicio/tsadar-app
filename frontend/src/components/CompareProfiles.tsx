/**
 * Overlaid parameter profiles across runs.
 *
 * One plot per parameter, one trace per run. Runs legitimately differ in which
 * parameters they fitted — an ele-only run has no ion parameters — so the union
 * of series is taken and each panel says how many runs contributed. Dropping a
 * parameter because one run lacks it would hide data the others do have.
 */

import { Plot } from "./Plot";
import { axisLabel } from "../lib/format";
import {
  axisLabels,
  comparableRuns,
  runLabel,
  runsWithSeries,
  sharedSeriesNames,
  type ComparisonRun,
} from "../lib/compare";
import { splitSeriesName } from "./ProfilesPanel";

export function CompareProfiles({ runs }: { runs: ComparisonRun[] }) {
  const comparable = comparableRuns(runs);
  const names = sharedSeriesNames(runs);
  const labels = axisLabels(runs);

  if (comparable.length === 0) {
    return (
      <section className="panel">
        <h2>Parameter profiles</h2>
        <p className="panel__status">
          None of the selected runs can be overlaid. See the notes above; the config diff below still
          works.
        </p>
      </section>
    );
  }

  return (
    <section className="panel" aria-labelledby="compare-profiles-heading">
      <header className="panel__header">
        <h2 id="compare-profiles-heading">Parameter profiles</h2>
        <span className="panel__meta">
          {comparable.length} of {runs.length} run{runs.length === 1 ? "" : "s"} overlaid
        </span>
      </header>

      <div className="profiles__grid">
        {names.map((name) => {
          const contributors = runsWithSeries(runs, name);
          const { parameter, species } = splitSeriesName(name);

          const traces = contributors.map((run) => {
            const series = run.profiles!.series.find((candidate) => candidate.name === name)!;
            return {
              type: "scatter",
              mode: "lines+markers",
              name: runLabel(run),
              x: run.profiles!.x,
              y: series.values,
              error_y: series.sigma
                ? { type: "data", array: series.sigma, visible: true, thickness: 1 }
                : undefined,
            };
          });

          return (
            <div key={name} className="profiles__cell">
              <Plot
                data={traces}
                layout={{
                  title: {
                    text: species ? `${parameter} (${species})` : parameter,
                    font: { size: 13 },
                  },
                  // With more than one axis label in play the axis title would be
                  // a lie, so it names both rather than picking one.
                  xaxis: { title: labels.map(axisLabel).join(" / ") },
                  margin: { t: 30, r: 12, b: 40, l: 52 },
                  showlegend: contributors.length > 1,
                  legend: { orientation: "h", y: -0.25 },
                }}
                height={240}
                ariaLabel={`${name} across runs`}
              />
              {contributors.length < comparable.length && (
                <p className="panel__note">
                  {contributors.length} of {comparable.length} runs fitted this parameter.
                </p>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
