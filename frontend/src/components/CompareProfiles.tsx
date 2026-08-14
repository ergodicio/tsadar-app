/**
 * Overlaid parameter profiles across runs.
 *
 * One plot per parameter, one trace per run. Runs legitimately differ in which
 * parameters they fitted — an ele-only run has no ion parameters — so the union
 * of series is taken and each panel says how many runs contributed. Dropping a
 * parameter because one run lacks it would hide data the others do have.
 */

import { memo, useMemo } from "react";

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

function CompareProfilesImpl({ runs }: { runs: ComparisonRun[] }) {
  const comparable = useMemo(() => comparableRuns(runs), [runs]);

  // One memo for every panel rather than one per panel: `Plot` re-plots when the
  // arrays it is handed change by identity, and this component renders N panels
  // from the same `runs`. Building them inline meant any render of this
  // component -- a parent state change, a removed run -- rebuilt every trace and
  // every layout for every parameter, and redrew the lot.
  const panels = useMemo(() => {
    const labels = axisLabels(runs);
    const xTitle = labels.map(axisLabel).join(" / ");

    return sharedSeriesNames(runs).map((name) => {
      const contributors = runsWithSeries(runs, name);
      const { parameter, species } = splitSeriesName(name);

      return {
        name,
        contributors: contributors.length,
        traces: contributors.map((run) => {
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
        }),
        layout: {
          title: {
            text: species ? `${parameter} (${species})` : parameter,
            font: { size: 13 },
          },
          // With more than one axis label in play the axis title would be a lie,
          // so it names both rather than picking one.
          xaxis: { title: xTitle },
          margin: { t: 30, r: 12, b: 40, l: 52 },
          showlegend: contributors.length > 1,
          legend: { orientation: "h", y: -0.25 },
        },
      };
    });
  }, [runs]);

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
        {panels.map((panel) => (
          <div key={panel.name} className="profiles__cell">
            <Plot
              data={panel.traces}
              layout={panel.layout}
              height={240}
              ariaLabel={`${panel.name} across runs`}
            />
            {panel.contributors < comparable.length && (
              <p className="panel__note">
                {panel.contributors} of {comparable.length} runs fitted this parameter.
              </p>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

/** Memoized because `runs` only changes when the comparison reloads, while this
 *  component's parent re-renders on every URL change. */
export const CompareProfiles = memo(CompareProfilesImpl);
