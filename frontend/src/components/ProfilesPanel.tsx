/**
 * Fitted parameters versus lineout, with sigma error bars where the run has them.
 *
 * The series are whatever `learned_parameters.csv` holds -- `<param>_<species>`,
 * so `Te_electron`, `ne_electron`, `Ti_ion`, flows and so on. They are not
 * hardcoded, because which parameters were active varies per run; the panel
 * groups them by parameter name so Te for two species overlays sensibly.
 *
 * `calc_sigmas` is off by default, so most runs have no error bars. That is
 * normal and reported rather than treated as missing data.
 */

import { useEffect, useMemo, useState } from "react";

import { ApiError, api, type Profiles } from "../api/client";
import { axisLabel } from "../lib/format";
import { Plot } from "./Plot";

/** Split `Te_electron` into its parameter and species halves. */
export function splitSeriesName(name: string): { parameter: string; species: string | null } {
  const underscore = name.lastIndexOf("_");
  if (underscore <= 0) return { parameter: name, species: null };
  return { parameter: name.slice(0, underscore), species: name.slice(underscore + 1) };
}

/** Group series by parameter so the same quantity for different species shares a
 *  plot, rather than producing one chart per column. */
export function groupSeries(profiles: Profiles): Array<{ parameter: string; names: string[] }> {
  const groups = new Map<string, string[]>();
  for (const series of profiles.series) {
    const { parameter } = splitSeriesName(series.name);
    const existing = groups.get(parameter);
    if (existing) existing.push(series.name);
    else groups.set(parameter, [series.name]);
  }
  return [...groups.entries()].map(([parameter, names]) => ({ parameter, names }));
}

interface ProfilesPanelProps {
  runId: string;
  lineoutIndex: number;
  onLineoutChange: (index: number) => void;
}

export function ProfilesPanel({ runId, lineoutIndex, onLineoutChange }: ProfilesPanelProps) {
  const [profiles, setProfiles] = useState<Profiles | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    api
      .profiles(runId, controller.signal)
      .then(setProfiles)
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setProfiles(null);
        setError(cause instanceof ApiError ? cause.message : "Could not load profiles.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [runId]);

  const groups = useMemo(() => (profiles ? groupSeries(profiles) : []), [profiles]);

  if (loading) {
    return (
      <section className="panel">
        <h2>Profiles</h2>
        <p className="panel__status">Loading profiles…</p>
      </section>
    );
  }

  if (error || !profiles) {
    return (
      <section className="panel">
        <h2>Profiles</h2>
        <p className="panel__status panel__status--error" role="alert">
          {error ?? "No profiles available."}
        </p>
      </section>
    );
  }

  const byName = new Map(profiles.series.map((series) => [series.name, series]));

  return (
    <section className="panel" aria-labelledby="profiles-heading">
      <header className="panel__header">
        <h2 id="profiles-heading">Fitted parameters</h2>
        <span className="panel__meta">
          {profiles.sigmas_available ? "with uncertainties" : "no uncertainties logged"}
        </span>
      </header>

      <div className="profiles__grid">
        {groups.map((group) => {
          const traces = group.names.map((name) => {
            const series = byName.get(name)!;
            const { species } = splitSeriesName(name);
            return {
              type: "scatter",
              mode: "lines+markers",
              name: species ?? name,
              x: profiles.x,
              y: series.values,
              error_y: series.sigma
                ? { type: "data", array: series.sigma, visible: true, thickness: 1 }
                : undefined,
            };
          });

          return (
            <div key={group.parameter} className="profiles__cell">
              <Plot
                data={traces}
                layout={{
                  title: { text: group.parameter, font: { size: 13 } },
                  xaxis: { title: axisLabel(profiles.x_label) },
                  margin: { t: 30, r: 12, b: 40, l: 52 },
                  showlegend: group.names.length > 1,
                  shapes: [
                    {
                      type: "line",
                      x0: profiles.x[lineoutIndex] ?? null,
                      x1: profiles.x[lineoutIndex] ?? null,
                      yref: "paper",
                      y0: 0,
                      y1: 1,
                      line: { color: "#ff7f0e", width: 1, dash: "dot" },
                    },
                  ],
                }}
                height={220}
                ariaLabel={`${group.parameter} versus lineout`}
                onPointClick={onLineoutChange}
              />
            </div>
          );
        })}
      </div>

      {!profiles.sigmas_available && (
        <p className="panel__note">
          This run did not compute uncertainties (<code>calc_sigmas</code> is off by default), so the
          profiles have no error bars.
        </p>
      )}
    </section>
  );
}
