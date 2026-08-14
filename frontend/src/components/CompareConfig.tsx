/**
 * N-way config diff: flattened key → one column per run.
 *
 * This is the panel that answers "what did I actually vary?", and it is
 * deliberately the one thing that still works for a selection the overlays
 * refuse — comparing an angular run's config against a 1D run's is perfectly
 * meaningful even though their profiles cannot share an axis.
 */

import { useMemo, useState } from "react";

import { diffAcrossRuns, runLabel, type ComparisonRun } from "../lib/compare";

export function CompareConfig({ runs }: { runs: ComparisonRun[] }) {
  const rows = useMemo(() => diffAcrossRuns(runs), [runs]);
  const [changedOnly, setChangedOnly] = useState(true);
  const [filter, setFilter] = useState("");

  const varying = rows.filter((row) => row.varies).length;

  const visible = rows
    .filter((row) => (changedOnly ? row.varies : true))
    .filter((row) => (filter ? row.key.toLowerCase().includes(filter.toLowerCase()) : true));

  return (
    <section className="panel" aria-labelledby="compare-config-heading">
      <header className="panel__header">
        <h2 id="compare-config-heading">Config diff</h2>
        <div className="panel__controls">
          <label className="control control--inline">
            <input
              type="checkbox"
              checked={changedOnly}
              onChange={(event) => setChangedOnly(event.target.checked)}
            />
            <span>Changed keys only ({varying})</span>
          </label>
          <label className="control">
            <span className="visually-hidden">Filter keys</span>
            <input
              type="search"
              placeholder="filter keys"
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
            />
          </label>
        </div>
      </header>

      <div className="comparetable__scroll">
        <table className="comparetable comparetable--diff">
          <thead>
            <tr>
              <th scope="col">Key</th>
              {runs.map((run) => (
                <th key={run.runId} scope="col">
                  {runLabel(run)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.map((row) => (
              <tr key={row.key} className={row.varies ? "comparetable__row--varies" : undefined}>
                <th scope="row">{row.key}</th>
                {row.values.map((value, index) => (
                  <td
                    key={runs[index]?.runId ?? index}
                    // An absent key is different from a differing value, and the
                    // distinction matters when runs came from different decks.
                    className={value === undefined ? "comparetable__absent" : undefined}
                  >
                    {value ?? "absent"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {visible.length === 0 && (
        <p className="panel__status">
          {filter
            ? `No keys match "${filter}".`
            : changedOnly
              ? "These runs have identical configurations."
              : "No config parameters were logged."}
        </p>
      )}
    </section>
  );
}
