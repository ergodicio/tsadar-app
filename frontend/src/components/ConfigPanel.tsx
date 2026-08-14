/**
 * The run's config, plus a diff when the run recorded one.
 *
 * The merged tree always comes from the run detail response (rebuilt from logged
 * params) and is always shown. The diff is extra: NERSC-queued runs log
 * `defaults.yaml` and `inputs.yaml` separately, so those can be diffed to answer
 * "what did this run actually change?". App-queued runs log a single merged
 * `config.yaml`, so there is nothing to diff and the panel says so rather than
 * showing an empty table.
 */

import { useEffect, useMemo, useState } from "react";
import { parse as parseYaml } from "yaml";

import { api, type RunDetail } from "../api/client";
import { CONFIG_ARTIFACTS, configSources, diffConfigs, displayValue, type ConfigDiffRow } from "../lib/config";

interface ConfigPanelProps {
  runId: string;
  run: RunDetail;
}

function ConfigTree({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return <span className="config__value">{displayValue(value)}</span>;
  }

  return (
    <ul className="config__tree" style={{ marginLeft: depth === 0 ? 0 : "0.9rem" }}>
      {Object.entries(value as Record<string, unknown>).map(([key, child]) => {
        const isBranch = child !== null && typeof child === "object" && !Array.isArray(child);
        return (
          <li key={key} className="config__node">
            {isBranch ? (
              <details open={depth < 1}>
                <summary className="config__key">{key}</summary>
                <ConfigTree value={child} depth={depth + 1} />
              </details>
            ) : (
              <>
                <span className="config__key">{key}</span>
                <ConfigTree value={child} depth={depth + 1} />
              </>
            )}
          </li>
        );
      })}
    </ul>
  );
}

export function ConfigPanel({ runId, run }: ConfigPanelProps) {
  const sources = useMemo(
    () => configSources(run.artifacts.filter((entry) => !entry.is_dir).map((entry) => entry.path)),
    [run.artifacts],
  );

  const [diff, setDiff] = useState<ConfigDiffRow[] | null>(null);
  const [diffError, setDiffError] = useState<string | null>(null);
  const [changedOnly, setChangedOnly] = useState(true);
  const [view, setView] = useState<"tree" | "diff">("tree");

  useEffect(() => {
    if (!sources.diffable) return;
    const controller = new AbortController();

    Promise.all([
      api.artifactText(runId, CONFIG_ARTIFACTS.defaults, controller.signal),
      api.artifactText(runId, CONFIG_ARTIFACTS.inputs, controller.signal),
    ])
      .then(([defaultsText, inputsText]) =>
        setDiff(diffConfigs(parseYaml(defaultsText), parseYaml(inputsText))),
      )
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setDiffError(cause instanceof Error ? cause.message : "Could not build the config diff.");
      });

    return () => controller.abort();
  }, [runId, sources.diffable]);

  const rows = diff?.filter((row) => (changedOnly ? row.status !== "same" : true)) ?? [];
  const changedCount = diff?.filter((row) => row.status !== "same").length ?? 0;

  return (
    <section className="panel" aria-labelledby="config-heading">
      <header className="panel__header">
        <h2 id="config-heading">Configuration</h2>
        <div className="panel__controls">
          <div className="segmented" role="group" aria-label="Config view">
            <button
              type="button"
              className={`segmented__option${view === "tree" ? " segmented__option--active" : ""}`}
              aria-pressed={view === "tree"}
              onClick={() => setView("tree")}
            >
              tree
            </button>
            <button
              type="button"
              className={`segmented__option${view === "diff" ? " segmented__option--active" : ""}`}
              aria-pressed={view === "diff"}
              onClick={() => setView("diff")}
              disabled={!sources.diffable}
              title={
                sources.diffable
                  ? undefined
                  : "This run logged a single merged config.yaml, so there is nothing to diff against."
              }
            >
              diff
            </button>
          </div>
        </div>
      </header>

      {run.config_unflatten_error && (
        <p className="panel__status panel__status--error" role="alert">
          The config tree could not be rebuilt from this run's logged parameters (
          {run.config_unflatten_error}). The flat parameter list is still available on the MLflow run
          page.
        </p>
      )}

      {view === "tree" && <ConfigTree value={run.config} />}

      {view === "diff" && (
        <>
          {diffError && (
            <p className="panel__status panel__status--error" role="alert">
              {diffError}
            </p>
          )}
          {!diff && !diffError && <p className="panel__status">Loading config diff…</p>}
          {diff && (
            <>
              <label className="control control--inline">
                <input
                  type="checkbox"
                  checked={changedOnly}
                  onChange={(event) => setChangedOnly(event.target.checked)}
                />
                <span>Changed keys only ({changedCount})</span>
              </label>

              <table className="difftable">
                <thead>
                  <tr>
                    <th scope="col">Key</th>
                    <th scope="col">defaults.yaml</th>
                    <th scope="col">inputs.yaml</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.key} className={`difftable__row difftable__row--${row.status}`}>
                      <td>{row.key}</td>
                      <td>{displayValue(row.base)}</td>
                      <td>{displayValue(row.override)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {rows.length === 0 && (
                <p className="panel__status">
                  {changedOnly ? "This run changed nothing from the defaults." : "No config keys."}
                </p>
              )}
            </>
          )}
        </>
      )}

      {!sources.diffable && view === "tree" && (
        <p className="panel__note">
          {sources.hasMerged
            ? "This run logged a single merged config.yaml, so there are no defaults to diff against."
            : "This run logged no config YAML artifacts; the tree above is rebuilt from its logged parameters."}
        </p>
      )}
    </section>
  );
}
