/** Filter controls. Every change writes straight to the URL, so the parent owns
 *  the state and this component is a pure function of it. */

import { RUN_STATUSES, type Experiment } from "../api/client";
import type { RunFilters } from "../lib/urlState";

/** tsadar's own progress tag values, distinct from MLflow's lifecycle status
 *  (see tsadar/inverse/fitter.py). A FAILED run stuck at `minimizing` is a
 *  useful thing to be able to find. */
const STAGES = ["preprocessing", "minimizing", "postprocessing", "plotting", "completed"] as const;

interface FiltersProps {
  filters: RunFilters;
  experiments: Experiment[];
  onChange: (next: RunFilters) => void;
  onClear: () => void;
  disabled?: boolean;
}

export function Filters({ filters, experiments, onChange, onClear, disabled }: FiltersProps) {
  const set = (key: keyof RunFilters, value: string) => {
    const next = { ...filters };
    if (value) next[key] = value;
    else delete next[key];
    onChange(next);
  };

  return (
    <form className="filters" onSubmit={(event) => event.preventDefault()}>
      <label className="filters__field">
        <span>Experiment</span>
        <select
          value={filters.experiment ?? ""}
          onChange={(event) => set("experiment", event.target.value)}
          disabled={disabled}
        >
          {/* "All Thomson" rather than "All": the list is already restricted to
              Thomson experiments, and a bare "All" would imply the whole server. */}
          <option value="">All Thomson</option>
          {experiments.map((experiment) => (
            <option key={experiment.experiment_id} value={experiment.name}>
              {experiment.name}
            </option>
          ))}
        </select>
      </label>

      <label className="filters__field">
        <span>Shot</span>
        <input
          type="text"
          inputMode="numeric"
          placeholder="101675"
          value={filters.shot ?? ""}
          onChange={(event) => set("shot", event.target.value)}
        />
      </label>

      <label className="filters__field">
        <span>Status</span>
        <select value={filters.status ?? ""} onChange={(event) => set("status", event.target.value)}>
          <option value="">Any</option>
          {RUN_STATUSES.map((status) => (
            <option key={status} value={status}>
              {status}
            </option>
          ))}
        </select>
      </label>

      <label className="filters__field">
        <span>Stage</span>
        <select value={filters.stage ?? ""} onChange={(event) => set("stage", event.target.value)}>
          <option value="">Any</option>
          {STAGES.map((stage) => (
            <option key={stage} value={stage}>
              {stage}
            </option>
          ))}
        </select>
      </label>

      <label className="filters__field">
        <span>User</span>
        <input
          type="text"
          value={filters.user ?? ""}
          onChange={(event) => set("user", event.target.value)}
        />
      </label>

      <label className="filters__field filters__field--grow">
        <span>Run name contains</span>
        <input
          type="search"
          placeholder="scan"
          value={filters.q ?? ""}
          onChange={(event) => set("q", event.target.value)}
        />
      </label>

      <button type="button" className="button" onClick={onClear}>
        Clear
      </button>
    </form>
  );
}
