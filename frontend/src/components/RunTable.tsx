/**
 * Virtualized run table.
 *
 * Only sortable columns get a header button. Duration is rendered as plain text
 * because MLflow cannot order by it -- it is computed from start/end timestamps --
 * so wiring a control there would just produce a 400.
 */

import { useVirtualizer } from "@tanstack/react-virtual";
import { useRef } from "react";

import type { RunSummary } from "../api/client";
import {
  formatDuration,
  formatLoss,
  formatTimestamp,
  isAngular,
  spectypeLabel,
} from "../lib/format";
import { sortDirection } from "../lib/urlState";

const ROW_HEIGHT = 40;

interface Column {
  key: string;
  label: string;
  /** Backend sort key, when the column is sortable. */
  sort?: string;
  width: string;
  align?: "right";
  /** Explains a caveat about the column or its sort. */
  hint?: string;
}

const COLUMNS: Column[] = [
  { key: "run_name", label: "Run", sort: "name", width: "minmax(10rem, 1.4fr)" },
  { key: "experiment_name", label: "Experiment", width: "minmax(9rem, 1fr)" },
  { key: "shot", label: "Shot", sort: "shot", width: "6rem" },
  { key: "spectype", label: "Type", width: "9rem" },
  { key: "status", label: "Status", sort: "status", width: "7rem" },
  { key: "stage", label: "Stage", width: "8rem" },
  {
    key: "final_loss",
    label: "Final loss",
    sort: "loss",
    width: "8rem",
    align: "right",
    // MLflow can only order by a named metric, so the sort always uses
    // "overall loss" -- a row marked with an asterisk is therefore sorted by a
    // metric it is not displaying. Say so rather than letting it mislead.
    hint: 'Sorts on the "overall loss" metric. Rows marked * report a different loss metric.',
  },
  { key: "duration_s", label: "Duration", width: "7rem", align: "right" },
  { key: "user", label: "User", width: "7rem" },
  { key: "start_time", label: "Created", sort: "created", width: "11rem" },
];

const TEMPLATE = COLUMNS.map((column) => column.width).join(" ");

interface RunTableProps {
  runs: RunSummary[];
  sort: string | undefined;
  onSort: (field: string) => void;
  onOpen: (runId: string) => void;
  selected: ReadonlySet<string>;
  onToggleSelected: (runId: string) => void;
}

export function RunTable({
  runs,
  sort,
  onSort,
  onOpen,
  selected,
  onToggleSelected,
}: RunTableProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({
    count: runs.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 12,
  });

  return (
    <div className="table">
      <div className="table__head" style={{ gridTemplateColumns: `2.5rem ${TEMPLATE}` }}>
        <span className="table__cell table__cell--head" aria-label="Select" />
        {COLUMNS.map((column) => {
          const direction = column.sort ? sortDirection(sort, column.sort) : null;
          return (
            <span
              key={column.key}
              className={`table__cell table__cell--head${column.align === "right" ? " table__cell--right" : ""}`}
              aria-sort={direction === "asc" ? "ascending" : direction === "desc" ? "descending" : undefined}
              title={column.hint}
            >
              {column.sort ? (
                <button type="button" className="table__sort" onClick={() => onSort(column.sort!)}>
                  {column.label}
                  {direction === "asc" && " ▲"}
                  {direction === "desc" && " ▼"}
                </button>
              ) : (
                column.label
              )}
            </span>
          );
        })}
      </div>

      <div className="table__body" ref={scrollRef} data-testid="run-table-scroll">
        <div style={{ height: virtualizer.getTotalSize(), position: "relative" }}>
          {virtualizer.getVirtualItems().map((item) => {
            const run = runs[item.index];
            if (!run) return null;
            return (
              <div
                key={run.run_id}
                className="table__row"
                style={{
                  gridTemplateColumns: `2.5rem ${TEMPLATE}`,
                  transform: `translateY(${item.start}px)`,
                  height: item.size,
                }}
                onClick={() => onOpen(run.run_id)}
                role="row"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter") onOpen(run.run_id);
                }}
              >
                <span className="table__cell" onClick={(event) => event.stopPropagation()}>
                  <input
                    type="checkbox"
                    aria-label={`Select ${run.run_name ?? run.run_id}`}
                    checked={selected.has(run.run_id)}
                    onChange={() => onToggleSelected(run.run_id)}
                  />
                </span>
                <span className="table__cell" title={run.run_id}>
                  {run.run_name ?? run.run_id}
                </span>
                <span className="table__cell">{run.experiment_name ?? "—"}</span>
                <span className="table__cell">{run.shot ?? "—"}</span>
                <span
                  className="table__cell"
                  title={
                    isAngular(run.spectype)
                      ? "Angular Thomson: listed, but interactive views are 1D only"
                      : undefined
                  }
                >
                  {spectypeLabel(run.spectype)}
                </span>
                <span className={`table__cell status status--${(run.status ?? "unknown").toLowerCase()}`}>
                  {run.status ?? "—"}
                </span>
                <span className="table__cell">{run.stage ?? "—"}</span>
                <span
                  className="table__cell table__cell--right"
                  // Runs can report different loss metrics, so name the source
                  // rather than implying the column is uniformly comparable.
                  title={run.loss_key ? `from "${run.loss_key}"` : undefined}
                >
                  {formatLoss(run.final_loss)}
                  {run.loss_key && run.loss_key !== "overall loss" && (
                    <sup className="table__note" aria-hidden="true">
                      *
                    </sup>
                  )}
                </span>
                <span className="table__cell table__cell--right">{formatDuration(run.duration_s)}</span>
                <span className="table__cell">{run.user ?? "—"}</span>
                <span className="table__cell">{formatTimestamp(run.start_time)}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
