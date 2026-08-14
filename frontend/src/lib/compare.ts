/**
 * Multi-run comparison logic.
 *
 * The hard part of comparing runs is not plotting — it is knowing when a
 * comparison would be meaningless. Two rules, both from #37's scope decision:
 *
 * - **Angular runs cannot be overlaid on 1D runs.** Their x axis is scattering
 *   angle, so putting one on the same axis as `Time (ps)` produces a chart that
 *   is not a degraded comparison but a meaningless one. They are excluded with a
 *   reason rather than reconciled.
 * - **Mixing spatial and temporal 1D runs is the same problem, milder.** Both are
 *   1D, but `Radius (μm)` and `Time (ps)` are still not the same axis. Those are
 *   plotted but flagged, since a deliberate comparison is imaginable and the
 *   axis label alone would not warn you.
 *
 * The config diff is the exception: flattened key → per-run values doesn't care
 * about axis semantics, so it stays useful even for a selection this module
 * refuses to overlay.
 */

import type { DatasetAvailability, Profiles, RunDetail } from "../api/client";

export interface ComparisonRun {
  runId: string;
  detail: RunDetail;
  availability: DatasetAvailability;
  profiles: Profiles | null;
  /** Why this run cannot take part in the overlays, when it cannot. */
  excluded: string | null;
}

/** Decide whether a run can be overlaid, and say why not when it cannot.
 *
 *  `profilesError` distinguishes "this run logged no profiles" from "the profiles
 *  request failed". Both leave `profiles` null and both exclude the run, but only
 *  one of them is worth retrying, and telling a user their run has no fitted
 *  parameters when the server merely 500'd sends them looking in the wrong place. */
export function exclusionReason(
  availability: DatasetAvailability,
  profiles: Profiles | null,
  profilesError: string | null = null,
): string | null {
  if (availability.kind === "angular") {
    return "Angular run: its x axis is scattering angle, so it cannot share an axis with 1D runs. Its config still appears in the diff.";
  }
  if (!availability.supported) {
    return availability.message ?? "No readable datasets for this run.";
  }
  if (profilesError) {
    return `Could not load this run's parameter profiles: ${profilesError}`;
  }
  if (!profiles) {
    return "No fitted-parameter profiles logged for this run.";
  }
  return null;
}

/** Runs that can actually be overlaid. */
export function comparableRuns(runs: ComparisonRun[]): ComparisonRun[] {
  return runs.filter((run) => run.excluded === null && run.profiles !== null);
}

/** Distinct lineout-axis labels among the comparable runs.
 *
 *  More than one means the overlay is putting different quantities on one axis. */
export function axisLabels(runs: ComparisonRun[]): string[] {
  return [...new Set(comparableRuns(runs).map((run) => run.profiles!.x_label))];
}

export function mixedAxisWarning(runs: ComparisonRun[]): string | null {
  const labels = axisLabels(runs);
  if (labels.length <= 1) return null;
  return `These runs do not share a lineout axis (${labels.join(", ")}). The overlay puts different quantities on one axis — compare with care.`;
}

/** Parameter names present across the comparable runs, in a stable order.
 *
 *  Runs legitimately differ here: an ele-only run has no ion parameters, and
 *  which parameters were active varies per run. The union is taken so a
 *  parameter missing from one run leaves a gap rather than dropping the panel. */
export function sharedSeriesNames(runs: ComparisonRun[]): string[] {
  const names = new Set<string>();
  for (const run of comparableRuns(runs)) {
    for (const series of run.profiles!.series) names.add(series.name);
  }
  return [...names].sort();
}

/** Which runs actually have a given series, so a panel can say "3 of 4 runs". */
export function runsWithSeries(runs: ComparisonRun[], name: string): ComparisonRun[] {
  return comparableRuns(runs).filter((run) =>
    run.profiles!.series.some((series) => series.name === name),
  );
}

// -- config diff across N runs -------------------------------------------------

export interface MultiConfigRow {
  key: string;
  /** One entry per run, in the order the runs were given. Undefined where a run
   *  does not have the key at all. */
  values: Array<string | undefined>;
  /** True when the runs do not all agree. */
  varies: boolean;
}

/** Diff the flattened configs of N runs.
 *
 *  Uses `config_flat` -- the raw params as MLflow stores them -- rather than the
 *  reconstructed tree, so the comparison is between what was actually logged and
 *  cannot be skewed by a failed unflattening on one run. */
export function diffAcrossRuns(runs: ComparisonRun[]): MultiConfigRow[] {
  const flats = runs.map((run) => run.detail.config_flat ?? {});
  const keys = [...new Set(flats.flatMap((flat) => Object.keys(flat)))].sort();

  return keys.map((key) => {
    const values = flats.map((flat) => flat[key]);
    const present = values.filter((value): value is string => value !== undefined);
    const varies = present.length !== values.length || new Set(present).size > 1;
    return { key, values, varies };
  });
}

/** How many runs a single comparison will load.
 *
 *  Chosen from what the overlay can actually express, not from a load estimate:
 *  Plotly's default colorway has ten entries, so an eleventh trace reuses the
 *  first colour and two runs become indistinguishable in the legend. It also
 *  bounds the fan-out -- every run costs up to three requests, all issued
 *  concurrently -- so a hand-edited URL cannot turn one page load into hundreds
 *  of them. */
export const MAX_COMPARE_RUNS = 10;

export interface ParsedRunIds {
  /** The ids to load: distinct, in URL order, at most `MAX_COMPARE_RUNS`. */
  ids: string[];
  /** How many distinct ids the URL asked for, which may exceed `ids.length`. */
  requested: number;
}

/** Parse the `runs` query parameter, dropping blanks and duplicates.
 *
 *  Order is preserved so trace colours stay stable for a given URL, and the cap
 *  is applied here rather than at the call site so no caller can forget it. The
 *  requested count is returned alongside so the page can say what it dropped
 *  instead of silently truncating. */
export function parseRunIds(raw: string | null): ParsedRunIds {
  if (!raw) return { ids: [], requested: 0 };
  const seen = new Set<string>();
  for (const candidate of raw.split(",")) {
    const id = candidate.trim();
    if (id) seen.add(id);
  }
  const ids = [...seen];
  return { ids: ids.slice(0, MAX_COMPARE_RUNS), requested: ids.length };
}

/** A short label for a run in legends and table headers. */
export function runLabel(run: ComparisonRun): string {
  const { detail } = run;
  const shot = detail.shot ? `shot ${detail.shot}` : null;
  return detail.run_name ?? shot ?? detail.run_id.slice(0, 8);
}
