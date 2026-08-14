/** Display helpers for run table cells. */

/** Render the spatial axis label sensibly.
 *
 * tsadar stores it as the literal LaTeX fragment `Radius (\mum)` (see
 * `load_ts_data.py`), which would otherwise show up verbatim in the UI.
 */
export function axisLabel(label: string | null | undefined): string {
  if (!label) return "";
  return label.replace(/\\mu m|\\mum/g, "µm").replace(/\\/g, "");
}

export function formatTimestamp(millis: number | null | undefined): string {
  if (millis === null || millis === undefined) return "—";
  return new Date(millis).toISOString().replace("T", " ").slice(0, 19);
}

export function formatDuration(seconds: number | null | undefined): string {
  // Null while a run is still going: start/end arithmetic has no answer yet.
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

export function formatLoss(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (value !== 0 && (Math.abs(value) < 1e-3 || Math.abs(value) >= 1e5)) return value.toExponential(3);
  return value.toFixed(4);
}

/** Spectrum types the browser supports interactive views for. Angular runs are
 *  out of scope (#37) but stay listed, so they are labelled rather than hidden. */
const ONE_D_SPECTYPES = new Set(["temporal", "imaging"]);

export function isAngular(spectype: string | null | undefined): boolean {
  return typeof spectype === "string" && spectype.toLowerCase().startsWith("angular");
}

export function spectypeLabel(spectype: string | null | undefined): string {
  if (!spectype) return "—";
  if (isAngular(spectype)) return `${spectype} (no interactive view)`;
  if (ONE_D_SPECTYPES.has(spectype)) return spectype;
  return spectype;
}
