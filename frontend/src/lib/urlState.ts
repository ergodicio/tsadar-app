/**
 * Filters live in the URL, not in component state.
 *
 * Every view being shareable by link is half the point of leaving Streamlit, so
 * the URL is the single source of truth for what the run browser is showing.
 *
 * The page cursor is deliberately *not* in the URL. MLflow paginates by opaque
 * token, and a pasted link carrying a stale cursor would either fail or silently
 * show a page from the middle of a different result set. Filters and sort are
 * shareable; scroll position is not.
 */

export interface RunFilters {
  experiment?: string;
  shot?: string;
  status?: string;
  stage?: string;
  user?: string;
  q?: string;
  sort?: string;
}

const FILTER_KEYS = ["experiment", "shot", "status", "stage", "user", "q", "sort"] as const;

export function filtersFromSearch(search: URLSearchParams): RunFilters {
  const filters: RunFilters = {};
  for (const key of FILTER_KEYS) {
    const value = search.get(key);
    if (value !== null && value.trim() !== "") filters[key] = value;
  }
  return filters;
}

export function searchFromFilters(filters: RunFilters): URLSearchParams {
  const search = new URLSearchParams();
  for (const key of FILTER_KEYS) {
    const value = filters[key];
    if (value !== undefined && value.trim() !== "") search.set(key, value);
  }
  return search;
}

/** True when no filter is active, so the empty state can tell "no runs exist"
 *  apart from "your filters matched nothing". */
export function hasActiveFilters(filters: RunFilters): boolean {
  return FILTER_KEYS.some((key) => key !== "sort" && filters[key]);
}

/** Toggle a sort key the way a table header does: first click ascending, second
 *  descending, and the `-` prefix is what the backend expects. */
export function toggleSort(current: string | undefined, field: string): string {
  if (current === field) return `-${field}`;
  if (current === `-${field}`) return field;
  return field;
}

export function sortDirection(current: string | undefined, field: string): "asc" | "desc" | null {
  if (current === field) return "asc";
  if (current === `-${field}`) return "desc";
  return null;
}
