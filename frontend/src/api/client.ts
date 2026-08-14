/**
 * Typed client for the browser API.
 *
 * Response types come from `schema.d.ts`, which is generated from the FastAPI
 * app's OpenAPI document (`npm run gen`). CI regenerates both and fails on a
 * diff, so these types cannot drift from the backend without someone noticing.
 */

import type { paths } from "./schema";

type Json<P extends keyof paths, M extends "get"> = paths[P][M] extends {
  responses: { 200: { content: { "application/json": infer R } } };
}
  ? R
  : never;

export type Experiment = Json<"/api/experiments", "get">["experiments"][number];
export type RunPage = Json<"/api/runs", "get">;
export type RunSummary = RunPage["runs"][number];
export type RunDetail = Json<"/api/runs/{run_id}", "get">;
export type Health = Json<"/api/health", "get">;

/** Sort keys the backend accepts. Duration is deliberately absent: it is computed
 *  from timestamps and MLflow cannot order by it, so offering it would 400. */
export const SORTABLE_FIELDS = ["created", "name", "status", "shot", "loss"] as const;
export type SortField = (typeof SORTABLE_FIELDS)[number];

/** MLflow lifecycle statuses, distinct from tsadar's own progress `stage`. */
export const RUN_STATUSES = ["RUNNING", "FINISHED", "FAILED", "KILLED"] as const;

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly reason?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Pull a useful message out of an error body, which may be a plain string or
 *  the dataset endpoints' `{reason, detail}` object. */
function describeError(status: number, body: unknown): ApiError {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return new ApiError(status, detail);
    if (detail && typeof detail === "object") {
      const { reason, detail: inner } = detail as { reason?: string; detail?: string };
      return new ApiError(status, inner ?? `request failed (${status})`, reason);
    }
  }
  return new ApiError(status, `request failed (${status})`);
}

async function request<T>(path: string, params?: URLSearchParams, signal?: AbortSignal): Promise<T> {
  const query = params?.toString();
  const response = await fetch(query ? `${path}?${query}` : path, {
    signal,
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      // A proxy or gateway error may not return JSON at all.
    }
    throw describeError(response.status, body);
  }
  return (await response.json()) as T;
}

export interface RunQuery {
  experiment?: string;
  shot?: string;
  status?: string;
  stage?: string;
  user?: string;
  q?: string;
  sort?: string;
  pageSize?: number;
  pageToken?: string;
}

/** Build the query string for `GET /api/runs`.
 *
 *  Exported for testing: the mapping from UI state to backend parameters is
 *  where filter bugs hide. */
export function runQueryParams(query: RunQuery): URLSearchParams {
  const params = new URLSearchParams();
  const simple: Array<[string, string | undefined]> = [
    ["experiment", query.experiment],
    ["shot", query.shot],
    ["status", query.status],
    ["stage", query.stage],
    ["user", query.user],
    ["q", query.q],
    ["sort", query.sort],
    ["page_token", query.pageToken],
  ];
  for (const [key, value] of simple) {
    if (value) params.set(key, value);
  }
  if (query.pageSize) params.set("page_size", String(query.pageSize));
  return params;
}

export const api = {
  health: (signal?: AbortSignal) => request<Health>("/api/health", undefined, signal),

  experiments: (signal?: AbortSignal) =>
    request<Json<"/api/experiments", "get">>("/api/experiments", undefined, signal).then(
      (body) => body.experiments,
    ),

  runs: (query: RunQuery, signal?: AbortSignal) =>
    request<RunPage>("/api/runs", runQueryParams(query), signal),

  run: (runId: string, signal?: AbortSignal) =>
    request<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`, undefined, signal),
};
