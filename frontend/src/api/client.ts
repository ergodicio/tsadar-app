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
/** Which experiments the browser is scoped to. Null in `Health` when MLflow is
 *  unreachable and the scope could not be resolved at all. */
export type ThomsonScope = NonNullable<Health["thomson"]>;
export type ArtifactEntry = RunDetail["artifacts"][number];
export type MetricHistory = Json<"/api/runs/{run_id}/metrics/{key}", "get">;

export type DatasetAvailability = Json<"/api/runs/{run_id}/datasets", "get">;
export type SpectrumInfo = DatasetAvailability["spectra"][number];
export type Spectrogram = Json<"/api/runs/{run_id}/spectrogram", "get">;
export type Lineout = Json<"/api/runs/{run_id}/lineout", "get">;
export type Profiles = Json<"/api/runs/{run_id}/profiles", "get">;
export type ProfileSeries = Profiles["series"][number];

/** Spectrogram fields the backend can serve. `residual` is derived as data - fit;
 *  `irf` is deliberately absent because it is not in the netCDF datasets. */
export const SPECTROGRAM_FIELDS = ["data", "fit", "residual"] as const;
export type SpectrogramField = (typeof SPECTROGRAM_FIELDS)[number];

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

/** URL for an artifact, served through the API so the browser never touches S3.
 *
 *  Each path segment is encoded separately: the slashes are real path structure
 *  (`plots/fit_and_data.png`), so encoding the whole thing would break it. */
export function artifactUrl(runId: string, artifactPath: string): string {
  const segments = artifactPath.split("/").map(encodeURIComponent).join("/");
  return `/api/runs/${encodeURIComponent(runId)}/artifacts/${segments}`;
}

const runPath = (runId: string, suffix = "") => `/api/runs/${encodeURIComponent(runId)}${suffix}`;

export const api = {
  health: (signal?: AbortSignal) => request<Health>("/api/health", undefined, signal),

  experiments: (signal?: AbortSignal) =>
    request<Json<"/api/experiments", "get">>("/api/experiments", undefined, signal).then(
      (body) => body.experiments,
    ),

  runs: (query: RunQuery, signal?: AbortSignal) =>
    request<RunPage>("/api/runs", runQueryParams(query), signal),

  run: (runId: string, signal?: AbortSignal) =>
    request<RunDetail>(runPath(runId), undefined, signal),

  /** Metric keys contain spaces ("overall loss"), so they must be encoded. */
  metricHistory: (runId: string, key: string, signal?: AbortSignal) =>
    request<MetricHistory>(runPath(runId, `/metrics/${encodeURIComponent(key)}`), undefined, signal),

  datasets: (runId: string, signal?: AbortSignal) =>
    request<DatasetAvailability>(runPath(runId, "/datasets"), undefined, signal),

  spectrogram: (
    runId: string,
    options: { which: string; field: SpectrogramField; maxPx?: number },
    signal?: AbortSignal,
  ) => {
    const params = new URLSearchParams({ which: options.which, field: options.field });
    if (options.maxPx) params.set("max_px", String(options.maxPx));
    return request<Spectrogram>(runPath(runId, "/spectrogram"), params, signal);
  },

  lineout: (runId: string, options: { which: string; index: number }, signal?: AbortSignal) =>
    request<Lineout>(
      runPath(runId, "/lineout"),
      new URLSearchParams({ which: options.which, index: String(options.index) }),
      signal,
    ),

  profiles: (runId: string, signal?: AbortSignal) =>
    request<Profiles>(runPath(runId, "/profiles"), undefined, signal),

  /** Fetch an artifact as text, for the YAML config files. */
  artifactText: async (runId: string, artifactPath: string, signal?: AbortSignal) => {
    const response = await fetch(artifactUrl(runId, artifactPath), { signal });
    if (!response.ok) throw new ApiError(response.status, `could not fetch ${artifactPath}`);
    return response.text();
  },
};
