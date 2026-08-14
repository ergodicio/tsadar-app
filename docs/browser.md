# Thomson analysis browser — backend

A read-only FastAPI layer over the MLflow tracking server. MLflow stays the
source of truth: this service owns no database and never writes to it. Runs are
immutable once terminal, so fetched artifacts are cached on disk indefinitely
and evicted only to stay under a size cap.

Tracking issue: [#37]. This document covers [#29]; the netCDF slicing endpoints
that render interactive plots are [#30].

## Running it locally

```bash
pip install -r requirements-browser-dev.txt

export MLFLOW_TRACKING_URI=https://continuum.ergodic.io/experiments
export MLFLOW_TRACKING_USERNAME=...   # Basic auth, backed by Cognito
export MLFLOW_TRACKING_PASSWORD=...

uvicorn tsadar_browser.app:app --reload --port 8000
```

Artifacts live in `s3://public-ergodic-continuum` and are read with ambient AWS
credentials locally, or the task role in deployment. Interactive docs are at
`/api/docs`; the schema the frontend client is generated from is at
`/api/openapi.json`.

Run the tests with `python -m pytest tests/browser`. They use a fake MLflow
client, so no credentials and no network are needed.

## Configuration

All configuration is environment variables (a `.env` file also works).

| Variable | Default | Purpose |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | `https://continuum.ergodic.io/experiments` | Tracking server |
| `MLFLOW_TRACKING_USERNAME` | — | Basic auth user |
| `MLFLOW_TRACKING_PASSWORD` | — | Basic auth password |
| `CACHE_DIR` | `$TMPDIR/tsadar-browser-cache` | Artifact cache root |
| `CACHE_MAX_GB` | `10` | Cache size cap; LRU eviction above it |
| `CORS_ORIGINS` | `localhost:5173` | Vite dev origins; comma-separated, empty disables CORS |
| `MAX_PAGE_SIZE` | `200` | Ceiling on `page_size` |
| `MLFLOW_HTTP_REQUEST_TIMEOUT` | `15` | Per-request timeout, seconds |
| `MLFLOW_HTTP_REQUEST_MAX_RETRIES` | `2` | Retries before giving up |
| `MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR` | `1` | Retry backoff |
| `HEALTH_PROBE_TTL_S` | `5` | How long a reachability probe is trusted |

The MLflow variables keep their canonical names because the mlflow client reads
them straight from the environment; values loaded from a `.env` file are exported
back into `os.environ` at startup so the client sees them too.

The HTTP timeout defaults are **deliberately much lower than MLflow's own**
(120 seconds with 7 retries and a backoff factor of 2, which can block for
minutes on a single unreachable call). Behind a load balancer that is
indistinguishable from a hung task, so the browser bounds it to fail fast and
report `degraded`. Setting any of these three variables explicitly wins over the
defaults above.

## Endpoints

| Endpoint | Notes |
| --- | --- |
| `GET /api/health` | Liveness plus MLflow reachability and cache stats |
| `GET /api/experiments` | Active experiments |
| `GET /api/runs` | Filter, sort, paginate; see below |
| `GET /api/runs/{run_id}` | Config tree, tags, metric summaries, artifact listing, manifest |
| `GET /api/runs/{run_id}/metrics/{key}` | Full metric history for loss curves |
| `GET /api/runs/{run_id}/artifacts/{path}` | Streaming passthrough with content type |

### Notes on the contract

**`/api/health` answers 200 even when MLflow is down.** It is the ALB target-group
check; a degraded browser that can explain itself beats a task the load balancer
keeps recycling. The `status` field (`ok` / `degraded`) carries the real verdict.

**Pagination is by cursor, not page number.** MLflow's `search_runs` paginates by
opaque token, so `/api/runs` accepts `page_token` and returns `next_page_token`
(null on the last page). Offset paging would mean walking every preceding page on
each request. This is a deliberate deviation from the `page=` parameter sketched
in [#29].

**Filters are translated to MLflow filter strings**, currently against `params.*`:

| Query param | Becomes |
| --- | --- |
| `shot` | `params."data.shotnum" = '...'` |
| `status` | `attributes.status = '...'` (MLflow lifecycle) |
| `stage` | `tags."status" = '...'` (tsadar's own progress tag) |
| `user` | `tags."mlflow.user" = '...'` |
| `q` | `attributes.run_name LIKE '%...%'` |

`status` and `stage` are genuinely different things and both are exposed. MLflow's
lifecycle status is `RUNNING`/`FINISHED`/`FAILED`/`KILLED`; tsadar separately sets
a `status` **tag** that tracks fit progress (`preprocessing` → `minimizing` →
`postprocessing` → `plotting` → `completed`, see `tsadar/inverse/fitter.py`). A
run can be `FINISHED` at the MLflow level while its tag says `completed`, and a
crashed run can be `FAILED` with the tag stuck at `minimizing` — which is exactly
the diagnostic a physicist wants. These move to canonical tags once
[ergodicio/tsadar#115] lands; only the translation layer changes.

MLflow's filter grammar has no escape sequence for a quote inside a string
literal, so filter values containing `'`, `"` or `\` are rejected with a 400
rather than being interpolated into a query that would mean something else.
Sort keys are allowlisted for the same reason.

**Sorting** accepts `created`, `name`, `status`, `shot`, `loss` (prefix with `-`
for descending). Duration is *not* sortable: it is computed from start/end
timestamps and MLflow cannot order by it.

**`final_loss` reports which metric it came from.** tsadar logs several loss
metrics whose names contain spaces — `overall loss`, `min loss`, `epoch loss`.
The first present wins and `loss_key` names it, so the table never silently
compares two different quantities. Because keys contain spaces, metric history
paths arrive URL-encoded (`/metrics/overall%20loss`).

**The config tree is reconstructed by round-tripping the flattening.**
`tsadar.utils.misc.log_mlflow` flattens the config with a dot reducer before
logging; `/api/runs/{run_id}` unflattens it back into `config`, re-parsing scalars
as YAML so numbers, booleans and lists come back as themselves rather than
strings (`"None"` is special-cased, since it is not a YAML null token). The raw
flat params stay available as `config_flat`. If a run logged colliding dotted keys
the tree is ill-defined, so `config` is left empty and
`config_unflatten_error` explains why instead of the API guessing.

**Artifact paths are untrusted.** A `..` segment is rejected outright; the
resolved path is also checked to be inside the cache root. Absolute-looking paths
are relativized rather than rejected — harmless once the leading slash is gone.

## Caching

Keyed `run_id/path`, which is also the on-disk layout, so the cache is
inspectable with `ls`. Artifacts of terminal runs (`FINISHED`/`FAILED`/`KILLED`)
are cached indefinitely; artifacts of a `RUNNING` run are re-fetched every time
because they may still be rewritten. Downloads land in a scratch directory inside
the cache root and are moved into place atomically, so an interrupted fetch never
leaves a truncated entry. Concurrent requests for the same cold artifact share
one download. Eviction is least-recently-used by mtime (bumped on each hit, since
`atime` is unreliable on the `relatime` mounts containers usually get).

## What this does not do

No netCDF reading — that is [#30]. No writes of any kind, so no job submission;
the Streamlit app's submit path targets retired infrastructure and is not
resurrected here ([#35]).

[#29]: https://github.com/ergodicio/tsadar-app/issues/29
[#30]: https://github.com/ergodicio/tsadar-app/issues/30
[#35]: https://github.com/ergodicio/tsadar-app/issues/35
[#37]: https://github.com/ergodicio/tsadar-app/issues/37
[ergodicio/tsadar#115]: https://github.com/ergodicio/tsadar/issues/115
