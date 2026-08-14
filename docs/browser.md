# Thomson analysis browser — backend

A read-only FastAPI layer over the MLflow tracking server. MLflow stays the
source of truth: this service owns no database and never writes to it. Runs are
immutable once terminal, so fetched artifacts are cached on disk indefinitely
and evicted only to stay under a size cap.

Tracking issue: [#37]. This document covers [#29]; the netCDF slicing endpoints
that render interactive plots are [#30].

## Scope: 1D Thomson, not angular

The browser targets **time-resolved (`Time (ps)`) and space-resolved / imaging
(`Radius (μm)`) Thomson scattering** — the routine OMEGA analysis workflow.
**Angularly-resolved Thomson (`spectype: angular` / `angular_full`) is out of
scope**: it needs its own diagnostics and is a research tool rather than a
production analysis path. See [#37].

This constrains *supported views*, not visibility. Angular runs share the same
experiments, so they stay listed and keep their PNG gallery; what they must never
do is render through a 1D code path whose axes mean something else.

`RunSummary.spectype` reports the logged spectrum type, but it is **a hint, not
ground truth**: `misc.log_mlflow(config)` runs in `runner.run` *before*
`fitter.fit`, and `loadData` overwrites `spectype` from the data file during
`prepare` — so a deck saying `temporal` run against angular data logs
`temporal`. Ground truth is the artifact shape:

| Signal | 1D | Angular |
| --- | --- | --- |
| Dataset | `binary/ele_fit_and_data.nc`, `binary/ion_fit_and_data.nc` | `binary/fit_and_data.nc` |
| Written by | `plotters.plot_ts_data` | `plotters.plot_data_angular` |
| x coordinate | `Time (ps)` / `Radius (\mum)` | `Scattering angle (degrees)` |

Both dataset kinds hold the same two variables (`fit`, `data`) with the same
number of dimensions, so falling back to `fit_and_data.nc` when the `ele_`/`ion_`
files are missing would silently serve angle-vs-wavelength data to a UI labelling
its x-axis as time. [#30] treats it as recognized-but-unsupported instead.

Note the spatial label is stored literally as `Radius (\mum)`, a raw LaTeX
fragment from `load_ts_data.py`; it needs handling before display.

For the same reason, no `spectype` **filter** is offered on `/api/runs`: filtering
on a value that can disagree with reality would quietly drop runs. The field is
returned for display, and type is confirmed from artifacts on the detail path.

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
| `STATIC_DIR` | unset | Built SPA to serve alongside `/api`; unset in development |
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
| `GET /api/runs/{run_id}/datasets` | What interactive views this run supports, and why not when it doesn't |
| `GET /api/runs/{run_id}/spectrogram` | 2D array, block-averaged to a pixel budget |
| `GET /api/runs/{run_id}/lineout` | Measured vs fitted spectrum at one lineout |
| `GET /api/runs/{run_id}/profiles` | Fitted parameters vs lineout, with sigmas where available |

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

In `q`, `%` and `_` remain live SQL `LIKE` wildcards rather than literals.
MLflow's filter grammar has no `ESCAPE` clause, and the backslash an escape would
need is already rejected as un-quotable, so this is documented behavior rather
than half-escaped: `q=scan%01` matches "scan" followed by anything then "01".

**Sorting** accepts `created`, `name`, `status`, `shot`, `loss` (prefix with `-`
for descending). Duration is *not* sortable: it is computed from start/end
timestamps and MLflow cannot order by it.

`sort=loss` orders on the **`overall loss` metric specifically**. MLflow can only
order by a named metric, so sorting cannot follow the same per-run fallback that
`final_loss` uses: a run that logged only `min loss` sorts as though it had no
loss, even though the table shows a value for it. `loss_key` is what makes that
visible, so a client rendering a loss column should surface it rather than
presenting the column as uniformly comparable.

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

The size cap is enforced after **every** fetch, including non-cacheable ones —
those still write into the cache root, so skipping eviction there would let the
directory grow past `CACHE_MAX_GB` until some later cacheable fetch cleaned up.

The artifact a fetch is about to return is exempt from that eviction pass. Once
Starlette has the file open, unlinking it is harmless on POSIX (the descriptor
keeps the data alive), but there is a window between the fetch returning and the
response opening the file where deleting it would produce a spurious 404. One
consequence worth knowing: a single artifact larger than the whole cap leaves the
cache over its limit rather than evicting the file being served. That is logged at
`WARNING` rather than silently tolerated.

Per-key download locks are reference counted and dropped once no thread is
waiting, so the lock table does not grow for the lifetime of the process.

## Interactive plots from the netCDF datasets

These endpoints ([#30]) are what make the browser better than a PNG gallery: they
render from the arrays the fit actually produced.

### Probe first

`GET /api/runs/{id}/datasets` answers for **every** run — angular, pre-contract,
or fully supported — rather than erroring, so the run detail view ([#32]) can pick
a layout in one call:

```json
{
  "kind": "one_d",
  "supported": true,
  "spectra": [{"which": "ele", "x_label": "Time (ps)", "lineout_count": 60,
               "wavelength_count": 1024, "fields": ["data", "fit", "residual"]}],
  "profiles_available": true,
  "sigmas_available": false,
  "unavailable_fields": {"irf": "..."}
}
```

When a run can't be served, `supported` is false and `reason` is a code, not
prose: `angular_not_supported`, `dataset_missing`, `dataset_unreadable`,
`unexpected_schema`, `field_unavailable`, `index_out_of_range`. The distinction
matters — showing "no data found" for an angular run reads as a bug when the
truth is that the view is deliberately out of scope.

The data endpoints carry the same codes in their **error** bodies, with **409**
for recognized-but-unsupported (angular) and **404** for genuinely absent. Note
the shape: FastAPI wraps anything an `HTTPException` carries under `detail`, so
the reason is one level down and the declared schema mirrors that nesting.

```json
{ "detail": { "reason": "angular_not_supported", "detail": "This is an angularly-resolved run. …" } }
```

Read it as `err.detail.reason`, not `err.reason`.

### What the datasets actually contain

Less than [#30] assumed, so two things are worth stating plainly:

- **Residual is derived**, as `data - fit`. It isn't stored.
- **IRF and noise components are not in the netCDFs at all.** `plotters.py`
  writes `{"fit": ..., "data": ...}` and nothing else; the components exist only
  baked into the pre-rendered `lineouts/`, `best/` and `worst/` PNGs. They are
  reported through `unavailable_fields` / `components_unavailable` rather than
  invented. Making them genuinely available is a `plotters.py` change and
  probably belongs with [ergodicio/tsadar#116].

`sigmas.nc` lives at the artifact **root**, not under `binary/` — `calc_sigmas`
is off by default, so absence is normal rather than an error. In
`learned_parameters.csv`, `to_csv` writes the DataFrame index as a leading
unnamed column, and the lineout axis column is matched **by name** against the
dataset's own axis label. The "first column with parenthesized units" heuristic
is only a fallback: it is positional, and safe today purely because
`get_final_params` inserts the axis ahead of the parameters, so a fitted
parameter that ever acquired a unit in its name would otherwise be mistaken for
the axis. Among parenthesized fallback candidates a monotonic one wins, since a
lineout axis always is and a parameter generally is not.

Classification lists only `binary/` rather than walking the whole artifact tree.
A real run has `binary/`, `csv/`, `plots/`, `lineouts/`, `best/` and `worst/`, and
each directory is an MLflow round trip; the lineout scrubber steps
interactively, so a full walk would be paid before every step. `/datasets` still
takes the full listing, because it reports on profiles and sigmas too — but it is
called once per page load rather than once per interaction.

### Downsampling

`max_px` is a pixel budget; the array is **block-averaged** (not decimated) down
to it, so a narrow spectral feature is attenuated rather than skipped entirely.

**Wavelength is reduced first and the lineout axis is spared wherever possible.**
Each lineout is a separate fit with its own parameters, and it's the axis the
scrubber in [#32] steps through, so trading lineouts for bytes costs far more than
spectral resolution does. At a realistic 60 × 1024 against a 2000-pixel budget
that means 60 × 32 — every lineout intact — rather than 30 × 34. The response
reports `downsample_factors`, `downsample_method`, `full_shape` and
`returned_shape` so the UI can say what it's showing.

`values` is shaped `(len(y), len(x))` — row-major by wavelength — so it drops
straight into a Plotly heatmap `z`, which is indexed `[y][x]`.

The default `max_px` is **20,000**, deliberately below a realistic
full-resolution spectrogram (60 × 1024 = 61,440), because [#30] asks that full
resolution never be shipped by default. A heatmap panel is a few hundred pixels
tall, so 1024 spectral points are already oversampled for display; clients that
genuinely want everything can raise `max_px`.

### Non-finite values and precision

JSON has no NaN or Infinity, and Python's `json.dumps` emits a bare `NaN` that
makes browser `JSON.parse` throw. Gaps in a fit are therefore serialized as
`null` throughout these endpoints.

Values are trimmed to **6 significant digits**. Full float repr roughly doubles
the payload (a 60 × 256 spectrogram goes from ~290 KB to ~130 KB) for precision no
plot can show. This is display data; anything needing the exact stored values
should fetch the `.nc` through the artifact passthrough instead.

One consequence: `residual` is differenced at full precision and *then* rounded,
so it is not bit-identical to `data - fit` computed from the rounded arrays. The
gap is the double-rounding floor — around 9e-6 absolute at these magnitudes.

## Deployment

The image ([#34]) is one container serving both halves: a node stage builds the
SPA, a python stage runs uvicorn and serves that bundle from `STATIC_DIR`. Because
it is one origin, the deployed app needs no proxy and no CORS — `CORS_ORIGINS` is
empty in the image and only exists for the Vite dev server.

```bash
docker build -f docker/browser/Dockerfile -t thomson-browser:dev .
docker run -p 8000:8000 -e MLFLOW_TRACKING_USERNAME=... -e MLFLOW_TRACKING_PASSWORD=... thomson-browser:dev
```

**Serving the SPA is not just a static mount.** Two cases a plain mount gets
wrong, both covered by tests:

- `/runs/abc123` is a route the router resolves in the browser, not a file, so it
  must return `index.html`. Otherwise every shared deep link 404s — much of the
  point of leaving Streamlit.
- An unknown `/api` path must stay a **JSON 404**, not the app shell. Serving the
  shell there would turn a typo'd endpoint into a 200 of HTML, so the client would
  fail while parsing rather than on the status code.

`index.html` is served `no-cache` while `assets/` is content-hashed and cached
hard, so a new image's assets are actually fetched rather than shadowed by a
cached shell.

**Tags are immutable.** The tag comes from the repository-root `VERSION` file
(`thomson-browser-v0.1.0`) and the workflow **refuses to push a tag that already
exists** in ECR. continuum-infra pins image tags, so a floating tag would let a
pinned deployment change underneath itself — the failure mode `continuum.yaml`'s
header comment documents. Bump `VERSION` in the same PR as the change you want
deployed; see `docker/browser/VERSION.md`.

The older `deploy.yaml` (Streamlit, runner, tesseract) is left in place, since
those images are being kept. It does *not* have this discipline — its tags are
hardcoded and overwritten on every push to main — and that is annotated there
rather than changed, because fixing it means deciding what happens to the
currently-deployed pins.

CI boots the container against a **deliberately unreachable** MLflow and asserts
`/api/health` answers 200 with `status: degraded` — not merely that it returns
200, which a hardcoded success would also do. That is the state a task starts in
when credentials are missing, and it must answer rather than hang.

One worker per container, deliberately: the artifact cache's LRU bookkeeping and
download deduplication are in-process locks, so multiple workers would each keep
their own view of it. Scale with tasks.

## What this does not do

No writes of any kind, so no job submission; the Streamlit app's submit path
targets retired infrastructure and is not resurrected here ([#35]).

Angular Thomson interactive views are out of scope by design (see Scope above) —
not unimplemented, but deliberately refused.

[#32]: https://github.com/ergodicio/tsadar-app/issues/32
[ergodicio/tsadar#116]: https://github.com/ergodicio/tsadar/issues/116

[#29]: https://github.com/ergodicio/tsadar-app/issues/29
[#30]: https://github.com/ergodicio/tsadar-app/issues/30
[#35]: https://github.com/ergodicio/tsadar-app/issues/35
[#37]: https://github.com/ergodicio/tsadar-app/issues/37
[ergodicio/tsadar#115]: https://github.com/ergodicio/tsadar/issues/115
