# Thomson analysis browser — frontend

A Vite + React + TypeScript SPA over the read layer in `tsadar_browser/`
([#29]). This covers the scaffold and the run browser ([#31]); the run detail view
is [#32] and the compare view is [#33].

## Running it

The API and the SPA run as separate dev servers; Vite proxies `/api` to uvicorn
(override with `VITE_API_TARGET`). In the deployed image ([#34]) the SPA is served
same-origin and no proxy applies.

```bash
# terminal 1 — the API (see ../docs/browser.md for the env it needs)
uvicorn tsadar_browser.app:app --reload --port 8000

# terminal 2 — the SPA
cd frontend
npm install
npm run dev
```

```bash
npm run typecheck   # tsc --noEmit
npm test            # vitest
npm run build       # typecheck + production bundle
npm run gen         # regenerate the API client from the backend
```

## The generated API client

`src/api/schema.d.ts` and `src/api/openapi.json` are **generated, not written**.
`npm run gen` dumps the OpenAPI document straight from the FastAPI app
(`scripts/dump_openapi.py`) and runs `openapi-typescript` over it. Both files are
committed so a frontend-only checkout typechecks without a Python environment.

CI runs `npm run check:client`, which regenerates both and fails on any diff — so
a backend response-model change cannot silently rot the client. If that job fails,
run `npm run gen` and commit the result.

`src/api/client.ts` is the hand-written part: a thin typed `fetch` wrapper whose
response types are derived from the generated `paths`.

## Things that surprised us, encoded here

Each of these is a real constraint from the backend rather than a style choice.

**Pagination is by cursor, not page number.** MLflow's `search_runs` paginates by
opaque token, so `useRuns` accumulates pages behind a "Load more" control instead
of jumping to an offset. There is no way to ask for "page 4".

**Filters and sort live in the URL; the cursor does not.** Every view being
shareable as a link is half the point of leaving Streamlit. A cursor in the URL
would be worse than useless — a pasted link carrying a stale token would fail or
silently show a page from the middle of a different result set.

**Duration is not sortable.** It is computed from start/end timestamps and MLflow
cannot order by it, so that column deliberately has no sort control; wiring one
would produce a 400.

**The loss column is not uniformly comparable.** tsadar logs several loss metrics
whose names contain spaces (`overall loss`, `min loss`, `epoch loss`) and there is
no metric called plain `loss`. The API reports which one each value came from via
`loss_key`, and the table surfaces it — with a marker when it isn't the usual
`overall loss` — rather than implying every row measures the same quantity.
`sort=loss` orders on `overall loss` specifically, since MLflow can only order by
a named metric.

**Status and stage are different things.** `status` is MLflow's lifecycle
(`RUNNING`/`FINISHED`/`FAILED`/`KILLED`); `stage` is tsadar's own progress tag
(`preprocessing` → … → `completed`). Both are filterable, because a `FAILED` run
whose stage is stuck at `minimizing` is exactly the diagnostic worth finding.

**Angular runs are listed, not hidden.** Interactive views are 1D-only ([#37]),
but angular runs live in the same experiments and hiding them would make this
table disagree with the MLflow UI. They are labelled instead. The `spectype` field
is a **hint** — it is logged before the fit runs and can disagree with reality — so
it is displayed but never used to filter.

**`Radius (\mum)` is a literal LaTeX fragment** in tsadar's stored axis labels, so
`lib/format.ts` renders it as `µm` rather than showing the backslash.

## Testing notes

Tests run in jsdom, which performs no layout: every element measures 0×0 and
there is no `ResizeObserver`. The virtualized table would render zero rows for
reasons unrelated to the component, so `src/test-setup.ts` shims a fixed viewport.
That shim is confined to the test setup — measurement is the browser's job in
production.

[#29]: https://github.com/ergodicio/tsadar-app/issues/29
[#31]: https://github.com/ergodicio/tsadar-app/issues/31
[#32]: https://github.com/ergodicio/tsadar-app/issues/32
[#33]: https://github.com/ergodicio/tsadar-app/issues/33
[#34]: https://github.com/ergodicio/tsadar-app/issues/34
[#37]: https://github.com/ergodicio/tsadar-app/issues/37
