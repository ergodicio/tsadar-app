/**
 * Routes. `/runs/:runId` and `/compare` are placeholders here so the URL
 * contract the run browser links into is fixed now; the views themselves land
 * with #32 and #33.
 */

import { Navigate, Route, Routes, useParams } from "react-router-dom";

import { RunBrowser } from "./routes/RunBrowser";

function Placeholder({ title, detail }: { title: string; detail: string }) {
  return (
    <section className="state">
      <p className="state__title">{title}</p>
      <p className="state__detail">{detail}</p>
      <a className="button" href="/runs">
        Back to runs
      </a>
    </section>
  );
}

function RunDetailPlaceholder() {
  const { runId } = useParams();
  return (
    <Placeholder
      title={`Run ${runId}`}
      detail="The run detail view — spectrogram, lineout scrubber, profiles and config diff — arrives with issue #32."
    />
  );
}

export function App() {
  return (
    <main className="app">
      <Routes>
        <Route path="/" element={<Navigate to="/runs" replace />} />
        <Route path="/runs" element={<RunBrowser />} />
        <Route path="/runs/:runId" element={<RunDetailPlaceholder />} />
        <Route
          path="/compare"
          element={
            <Placeholder
              title="Compare runs"
              detail="The multi-run compare view arrives with issue #33."
            />
          }
        />
        <Route
          path="*"
          element={<Placeholder title="Not found" detail="That page does not exist." />}
        />
      </Routes>
    </main>
  );
}
