/**
 * Routes. `/compare` stays a placeholder so the URL contract the run browser
 * links into is fixed; the view itself lands with #33.
 */

import { Link, Navigate, Route, Routes } from "react-router-dom";

import { RunBrowser } from "./routes/RunBrowser";
import { RunDetail } from "./routes/RunDetail";

function Placeholder({ title, detail }: { title: string; detail: string }) {
  return (
    <section className="state">
      <p className="state__title">{title}</p>
      <p className="state__detail">{detail}</p>
      <Link className="button" to="/runs">
        Back to runs
      </Link>
    </section>
  );
}

export function App() {
  return (
    <main className="app">
      <Routes>
        <Route path="/" element={<Navigate to="/runs" replace />} />
        <Route path="/runs" element={<RunBrowser />} />
        <Route path="/runs/:runId" element={<RunDetail />} />
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
