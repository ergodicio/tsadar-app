/** Routes. */

import { Link, Navigate, Route, Routes } from "react-router-dom";

import { Compare } from "./routes/Compare";
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
        <Route path="/compare" element={<Compare />} />
        <Route
          path="*"
          element={<Placeholder title="Not found" detail="That page does not exist." />}
        />
      </Routes>
    </main>
  );
}
