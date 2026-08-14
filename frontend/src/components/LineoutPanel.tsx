/**
 * Measured vs fitted spectrum at one lineout, with a scrubber.
 *
 * Linked to the spectrogram: clicking a column there moves this, and moving the
 * slider moves the marker there. This replaces the pre-rendered `lineouts/`,
 * `best/` and `worst/` PNGs with something you can step through.
 */

import { useEffect, useState } from "react";

import { ApiError, api, type Lineout, type SpectrumInfo } from "../api/client";
import { axisLabel } from "../lib/format";
import { Plot } from "./Plot";

interface LineoutPanelProps {
  runId: string;
  spectrum: SpectrumInfo;
  which: string;
  index: number;
  onIndexChange: (index: number) => void;
}

export function LineoutPanel({ runId, spectrum, which, index, onIndexChange }: LineoutPanelProps) {
  const [lineout, setLineout] = useState<Lineout | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const maxIndex = Math.max(0, spectrum.lineout_count - 1);
  const clamped = Math.min(Math.max(index, 0), maxIndex);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    api
      .lineout(runId, { which, index: clamped }, controller.signal)
      .then(setLineout)
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setLineout(null);
        setError(cause instanceof ApiError ? cause.message : "Could not load the lineout.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [runId, which, clamped]);

  const traces = lineout
    ? [
        { type: "scatter", mode: "lines", name: "Data", x: lineout.wavelength, y: lineout.data },
        { type: "scatter", mode: "lines", name: "Fit", x: lineout.wavelength, y: lineout.fit },
        {
          type: "scatter",
          mode: "lines",
          name: "Residual",
          x: lineout.wavelength,
          y: lineout.residual,
          yaxis: "y2",
          line: { width: 1, dash: "dot" },
        },
      ]
    : [];

  const layout = lineout
    ? {
        xaxis: { title: axisLabel(lineout.y_label) },
        yaxis: { title: "Amplitude (arb.)", domain: [0.32, 1] },
        yaxis2: { title: "Residual", domain: [0, 0.24] },
        showlegend: true,
        legend: { orientation: "h", y: 1.12 },
        margin: { t: 30, r: 20, b: 45, l: 60 },
      }
    : {};

  return (
    <section className="panel" aria-labelledby="lineout-heading">
      <header className="panel__header">
        <h2 id="lineout-heading">Lineout</h2>
        <span className="panel__meta">
          {lineout
            ? `${axisLabel(lineout.x_label)} = ${lineout.x_value.toPrecision(4)}`
            : `index ${clamped}`}
        </span>
      </header>

      <label className="scrubber">
        <span className="scrubber__label">
          Lineout {clamped + 1} of {spectrum.lineout_count}
        </span>
        <input
          type="range"
          min={0}
          max={maxIndex}
          value={clamped}
          aria-label="Lineout index"
          onChange={(event) => onIndexChange(Number(event.target.value))}
        />
      </label>

      {loading && <p className="panel__status">Loading lineout…</p>}
      {error && (
        <p className="panel__status panel__status--error" role="alert">
          {error}
        </p>
      )}

      {lineout && !error && (
        <>
          <Plot data={traces} layout={layout} height={360} ariaLabel="Measured versus fitted spectrum" />
          {lineout.components_unavailable && (
            // Stated rather than silently omitted: someone comparing this with
            // the old PNGs will notice the components are missing and deserve to
            // know it is a data limitation, not a rendering bug.
            <p className="panel__note">
              IRF and noise components are not available for interactive plots — they are not stored
              in the netCDF datasets, only in the pre-rendered lineout images below.
            </p>
          )}
        </>
      )}
    </section>
  );
}
