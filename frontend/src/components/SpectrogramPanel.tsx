/**
 * Interactive spectrogram with a data / fit / residual toggle.
 *
 * Clicking a column selects that lineout, which is what drives the linked
 * spectrum panel -- the interactive replacement for the pre-rendered
 * `lineouts/`, `best/` and `worst/` PNGs.
 *
 * There is no `irf` toggle: IRF and noise components are not in the netCDF
 * datasets at all, only baked into those PNGs. The backend reports that through
 * `unavailable_fields` and the gallery still has the images.
 */

import { useEffect, useMemo, useState } from "react";

import {
  ApiError,
  SPECTROGRAM_FIELDS,
  api,
  type SpectrogramField,
  type Spectrogram,
  type SpectrumInfo,
} from "../api/client";
import { axisLabel } from "../lib/format";
import { Plot } from "./Plot";

/** Say what the server did to the array, so the plot never silently implies
 *  full resolution. Factors arrive as a plain map, hence the defaults. */
export function describeDownsampling(spectrogram: Spectrogram): string {
  const [lineouts, wavelengths] = spectrogram.full_shape;
  const shape = `${lineouts} × ${wavelengths}`;

  if (spectrogram.downsample_method === "none") return `Full resolution, ${shape}.`;

  const wavelengthFactor = spectrogram.downsample_factors.wavelength ?? 1;
  const lineoutFactor = spectrogram.downsample_factors.x ?? 1;

  const parts = [`Block-averaged ${wavelengthFactor}× in wavelength`];
  if (lineoutFactor > 1) {
    parts.push(`and ${lineoutFactor}× along ${axisLabel(spectrogram.x_label)}`);
  }
  return `${parts.join(" ")} from ${shape}.`;
}

interface SpectrogramPanelProps {
  runId: string;
  spectra: SpectrumInfo[];
  which: string;
  onWhichChange: (which: string) => void;
  lineoutIndex: number;
  onLineoutChange: (index: number) => void;
}

export function SpectrogramPanel({
  runId,
  spectra,
  which,
  onWhichChange,
  lineoutIndex,
  onLineoutChange,
}: SpectrogramPanelProps) {
  const [field, setField] = useState<SpectrogramField>("data");
  const [spectrogram, setSpectrogram] = useState<Spectrogram | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    api
      .spectrogram(runId, { which, field }, controller.signal)
      .then(setSpectrogram)
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setSpectrogram(null);
        setError(cause instanceof ApiError ? cause.message : "Could not load the spectrogram.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [runId, which, field]);

  // Memoized because `Plot` re-plots when these change by identity. Without it,
  // dragging the lineout scrubber -- which re-renders this component on every
  // pointer move -- would hand Plotly a brand new heatmap trace each time and
  // redraw the whole array. The traces do not depend on `lineoutIndex` at all;
  // only the marker in the layout does.
  const traces = useMemo(
    () =>
      spectrogram
        ? [
            {
              type: "heatmap",
              x: spectrogram.x,
              y: spectrogram.y,
              z: spectrogram.values,
              // Residual is signed, so it reads far better on a diverging scale
              // centred at zero than on a sequential one.
              colorscale: field === "residual" ? "RdBu" : "Viridis",
              zmid: field === "residual" ? 0 : undefined,
              colorbar: { title: field },
            },
          ]
        : [],
    [spectrogram, field],
  );

  const layout = useMemo(
    () =>
      spectrogram
        ? {
            xaxis: { title: axisLabel(spectrogram.x_label) },
            yaxis: { title: axisLabel(spectrogram.y_label) },
            margin: { t: 10, r: 10, b: 45, l: 60 },
            shapes: [
              // The selected lineout, drawn as a vertical rule so the scrubber
              // and the heatmap always agree about where you are.
              {
                type: "line",
                x0: spectrogram.x[lineoutIndex] ?? null,
                x1: spectrogram.x[lineoutIndex] ?? null,
                yref: "paper",
                y0: 0,
                y1: 1,
                line: { color: "#ff7f0e", width: 1.5, dash: "dot" },
              },
            ],
          }
        : {},
    [spectrogram, lineoutIndex],
  );

  return (
    <section className="panel" aria-labelledby="spectrogram-heading">
      <header className="panel__header">
        <h2 id="spectrogram-heading">Spectrogram</h2>

        <div className="panel__controls">
          {spectra.length > 1 && (
            <label className="control">
              <span>Spectrum</span>
              <select value={which} onChange={(event) => onWhichChange(event.target.value)}>
                {spectra.map((spectrum) => (
                  <option key={spectrum.which} value={spectrum.which}>
                    {spectrum.which === "ele" ? "electron (EPW)" : "ion (IAW)"}
                  </option>
                ))}
              </select>
            </label>
          )}

          <div className="segmented" role="group" aria-label="Field">
            {SPECTROGRAM_FIELDS.map((candidate) => (
              <button
                key={candidate}
                type="button"
                className={`segmented__option${candidate === field ? " segmented__option--active" : ""}`}
                aria-pressed={candidate === field}
                onClick={() => setField(candidate)}
              >
                {candidate}
              </button>
            ))}
          </div>
        </div>
      </header>

      {loading && <p className="panel__status">Loading spectrogram…</p>}
      {error && (
        <p className="panel__status panel__status--error" role="alert">
          {error}
        </p>
      )}

      {spectrogram && !error && (
        <>
          <Plot
            data={traces}
            layout={layout}
            height={340}
            ariaLabel={`${field} spectrogram`}
            onPointClick={(pointIndex) => onLineoutChange(pointIndex)}
          />
          <p className="panel__note">
            {describeDownsampling(spectrogram)}
            {field === "residual" && " Residual is derived as data − fit."}
          </p>
        </>
      )}
    </section>
  );
}
