/**
 * The only place Plotly is touched.
 *
 * Two reasons this is a wrapper rather than direct calls in each panel:
 *
 * - Plotly is large even as the cartesian-only bundle, so it is loaded with a
 *   dynamic import. The run browser never mounts a chart and should not pay for
 *   one; this keeps it a separate chunk fetched on the detail page.
 * - jsdom has no canvas or WebGL, so Plotly cannot render under test. Panels
 *   mock this component and assert on the traces they pass in, which is the part
 *   worth testing -- that the right arrays reach the chart.
 */

import { useEffect, useRef } from "react";

export interface PlotProps {
  data: unknown[];
  layout?: Record<string, unknown>;
  /** Fired with the clicked point's index along the trace. */
  onPointClick?: (pointIndex: number) => void;
  height?: number;
  ariaLabel?: string;
}

const CONFIG = { displaylogo: false, responsive: true };

export function Plot({ data, layout, onPointClick, height = 320, ariaLabel }: PlotProps) {
  const container = useRef<HTMLDivElement>(null);
  // Held in a ref so changing the handler does not force a re-plot.
  const clickHandler = useRef(onPointClick);
  clickHandler.current = onPointClick;

  // Updates and teardown are separate effects on purpose. Purging on every
  // `data`/`layout` change threw the graph away and rebuilt it from scratch --
  // losing the user's zoom and pan, and forcing a full redraw where
  // `Plotly.react` would have diffed. Worse, `Plotly.react` is async: a purge
  // firing while one was in flight could leave an empty div behind.
  //
  // Declaration order matters. React runs cleanups in the order the effects
  // were declared, so this one's cleanup marks the in-flight update stale
  // *before* the teardown effect below purges, and nothing can render into an
  // already-purged element.
  useEffect(() => {
    const element = container.current;
    if (!element) return;

    let stale = false;

    void (async () => {
      const Plotly = await import("plotly.js-cartesian-dist-min");
      if (stale) return;

      const plot = await Plotly.react(element, data, { autosize: true, height, ...layout }, CONFIG);
      if (stale) return;

      // `react` reuses the graph div, so a listener from the previous render is
      // still attached and would fire twice.
      plot.removeAllListeners?.("plotly_click");
      plot.on("plotly_click", (event) => {
        const point = event.points[0];
        const index = point?.pointIndex ?? point?.pointNumber;
        if (index !== undefined) clickHandler.current?.(index);
      });
    })();

    return () => {
      stale = true;
    };
  }, [data, layout, height]);

  useEffect(() => {
    const element = container.current;
    if (!element) return;

    return () => {
      void import("plotly.js-cartesian-dist-min").then((Plotly) => Plotly.purge(element));
    };
  }, []);

  return <div ref={container} role="img" aria-label={ariaLabel} className="plot" />;
}
