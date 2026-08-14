/**
 * Minimal declarations for `plotly.js-cartesian-dist-min`, which ships no types.
 *
 * Deliberately narrow rather than pulling in @types/plotly.js: only the handful
 * of calls `components/Plot.tsx` makes are declared, so the surface stays
 * obvious and the dependency stays small.
 *
 * The *cartesian* bundle, not the full one: it carries heatmap and scatter --
 * everything these panels draw -- at 1.4 MB instead of 4.7 MB.
 */
declare module "plotly.js-cartesian-dist-min" {
  export interface PlotlyClickPoint {
    x?: number | string;
    y?: number | string;
    pointIndex?: number;
    pointNumber?: number;
    curveNumber?: number;
  }

  export interface PlotlyClickEvent {
    points: PlotlyClickPoint[];
  }

  /** The div Plotly attaches its event emitter to. */
  export interface PlotlyHTMLElement extends HTMLDivElement {
    on(event: "plotly_click", handler: (event: PlotlyClickEvent) => void): void;
    removeAllListeners?(event: string): void;
  }

  export function react(
    element: HTMLElement,
    data: unknown[],
    layout?: unknown,
    config?: unknown,
  ): Promise<PlotlyHTMLElement>;

  export function purge(element: HTMLElement): void;
}
