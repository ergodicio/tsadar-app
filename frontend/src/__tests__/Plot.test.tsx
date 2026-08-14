/**
 * The Plot wrapper itself, with Plotly mocked at the module boundary.
 *
 * The panel tests mock `Plot` wholesale and assert on the traces it receives,
 * which leaves the wrapper's own lifecycle untested -- and that lifecycle is
 * where the interesting bug was: purging on every data change destroyed the
 * graph (losing zoom/pan) instead of letting `Plotly.react` diff it, and could
 * race an in-flight `react` into an empty div.
 */

import { act, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Plot } from "../components/Plot";

const graph = {
  on: vi.fn(),
  removeAllListeners: vi.fn(),
};

const plotly = {
  // Parameters are declared so `mock.calls` is typed and the assertions below can
  // index into a call's arguments.
  react: vi.fn(
    async (_element: HTMLElement, _data: unknown[], _layout?: unknown, _config?: unknown) => graph,
  ),
  purge: vi.fn((_element: HTMLElement) => {}),
};

vi.mock("plotly.js-cartesian-dist-min", () => plotly);

/** The wrapper awaits a dynamic import and then `react`, so a render is not
 *  finished until those microtasks have drained. A macrotask boundary drains all
 *  of them, however many the chain happens to be -- counting `Promise.resolve()`
 *  ticks by hand makes the assertions sensitive to that depth rather than to
 *  what the component actually did. */
async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

beforeEach(() => {
  plotly.react.mockClear();
  plotly.purge.mockClear();
  graph.on.mockClear();
  graph.removeAllListeners.mockClear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Plot", () => {
  it("updates in place without purging when the data changes", async () => {
    const first = [{ y: [1, 2, 3] }];
    const second = [{ y: [4, 5, 6] }];

    const { rerender } = render(<Plot data={first} ariaLabel="chart" />);
    await settle();
    expect(plotly.react).toHaveBeenCalledTimes(1);

    rerender(<Plot data={second} ariaLabel="chart" />);
    await settle();

    // Two `react` calls, no purge: the graph div is reused, so Plotly diffs the
    // traces and the user's zoom and pan survive.
    expect(plotly.react).toHaveBeenCalledTimes(2);
    expect(plotly.react.mock.calls[1]?.[1]).toBe(second);
    expect(plotly.purge).not.toHaveBeenCalled();
  });

  it("does not re-plot when only the click handler changes", async () => {
    const data = [{ y: [1, 2, 3] }];

    const { rerender } = render(<Plot data={data} onPointClick={() => {}} />);
    await settle();

    rerender(<Plot data={data} onPointClick={() => {}} />);
    await settle();

    expect(plotly.react).toHaveBeenCalledTimes(1);
  });

  it("purges once, on unmount", async () => {
    const { rerender, unmount } = render(<Plot data={[{ y: [1] }]} />);
    await settle();
    rerender(<Plot data={[{ y: [2] }]} />);
    await settle();
    expect(plotly.purge).not.toHaveBeenCalled();

    unmount();
    await settle();
    expect(plotly.purge).toHaveBeenCalledTimes(1);
  });

  it("replaces the click listener on each re-plot rather than stacking them", async () => {
    const { rerender } = render(<Plot data={[{ y: [1] }]} onPointClick={() => {}} />);
    await settle();
    rerender(<Plot data={[{ y: [2] }]} onPointClick={() => {}} />);
    await settle();

    // One listener attached per `react`, each preceded by a removal, so a click
    // fires the handler once no matter how many times the data has changed.
    expect(graph.removeAllListeners).toHaveBeenCalledTimes(2);
    expect(graph.on).toHaveBeenCalledTimes(2);
  });

  it("reports the clicked point index", async () => {
    const clicked = vi.fn();
    render(<Plot data={[{ y: [1, 2, 3] }]} onPointClick={clicked} />);
    await settle();

    const [event, handler] = graph.on.mock.calls[0] as [string, (payload: unknown) => void];
    expect(event).toBe("plotly_click");

    handler({ points: [{ pointIndex: 2 }] });
    expect(clicked).toHaveBeenCalledWith(2);

    // Plotly uses `pointNumber` for some trace types and `pointIndex` for
    // others; heatmap clicks are how the spectrogram selects a lineout.
    handler({ points: [{ pointNumber: 5 }] });
    expect(clicked).toHaveBeenCalledWith(5);

    // A click on empty canvas carries no point at all.
    handler({ points: [] });
    expect(clicked).toHaveBeenCalledTimes(2);
  });

  it("calls the latest click handler after a re-render", async () => {
    const stale = vi.fn();
    const fresh = vi.fn();

    const data = [{ y: [1] }];
    const { rerender } = render(<Plot data={data} onPointClick={stale} />);
    await settle();
    rerender(<Plot data={data} onPointClick={fresh} />);
    await settle();

    const handler = graph.on.mock.calls[0]?.[1] as (payload: unknown) => void;
    handler({ points: [{ pointIndex: 0 }] });

    expect(stale).not.toHaveBeenCalled();
    expect(fresh).toHaveBeenCalledWith(0);
  });
});
