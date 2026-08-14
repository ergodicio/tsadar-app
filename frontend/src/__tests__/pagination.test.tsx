/**
 * Regression tests for the two `useRuns` pagination bugs found in review on #41.
 *
 * Both were written to fail against the original version first, so they pin real
 * behavior rather than describing the fix after the fact.
 */

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { RunBrowser } from "../routes/RunBrowser";

function makeRun(id: string, name: string) {
  return {
    run_id: id,
    run_name: name,
    experiment_id: "1",
    experiment_name: "inverse-thomson-scattering",
    status: "FINISHED",
    stage: "completed",
    shot: "101675",
    spectype: "temporal",
    final_loss: 1,
    loss_key: "overall loss",
    start_time: 1_700_000_000_000,
    end_time: 1_700_000_001_000,
    duration_s: 1,
    user: "archis",
  };
}

afterEach(() => vi.unstubAllGlobals());

function renderBrowser(url = "/runs") {
  return render(
    <MemoryRouter initialEntries={[url]}>
      <RunBrowser />
    </MemoryRouter>,
  );
}

describe("a failed Load more keeps the pages already loaded", () => {
  it("shows an inline error without discarding the table", async () => {
    // The cursor is deliberately not in the URL, so there is no way to resume:
    // dropping the table means re-scrolling from page 1. Someone 400 runs deep
    // should not lose all of it to one flaky response.
    let call = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.startsWith("/api/experiments")) {
          return { ok: true, status: 200, json: async () => ({ experiments: [] }) };
        }
        call += 1;
        if (call === 1) {
          return {
            ok: true,
            status: 200,
            json: async () => ({
              runs: [makeRun("run-1", "first"), makeRun("run-2", "second")],
              page_size: 50,
              next_page_token: "cursor-2",
            }),
          };
        }
        return { ok: false, status: 502, json: async () => ({ detail: "MLflow timed out" }) };
      }),
    );

    renderBrowser();
    await waitFor(() => expect(screen.getByText("first")).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: "Load more" }));

    await waitFor(() => expect(screen.getByText(/MLflow timed out/)).toBeInTheDocument());
    // The rows already fetched must survive.
    expect(screen.getByText("first")).toBeInTheDocument();
    expect(screen.getByText("second")).toBeInTheDocument();
  });
});

describe("a stale page cannot leak into a filtered table", () => {
  it("drops a Load more response that resolves after the filters changed", async () => {
    // The leaked row is real data from a different query, so it looks like a run
    // that does not match your filter rather than like a bug.
    // A holder rather than a bare `let`: TypeScript narrows a variable only
    // assigned inside a callback to `never`.
    const gate: { release: (() => void) | null } = { release: null };

    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.startsWith("/api/experiments")) {
          return { ok: true, status: 200, json: async () => ({ experiments: [] }) };
        }
        if (url.includes("page_token=")) {
          // Hold page 2 open until after the filter change.
          await new Promise<void>((resolve) => {
            gate.release = resolve;
          });
          return {
            ok: true,
            status: 200,
            json: async () => ({
              runs: [makeRun("run-stale", "stale-unfiltered-run")],
              page_size: 50,
              next_page_token: null,
            }),
          };
        }
        const filtered = url.includes("shot=");
        return {
          ok: true,
          status: 200,
          json: async () => ({
            runs: [makeRun(filtered ? "run-f" : "run-1", filtered ? "filtered-run" : "unfiltered-run")],
            page_size: 50,
            next_page_token: filtered ? null : "cursor-2",
          }),
        };
      }),
    );

    renderBrowser();
    await waitFor(() => expect(screen.getByText("unfiltered-run")).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: "Load more" }));
    // Change the filters while page 2 is still in flight.
    await userEvent.type(screen.getByPlaceholderText("101675"), "9");
    await waitFor(() => expect(screen.getByText("filtered-run")).toBeInTheDocument());

    gate.release?.();

    await waitFor(() => expect(screen.getByText("filtered-run")).toBeInTheDocument());
    expect(screen.queryByText("stale-unfiltered-run")).toBeNull();
  });
});
