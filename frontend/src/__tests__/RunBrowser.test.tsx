import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RunBrowser } from "../routes/RunBrowser";

interface RunOverrides {
  run_id?: string;
  run_name?: string;
  shot?: string | null;
  status?: string;
  stage?: string | null;
  spectype?: string | null;
  final_loss?: number | null;
  loss_key?: string | null;
  duration_s?: number | null;
}

function run(overrides: RunOverrides = {}) {
  return {
    run_id: "run-abc",
    run_name: "test-run",
    experiment_id: "1",
    experiment_name: "inverse-thomson-scattering",
    status: "FINISHED",
    stage: "completed",
    shot: "101675",
    spectype: "temporal",
    final_loss: 12.5,
    loss_key: "overall loss",
    start_time: 1_700_000_000_000,
    end_time: 1_700_000_123_000,
    duration_s: 123,
    user: "archis",
    ...overrides,
  };
}

let calls: string[] = [];

function stubApi(pages: Array<{ runs: unknown[]; next_page_token?: string | null }>) {
  let index = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      calls.push(url);
      if (url.startsWith("/api/experiments")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            experiments: [{ experiment_id: "1", name: "inverse-thomson-scattering", tags: {} }],
          }),
        };
      }
      const page = pages[Math.min(index, pages.length - 1)];
      index += 1;
      return {
        ok: true,
        status: 200,
        json: async () => ({ page_size: 50, next_page_token: null, ...page }),
      };
    }),
  );
}

function stubFailure(status: number, detail: string) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      calls.push(url);
      if (url.startsWith("/api/experiments")) {
        return { ok: true, status: 200, json: async () => ({ experiments: [] }) };
      }
      return { ok: false, status, json: async () => ({ detail }) };
    }),
  );
}

function renderBrowser(initialUrl = "/runs") {
  return render(
    <MemoryRouter initialEntries={[initialUrl]}>
      <RunBrowser />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  calls = [];
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("run browser table", () => {
  it("renders a row per run with the fields the issue asks for", async () => {
    stubApi([{ runs: [run()] }]);
    renderBrowser();

    await waitFor(() => expect(screen.getByText("test-run")).toBeInTheDocument());

    // Scoped to the row: "FINISHED" and "completed" are also filter dropdown
    // options, so an unscoped query matches more than one element.
    const row = within(screen.getByRole("row"));
    expect(row.getByText("101675")).toBeInTheDocument();
    expect(row.getByText("FINISHED")).toBeInTheDocument();
    expect(row.getByText("completed")).toBeInTheDocument();
    expect(row.getByText("archis")).toBeInTheDocument();
    expect(row.getByText("12.5000")).toBeInTheDocument();
    expect(row.getByText("2m 3s")).toBeInTheDocument();
    expect(row.getByText("inverse-thomson-scattering")).toBeInTheDocument();
  });

  it("shows a dash rather than a blank cell for a running run's duration", async () => {
    stubApi([{ runs: [run({ status: "RUNNING", duration_s: null, final_loss: null, loss_key: null })] }]);
    renderBrowser();
    await waitFor(() => expect(screen.getByText("RUNNING")).toBeInTheDocument());
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("marks a loss that came from a different metric", async () => {
    // Runs log different loss metrics, so the column is not uniformly comparable.
    stubApi([{ runs: [run({ final_loss: 3, loss_key: "min loss" })] }]);
    renderBrowser();
    await waitFor(() => expect(screen.getByText("3.0000")).toBeInTheDocument());
    expect(screen.getByTitle('from "min loss"')).toBeInTheDocument();
  });

  it("lists angular runs rather than hiding them, and labels them", async () => {
    // Out of scope for interactive views (#37), but hiding them would make the
    // table disagree with the MLflow UI.
    stubApi([{ runs: [run({ spectype: "angular_full" })] }]);
    renderBrowser();
    await waitFor(() =>
      expect(screen.getByText("angular_full (no interactive view)")).toBeInTheDocument(),
    );
  });

  it("does not offer a duration sort, which the backend cannot serve", async () => {
    stubApi([{ runs: [run()] }]);
    renderBrowser();
    await waitFor(() => expect(screen.getByText("test-run")).toBeInTheDocument());

    const header = screen.getByText("Duration");
    expect(header.tagName).not.toBe("BUTTON");
    expect(within(header).queryByRole("button")).toBeNull();
  });
});

describe("filters and sort drive the URL and the request", () => {
  it("reads filters from the URL on first load", async () => {
    stubApi([{ runs: [] }]);
    renderBrowser("/runs?shot=101675&status=FAILED");

    await waitFor(() => expect(calls.some((url) => url.startsWith("/api/runs"))).toBe(true));
    const runsCall = calls.find((url) => url.startsWith("/api/runs"))!;
    expect(runsCall).toContain("shot=101675");
    expect(runsCall).toContain("status=FAILED");
  });

  it("typing a shot number refetches with it", async () => {
    stubApi([{ runs: [run()] }]);
    renderBrowser();
    await waitFor(() => expect(screen.getByText("test-run")).toBeInTheDocument());

    await userEvent.type(screen.getByPlaceholderText("101675"), "9");
    await waitFor(() => expect(calls.some((url) => url.includes("shot=9"))).toBe(true));
  });

  it("clicking a sortable header requests that sort", async () => {
    stubApi([{ runs: [run()] }]);
    renderBrowser();
    await waitFor(() => expect(screen.getByText("test-run")).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: /Shot/ }));
    await waitFor(() => expect(calls.some((url) => url.includes("sort=shot"))).toBe(true));
  });

  it("clearing filters empties the query string", async () => {
    stubApi([{ runs: [run()] }]);
    renderBrowser("/runs?shot=101675");
    await waitFor(() => expect(screen.getByText("test-run")).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: "Clear" }));
    await waitFor(() => {
      const last = calls.filter((url) => url.startsWith("/api/runs")).at(-1)!;
      expect(last).not.toContain("shot=");
    });
  });
});

describe("pagination is by cursor", () => {
  it("loads more and appends, sending the token from the previous page", async () => {
    stubApi([
      { runs: [run({ run_id: "run-1", run_name: "first" })], next_page_token: "cursor-2" },
      { runs: [run({ run_id: "run-2", run_name: "second" })], next_page_token: null },
    ]);
    renderBrowser();
    await waitFor(() => expect(screen.getByText("first")).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: "Load more" }));

    await waitFor(() => expect(screen.getByText("second")).toBeInTheDocument());
    // Appended, not replaced.
    expect(screen.getByText("first")).toBeInTheDocument();
    expect(calls.some((url) => url.includes("page_token=cursor-2"))).toBe(true);
  });

  it("hides the load-more control on the last page", async () => {
    stubApi([{ runs: [run()], next_page_token: null }]);
    renderBrowser();
    await waitFor(() => expect(screen.getByText("test-run")).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Load more" })).toBeNull();
  });
});

describe("honest loading, empty and error states", () => {
  it("shows a loading state before the first page arrives", async () => {
    stubApi([{ runs: [] }]);
    renderBrowser();
    expect(screen.getByText("Loading runs…")).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText("Loading runs…")).toBeNull());
  });

  it("distinguishes no-runs-at-all from no-matches", async () => {
    stubApi([{ runs: [] }]);
    const { unmount } = renderBrowser();
    await waitFor(() => expect(screen.getByText("No runs yet")).toBeInTheDocument());
    unmount();

    stubApi([{ runs: [] }]);
    renderBrowser("/runs?shot=999999");
    await waitFor(() =>
      expect(screen.getByText("No runs match these filters")).toBeInTheDocument(),
    );
  });

  it("surfaces the backend's message and offers a retry", async () => {
    stubFailure(400, "cannot sort by nonsense");
    renderBrowser("/runs?sort=nonsense");

    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(screen.getByText("cannot sort by nonsense")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });

  it("a failed experiment list does not break the table", async () => {
    // The dropdown loses its options; the runs table is unaffected.
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.startsWith("/api/experiments")) {
          return { ok: false, status: 502, json: async () => ({ detail: "MLflow down" }) };
        }
        return {
          ok: true,
          status: 200,
          json: async () => ({ runs: [run()], page_size: 50, next_page_token: null }),
        };
      }),
    );
    renderBrowser();
    await waitFor(() => expect(screen.getByText("test-run")).toBeInTheDocument());
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

describe("selection for the compare view", () => {
  it("selecting rows builds a shareable /compare URL", async () => {
    stubApi([
      {
        runs: [run({ run_id: "run-1", run_name: "first" }), run({ run_id: "run-2", run_name: "second" })],
      },
    ]);
    renderBrowser();
    await waitFor(() => expect(screen.getByText("first")).toBeInTheDocument());

    await userEvent.click(screen.getByLabelText("Select first"));
    await userEvent.click(screen.getByLabelText("Select second"));

    expect(screen.getByText("2 runs selected")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Compare" })).toHaveAttribute(
      "href",
      "/compare?runs=run-1,run-2",
    );
  });
});
