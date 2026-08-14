/**
 * Multi-run compare view (issue #33).
 *
 * The behaviour worth pinning is the refusals: which runs get overlaid, which get
 * excluded-with-a-reason, and the fact that the config diff keeps working for a
 * selection the overlays reject.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Compare } from "../routes/Compare";
import { MAX_COMPARE_RUNS } from "../lib/compare";
import { angularAvailability, availability, metricHistory, profiles, runDetail } from "./fixtures";

const plotCalls: Array<{ data: unknown[]; layout?: Record<string, unknown>; ariaLabel?: string }> = [];

vi.mock("../components/Plot", () => ({
  Plot: (props: { data: unknown[]; layout?: Record<string, unknown>; ariaLabel?: string }) => {
    plotCalls.push(props);
    return <div data-testid="plot" data-aria={props.ariaLabel} />;
  },
}));

interface RunStub {
  detail?: Record<string, unknown>;
  probe?: Record<string, unknown>;
  profiles?: Record<string, unknown> | null;
  fail?: boolean;
}

let requested: string[] = [];

function stubRuns(byId: Record<string, RunStub>) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      requested.push(url);
      const runId = url.match(/\/api\/runs\/([^/?]+)/)?.[1] ?? "";
      const stub = byId[runId];
      const ok = (body: unknown) =>
        ({ ok: true, status: 200, json: async () => body }) as unknown as Response;
      const notFound = () =>
        ({ ok: false, status: 404, json: async () => ({ detail: "run not found" }) }) as unknown as Response;

      if (!stub || stub.fail) return notFound();
      if (url.includes("/datasets")) return ok(stub.probe ?? availability());
      if (url.includes("/profiles")) {
        if (stub.profiles === null) return notFound();
        return ok(stub.profiles ?? profiles());
      }
      if (url.includes("/metrics/")) {
        const key = decodeURIComponent(url.split("/metrics/")[1] ?? "");
        return ok(metricHistory(key));
      }
      return ok(stub.detail ?? runDetail({ run_id: runId, run_name: runId }));
    }),
  );
}

function renderCompare(url: string) {
  return render(
    <MemoryRouter initialEntries={[url]}>
      <Routes>
        <Route path="/compare" element={<Compare />} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  plotCalls.length = 0;
  requested = [];
});

afterEach(() => vi.unstubAllGlobals());

describe("entry", () => {
  it("prompts when no runs are given", async () => {
    stubRuns({});
    renderCompare("/compare");
    expect(screen.getByText("No runs selected")).toBeInTheDocument();
  });

  it("loads every run in the URL", async () => {
    stubRuns({ a: {}, b: {} });
    renderCompare("/compare?runs=a,b");

    await waitFor(() => expect(screen.getByText("Comparing 2 runs")).toBeInTheDocument());
    expect(requested.some((url) => url.includes("/api/runs/a"))).toBe(true);
    expect(requested.some((url) => url.includes("/api/runs/b"))).toBe(true);
  });

  it("deduplicates repeated ids from a hand-edited URL", async () => {
    stubRuns({ a: {} });
    renderCompare("/compare?runs=a,a,a");
    await waitFor(() => expect(screen.getByText("Comparing 1 runs")).toBeInTheDocument());
  });
});

describe("summary table", () => {
  it("shows one column per run with its key facts", async () => {
    stubRuns({ a: {}, b: {} });
    renderCompare("/compare?runs=a,b");

    await waitFor(() => expect(screen.getByRole("link", { name: "a" })).toBeInTheDocument());

    // There are two tables on the page (summary and config diff), so anchor on
    // the summary's own heading rather than querying by role alone.
    const summary = within(screen.getByRole("region", { name: "Runs" }));
    const headers = summary.getAllByRole("columnheader").map((cell) => cell.textContent);
    expect(headers[0]).toBe("Field");
    expect(headers[1]).toContain("a");
    expect(headers[2]).toContain("b");

    const shotRow = summary.getByRole("rowheader", { name: "Shot" }).closest("tr")!;
    expect(within(shotRow).getAllByRole("cell").map((cell) => cell.textContent)).toEqual([
      "101675",
      "101675",
    ]);
  });

  it("names each run's loss metric rather than implying comparability", async () => {
    stubRuns({
      a: { detail: runDetail({ run_id: "a", run_name: "a", loss_key: "overall loss" }) },
      b: { detail: runDetail({ run_id: "b", run_name: "b", loss_key: "min loss", final_loss: 3 }) },
    });
    renderCompare("/compare?runs=a,b");

    await waitFor(() => expect(screen.getByText("(overall loss)")).toBeInTheDocument());
    expect(screen.getByText("(min loss)")).toBeInTheDocument();
    // And warns that the row mixes metrics.
    expect(screen.getByText(/different loss metrics/)).toBeInTheDocument();
  });

  it("removing a run rewrites the URL rather than hiding it locally", async () => {
    stubRuns({ a: {}, b: {} });
    renderCompare("/compare?runs=a,b");
    await waitFor(() => expect(screen.getByText("Comparing 2 runs")).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: "Remove a from the comparison" }));
    await waitFor(() => expect(screen.getByText("Comparing 1 runs")).toBeInTheDocument());
  });
});

describe("overlaid profiles", () => {
  it("puts one trace per run on each parameter plot", async () => {
    stubRuns({ a: {}, b: {} });
    renderCompare("/compare?runs=a,b");

    await waitFor(() => {
      const call = plotCalls.find((entry) => entry.ariaLabel === "Te_electron across runs");
      expect(call).toBeDefined();
      expect((call!.data as Array<Record<string, unknown>>).map((trace) => trace.name)).toEqual(["a", "b"]);
    });
  });

  it("plots a parameter only some runs fitted, and says how many", async () => {
    // ele-only vs ele+ion: the union is taken so nothing is hidden.
    stubRuns({
      a: {},
      b: {
        profiles: profiles({
          series: [{ name: "Te_electron", values: [1, 2, 3, 4, 5, 6], sigma: null }],
        }),
      },
    });
    renderCompare("/compare?runs=a,b");

    await waitFor(() => {
      const ne = plotCalls.find((entry) => entry.ariaLabel === "ne_electron across runs");
      expect(ne).toBeDefined();
      expect((ne!.data as unknown[]).length).toBe(1);
    });
    expect(screen.getByText("1 of 2 runs fitted this parameter.")).toBeInTheDocument();
  });
});

describe("angular runs are excluded, not reconciled", () => {
  it("keeps an angular run out of the overlays and says why", async () => {
    stubRuns({
      a: {},
      ang: { probe: angularAvailability(), profiles: null },
    });
    renderCompare("/compare?runs=a,ang");

    await waitFor(() => expect(screen.getByText("Not included in the overlays:")).toBeInTheDocument());
    expect(screen.getByText(/scattering angle/)).toBeInTheDocument();

    // Overlay has only the 1D run.
    const te = plotCalls.find((entry) => entry.ariaLabel === "Te_electron across runs");
    expect((te!.data as unknown[]).length).toBe(1);
    expect(screen.getByText("1 of 2 runs overlaid")).toBeInTheDocument();
  });

  it("still includes the angular run in the config diff", async () => {
    // This is the exception the scope note calls out: config comparison does not
    // care about axis semantics.
    stubRuns({
      a: { detail: runDetail({ run_id: "a", run_name: "a", config_flat: { "data.shotnum": "1" } }) },
      ang: {
        probe: angularAvailability(),
        profiles: null,
        detail: runDetail({ run_id: "ang", run_name: "ang", config_flat: { "data.shotnum": "9" } }),
      },
    });
    renderCompare("/compare?runs=a,ang");

    await waitFor(() => expect(screen.getByText("Config diff")).toBeInTheDocument());
    const row = screen.getByText("data.shotnum").closest("tr")!;
    expect(within(row).getByText("1")).toBeInTheDocument();
    expect(within(row).getByText("9")).toBeInTheDocument();
  });

  it("does not request profiles for an angular run", async () => {
    stubRuns({ ang: { probe: angularAvailability(), profiles: null } });
    renderCompare("/compare?runs=ang");
    await waitFor(() => expect(screen.getByText("Not included in the overlays:")).toBeInTheDocument());
    expect(requested.some((url) => url.includes("/profiles"))).toBe(false);
  });

  it("says a run logged no profiles when the probe reports none", async () => {
    stubRuns({ a: {}, none: { probe: availability({ profiles_available: false }) } });
    renderCompare("/compare?runs=a,none");

    await waitFor(() => expect(screen.getByText("Not included in the overlays:")).toBeInTheDocument());
    expect(screen.getByText(/No fitted-parameter profiles logged/i)).toBeInTheDocument();
  });

  it("distinguishes a failed profiles request from a run with no profiles", async () => {
    // The probe said profiles exist and the request still failed, so this is a
    // retryable error rather than a property of the run. Reporting it as "no
    // profiles logged" would send someone looking at their input deck instead.
    stubRuns({ a: {}, broken: { profiles: null } });
    renderCompare("/compare?runs=a,broken");

    await waitFor(() => expect(screen.getByText("Not included in the overlays:")).toBeInTheDocument());
    const notice = screen.getByText("Not included in the overlays:").closest("div")!;
    expect(within(notice).getByText(/Could not load this run's parameter profiles/i)).toBeInTheDocument();
    expect(within(notice).getByText(/run not found/)).toBeInTheDocument();
    expect(within(notice).queryByText(/No fitted-parameter profiles logged/i)).not.toBeInTheDocument();

    // The other run still overlays, and the config diff still covers both.
    expect(screen.getByText("1 of 2 runs overlaid")).toBeInTheDocument();
  });
});

describe("run cap", () => {
  it("loads at most MAX_COMPARE_RUNS and says what it dropped", async () => {
    const ids = Array.from({ length: MAX_COMPARE_RUNS + 4 }, (_, index) => `r${index}`);
    stubRuns(Object.fromEntries(ids.map((id) => [id, {}])));
    renderCompare(`/compare?runs=${ids.join(",")}`);

    await waitFor(() =>
      expect(screen.getByText(`Comparing ${MAX_COMPARE_RUNS} runs`)).toBeInTheDocument(),
    );

    expect(screen.getByText(new RegExp(`asked for ${MAX_COMPARE_RUNS + 4} runs`))).toBeInTheDocument();

    // The cap has to bound the requests, not just the rendering: the four runs
    // past the limit are never fetched at all.
    for (const dropped of ids.slice(MAX_COMPARE_RUNS)) {
      expect(requested.some((url) => url.includes(`/api/runs/${dropped}`))).toBe(false);
    }
  });

  it("shows no cap notice when the URL is within the limit", async () => {
    stubRuns({ a: {}, b: {} });
    renderCompare("/compare?runs=a,b");

    await waitFor(() => expect(screen.getByText("Comparing 2 runs")).toBeInTheDocument());
    expect(screen.queryByText(/asked for/)).not.toBeInTheDocument();
  });
});

describe("mixed axes", () => {
  it("warns when runs do not share a lineout axis", async () => {
    stubRuns({
      temporal: {},
      spatial: { profiles: profiles({ x_label: "Radius (\\mum)" }) },
    });
    renderCompare("/compare?runs=temporal,spatial");

    await waitFor(() => expect(screen.getByText(/do not share a lineout axis/)).toBeInTheDocument());
  });
});

describe("config diff", () => {
  it("shows changed keys only by default and can reveal the rest", async () => {
    stubRuns({
      a: { detail: runDetail({ run_id: "a", run_name: "a", config_flat: { "data.shotnum": "1", "other.refit": "False" } }) },
      b: { detail: runDetail({ run_id: "b", run_name: "b", config_flat: { "data.shotnum": "2", "other.refit": "False" } }) },
    });
    renderCompare("/compare?runs=a,b");

    await waitFor(() => expect(screen.getByText("data.shotnum")).toBeInTheDocument());
    expect(screen.queryByText("other.refit")).toBeNull();

    await userEvent.click(screen.getByRole("checkbox", { name: /Changed keys only/ }));
    await waitFor(() => expect(screen.getByText("other.refit")).toBeInTheDocument());
  });

  it("marks a key absent from one run rather than showing it blank", async () => {
    stubRuns({
      a: { detail: runDetail({ run_id: "a", run_name: "a", config_flat: { "only.a": "1" } }) },
      b: { detail: runDetail({ run_id: "b", run_name: "b", config_flat: {} }) },
    });
    renderCompare("/compare?runs=a,b");

    await waitFor(() => expect(screen.getByText("only.a")).toBeInTheDocument());
    expect(within(screen.getByText("only.a").closest("tr")!).getByText("absent")).toBeInTheDocument();
  });

  it("filters keys", async () => {
    stubRuns({
      a: { detail: runDetail({ run_id: "a", run_name: "a", config_flat: { "data.shotnum": "1", "other.thing": "2" } }) },
      b: { detail: runDetail({ run_id: "b", run_name: "b", config_flat: { "data.shotnum": "9", "other.thing": "8" } }) },
    });
    renderCompare("/compare?runs=a,b");
    await waitFor(() => expect(screen.getByText("data.shotnum")).toBeInTheDocument());

    await userEvent.type(screen.getByPlaceholderText("filter keys"), "shot");
    await waitFor(() => expect(screen.queryByText("other.thing")).toBeNull());
    expect(screen.getByText("data.shotnum")).toBeInTheDocument();
  });

  it("says so when the configs are identical", async () => {
    stubRuns({
      a: { detail: runDetail({ run_id: "a", run_name: "a", config_flat: { "x.y": "1" } }) },
      b: { detail: runDetail({ run_id: "b", run_name: "b", config_flat: { "x.y": "1" } }) },
    });
    renderCompare("/compare?runs=a,b");
    await waitFor(() =>
      expect(screen.getByText("These runs have identical configurations.")).toBeInTheDocument(),
    );
  });
});

describe("loss curves", () => {
  it("overlays one trace per run", async () => {
    stubRuns({ a: {}, b: {} });
    renderCompare("/compare?runs=a,b");

    // The last call, not the first: the panel renders once before the histories
    // arrive, so the earliest call legitimately has no traces.
    await waitFor(() => {
      const loss = plotCalls.filter((entry) => entry.ariaLabel === "epoch loss across runs").at(-1);
      expect(loss).toBeDefined();
      expect((loss!.data as unknown[]).length).toBe(2);
    });
  });

  it("names runs that did not log the selected metric", async () => {
    stubRuns({
      a: {},
      b: { detail: runDetail({ run_id: "b", run_name: "b", metrics: [{ key: "overall loss", value: 1 }] }) },
    });
    renderCompare("/compare?runs=a,b");
    await waitFor(() => expect(screen.getByText(/Not logged by: b/)).toBeInTheDocument());
  });
});

describe("failures", () => {
  it("compares the runs it could load and reports the ones it could not", async () => {
    stubRuns({ a: {}, broken: { fail: true } });
    renderCompare("/compare?runs=a,broken");

    await waitFor(() => expect(screen.getByText("Comparing 1 runs")).toBeInTheDocument());
    expect(screen.getByRole("alert")).toHaveTextContent(/broken/);
  });

  it("errors with a retry only when nothing could be loaded", async () => {
    stubRuns({ x: { fail: true } });
    renderCompare("/compare?runs=x");

    await waitFor(() => expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument());
  });
});
