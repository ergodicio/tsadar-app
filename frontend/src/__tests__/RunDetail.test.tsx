/**
 * Run detail view (issue #32).
 *
 * Plotly is mocked, so the assertions are about the traces each panel builds --
 * which is the part worth testing. Whether Plotly draws a heatmap correctly is
 * Plotly's problem; whether we hand it the fit array when the user asked for the
 * fit is ours.
 */

import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RunDetail } from "../routes/RunDetail";
import {
  angularAvailability,
  availability,
  lineout,
  metricHistory,
  profiles,
  runDetail,
  spectrogram,
} from "./fixtures";

// Capture what each Plot receives instead of rendering one.
const plotCalls: Array<{ data: unknown[]; layout?: Record<string, unknown>; ariaLabel?: string }> = [];

vi.mock("../components/Plot", () => ({
  Plot: (props: { data: unknown[]; layout?: Record<string, unknown>; ariaLabel?: string }) => {
    plotCalls.push(props);
    return <div data-testid="plot" data-aria={props.ariaLabel} />;
  },
}));

interface StubOptions {
  detail?: Record<string, unknown>;
  probe?: Record<string, unknown>;
  spectrogramBody?: Record<string, unknown>;
  failMetric?: boolean;
  yaml?: Record<string, string>;
}

let requested: string[] = [];

function stubApi(options: StubOptions = {}) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      requested.push(url);
      const ok = (body: unknown): Response =>
        ({
          ok: true,
          status: 200,
          json: async (): Promise<unknown> => body,
          text: async (): Promise<string> => "",
        }) as unknown as Response;

      if (url.includes("/datasets")) return ok(options.probe ?? availability());
      if (url.includes("/spectrogram")) {
        const field = new URL(url, "http://x").searchParams.get("field") ?? "data";
        return ok(spectrogram({ field, ...options.spectrogramBody }));
      }
      if (url.includes("/lineout")) {
        const index = Number(new URL(url, "http://x").searchParams.get("index") ?? "0");
        return ok(lineout({ index, x_value: -100 + index * 40 }));
      }
      if (url.includes("/profiles")) return ok(options.probe === undefined ? profiles() : profiles());
      if (url.includes("/metrics/")) {
        if (options.failMetric) return { ok: false, status: 404, json: async () => ({ detail: "no history" }) };
        const key = decodeURIComponent(url.split("/metrics/")[1] ?? "");
        return ok(metricHistory(key));
      }
      if (url.includes("/artifacts/")) {
        const path = url.split("/artifacts/")[1] ?? "";
        const text = options.yaml?.[path];
        if (text === undefined) {
          return { ok: false, status: 404, text: async (): Promise<string> => "" } as unknown as Response;
        }
        return {
          ok: true,
          status: 200,
          text: async (): Promise<string> => text,
          json: async (): Promise<unknown> => ({}),
        } as unknown as Response;
      }
      return ok(options.detail ?? runDetail());
    }),
  );
}

function renderDetail(url = "/runs/run-abc") {
  return render(
    <MemoryRouter initialEntries={[url]}>
      <Routes>
        <Route path="/runs/:runId" element={<RunDetail />} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  plotCalls.length = 0;
  requested = [];
});

afterEach(() => vi.unstubAllGlobals());

describe("header", () => {
  it("shows identity, status and stage as separate badges", async () => {
    stubApi();
    renderDetail();

    await waitFor(() => expect(screen.getByText("shot-101675-scan")).toBeInTheDocument());

    // Scoped to the run header: the shot number also appears in the config tree,
    // and every panel has its own <header>, so anchor on the h1.
    const header = within(
      screen.getByRole("heading", { level: 1 }).closest("header") as HTMLElement,
    );
    // Two badges, not one: they answer different questions.
    expect(header.getByText("FINISHED")).toBeInTheDocument();
    expect(header.getByText("completed")).toBeInTheDocument();
    expect(header.getByText("101675")).toBeInTheDocument();
    expect(header.getByText("archis")).toBeInTheDocument();
  });

  it("names the metric its final loss came from", async () => {
    stubApi();
    renderDetail();
    await waitFor(() => expect(screen.getByText("Final loss (overall loss)")).toBeInTheDocument());
  });

  it("links out to the raw MLflow run page", async () => {
    stubApi();
    renderDetail();
    await waitFor(() =>
      expect(screen.getByRole("link", { name: "Open in MLflow" })).toHaveAttribute(
        "href",
        "https://continuum.ergodic.io/experiments/#/experiments/1/runs/run-abc",
      ),
    );
  });
});

describe("spectrogram panel", () => {
  it("plots the heatmap with axis coordinates", async () => {
    stubApi();
    renderDetail();

    await waitFor(() => expect(plotCalls.length).toBeGreaterThan(0));
    const heatmap = plotCalls.find((call) => call.ariaLabel?.includes("spectrogram"));
    expect(heatmap).toBeDefined();
    const trace = heatmap!.data[0] as Record<string, unknown>;
    expect(trace.type).toBe("heatmap");
    expect(trace.z).toEqual(spectrogram().values);
    expect(trace.x).toEqual(spectrogram().x);
  });

  it("offers exactly data, fit and residual -- no irf", async () => {
    // IRF isn't in the netCDF datasets, so offering a toggle for it would
    // promise something the backend cannot serve.
    stubApi();
    renderDetail();

    await waitFor(() => expect(screen.getByRole("group", { name: "Field" })).toBeInTheDocument());
    const fields = within(screen.getByRole("group", { name: "Field" })).getAllByRole("button");
    expect(fields.map((button) => button.textContent)).toEqual(["data", "fit", "residual"]);
  });

  it("switching to fit refetches that field", async () => {
    stubApi();
    renderDetail();
    await waitFor(() => expect(screen.getByRole("group", { name: "Field" })).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: "fit" }));
    await waitFor(() => expect(requested.some((url) => url.includes("field=fit"))).toBe(true));
  });

  it("uses a diverging scale centred at zero for the signed residual", async () => {
    stubApi();
    renderDetail();
    await waitFor(() => expect(screen.getByRole("group", { name: "Field" })).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: "residual" }));
    await waitFor(() => {
      const call = plotCalls.filter((c) => c.ariaLabel === "residual spectrogram").at(-1);
      expect(call).toBeDefined();
      const trace = call!.data[0] as Record<string, unknown>;
      expect(trace.colorscale).toBe("RdBu");
      expect(trace.zmid).toBe(0);
    });
  });

  it("says what it did to the array rather than implying full resolution", async () => {
    stubApi();
    renderDetail();
    await waitFor(() =>
      expect(screen.getByText(/Block-averaged 8× in wavelength from 6 × 32/)).toBeInTheDocument(),
    );
  });
});

describe("lineout scrubber", () => {
  it("renders measured, fitted and residual traces", async () => {
    stubApi();
    renderDetail();

    await waitFor(() => {
      const call = plotCalls.find((c) => c.ariaLabel === "Measured versus fitted spectrum");
      expect(call).toBeDefined();
      const names = (call!.data as Array<Record<string, unknown>>).map((trace) => trace.name);
      expect(names).toEqual(["Data", "Fit", "Residual"]);
    });
  });

  it("moving the scrubber refetches that lineout and puts it in the URL", async () => {
    stubApi();
    renderDetail();
    await waitFor(() => expect(screen.getByLabelText("Lineout index")).toBeInTheDocument());

    // Range inputs cannot be typed into; a change event is how they move.
    fireEvent.change(screen.getByLabelText("Lineout index"), { target: { value: "3" } });

    await waitFor(() => expect(requested.some((url) => url.includes("index=3"))).toBe(true));
  });

  it("reads the starting lineout from the URL so a link can point at one", async () => {
    stubApi();
    renderDetail("/runs/run-abc?lineout=4");
    await waitFor(() => expect(requested.some((url) => url.includes("index=4"))).toBe(true));
    expect(screen.getByText("Lineout 5 of 6")).toBeInTheDocument();
  });

  it("explains that IRF components are unavailable rather than omitting them", async () => {
    stubApi();
    renderDetail();
    await waitFor(() =>
      expect(screen.getByText(/IRF and noise components are not available/)).toBeInTheDocument(),
    );
  });
});

describe("profiles panel", () => {
  it("groups series by parameter and plots against the lineout axis", async () => {
    stubApi();
    renderDetail();

    await waitFor(() => {
      const te = plotCalls.find((c) => c.ariaLabel === "Te versus lineout");
      expect(te).toBeDefined();
      const trace = (te!.data as Array<Record<string, unknown>>)[0]!;
      // Species becomes the trace name so two species overlay on one plot.
      expect(trace.name).toBe("electron");
      expect(trace.x).toEqual(profiles().x);
    });
  });

  it("says when a run computed no uncertainties instead of showing bare lines", async () => {
    stubApi();
    renderDetail();
    await waitFor(() => expect(screen.getByText(/did not compute uncertainties/)).toBeInTheDocument());
  });
});

describe("loss panel", () => {
  it("prefers a per-step metric over a single-point summary", async () => {
    // There is no metric called "loss"; requesting one would 404 on every run.
    stubApi();
    renderDetail();

    await waitFor(() => expect(requested.some((url) => url.includes("/metrics/"))).toBe(true));
    const metricRequest = requested.find((url) => url.includes("/metrics/"))!;
    expect(decodeURIComponent(metricRequest)).toContain("/metrics/epoch loss");
    expect(metricRequest).not.toContain("/metrics/loss?");
  });

  it("encodes the space in the metric name", async () => {
    stubApi();
    renderDetail();
    await waitFor(() => expect(requested.some((url) => url.includes("epoch%20loss"))).toBe(true));
  });

  it("offers the run's other loss metrics but not unrelated ones", async () => {
    stubApi();
    renderDetail();
    await waitFor(() => expect(screen.getByLabelText(/Metric/i)).toBeInTheDocument());

    const options = within(screen.getByRole("combobox", { name: /Metric/i })).getAllByRole("option");
    expect(options.map((option) => option.textContent)).toEqual(["epoch loss", "overall loss"]);
  });

  it("says so when a run logged no loss metric at all", async () => {
    stubApi({ detail: runDetail({ metrics: [{ key: "fit_time", value: 42 }] }) });
    renderDetail();
    await waitFor(() => expect(screen.getByText("This run logged no loss metrics.")).toBeInTheDocument());
  });
});

describe("config panel", () => {
  it("shows the merged tree rebuilt from logged params", async () => {
    stubApi();
    renderDetail();
    await waitFor(() => expect(screen.getByText("shotnum")).toBeInTheDocument());
    // The tree renders the value next to its key, distinct from the header fact.
    const tree = screen.getByText("shotnum").closest("li");
    expect(tree).not.toBeNull();
    expect(within(tree as HTMLElement).getByText("101675")).toBeInTheDocument();
  });

  it("disables the diff for an app-queued run that logged one merged config", async () => {
    stubApi();
    renderDetail();

    await waitFor(() => expect(screen.getByRole("button", { name: "diff" })).toBeDisabled());
    expect(
      screen.getByText(/logged a single merged config.yaml, so there are no defaults to diff/),
    ).toBeInTheDocument();
  });

  it("diffs inputs against defaults for a NERSC-queued run", async () => {
    stubApi({
      detail: runDetail({
        artifacts: [
          { path: "defaults.yaml", is_dir: false, size: 200 },
          { path: "inputs.yaml", is_dir: false, size: 80 },
        ],
      }),
      yaml: {
        "defaults.yaml": "data:\n  shotnum: 1\n  lineouts:\n    start: 800\nother:\n  refit: false\n",
        "inputs.yaml": "data:\n  shotnum: 101675\n  lineouts:\n    start: 800\n",
      },
    });
    renderDetail();

    await waitFor(() => expect(screen.getByRole("button", { name: "diff" })).toBeEnabled());
    await userEvent.click(screen.getByRole("button", { name: "diff" }));

    await waitFor(() => expect(screen.getByText("data.shotnum")).toBeInTheDocument());
    // Changed-only by default: that's the "what did I vary?" question.
    expect(screen.queryByText("data.lineouts.start")).toBeNull();

    await userEvent.click(screen.getByRole("checkbox", { name: /Changed keys only/ }));
    await waitFor(() => expect(screen.getByText("data.lineouts.start")).toBeInTheDocument());
  });
});

describe("angular and pre-contract runs", () => {
  it("shows the gallery with the backend's reason, not an error", async () => {
    stubApi({
      detail: runDetail({ spectype: "angular_full" }),
      probe: angularAvailability(),
    });
    renderDetail();

    await waitFor(() => expect(screen.getByText("Interactive views unavailable")).toBeInTheDocument());
    expect(screen.getByText(/angularly-resolved run/)).toBeInTheDocument();
    // Not an error: this is a scope decision, not a failure.
    expect(screen.queryByRole("alert")).toBeNull();
    // And the page is never blank.
    expect(screen.getByText("Plots and files")).toBeInTheDocument();
    expect(screen.getByAltText("plots/fit_and_data.png")).toBeInTheDocument();
  });

  it("labels an angular run in the header", async () => {
    stubApi({ detail: runDetail({ spectype: "angular_full" }), probe: angularAvailability() });
    renderDetail();
    await waitFor(() => expect(screen.getByText("angular")).toBeInTheDocument());
  });

  it("does not request slicing endpoints for an unsupported run", async () => {
    stubApi({ probe: angularAvailability() });
    renderDetail();
    await waitFor(() => expect(screen.getByText("Interactive views unavailable")).toBeInTheDocument());
    expect(requested.some((url) => url.includes("/spectrogram"))).toBe(false);
    expect(requested.some((url) => url.includes("/lineout"))).toBe(false);
  });

  it("shows the gallery for a pre-contract run with its own message", async () => {
    stubApi({
      probe: availability({
        kind: "unknown",
        supported: false,
        reason: "dataset_missing",
        message: "This run has no readable fit/data datasets, which is expected for older runs.",
        spectra: [],
        profiles_available: false,
      }),
    });
    renderDetail();
    await waitFor(() => expect(screen.getByText(/no readable fit\/data datasets/)).toBeInTheDocument());
  });
});

describe("artifact gallery", () => {
  it("serves images through the API rather than S3", async () => {
    stubApi();
    renderDetail();
    await waitFor(() =>
      expect(screen.getByAltText("plots/fit_and_data.png")).toHaveAttribute(
        "src",
        "/api/runs/run-abc/artifacts/plots/fit_and_data.png",
      ),
    );
  });

  it("lists non-image artifacts separately", async () => {
    stubApi();
    renderDetail();
    await waitFor(() =>
      expect(screen.getByRole("link", { name: "csv/learned_parameters.csv" })).toBeInTheDocument(),
    );
  });

  it("skips directory entries", async () => {
    stubApi();
    renderDetail();
    await waitFor(() => expect(screen.getByText("Plots and files")).toBeInTheDocument());
    expect(screen.queryByRole("link", { name: "binary" })).toBeNull();
  });
});

describe("failures", () => {
  it("surfaces an error with a retry when the run cannot be read", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: false, status: 404, json: async () => ({ detail: "run not found" }) })),
    );
    renderDetail();

    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(screen.getByText("run not found")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });

  it("explains a non-Thomson run instead of calling it an error", async () => {
    // The tracking server is shared, so following a link to an ADEPT run is a
    // normal thing to do. It must read as out of scope, not as a broken page --
    // and must not offer a retry, since retrying cannot change the answer.
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 404,
        json: async () => ({
          detail: {
            reason: "not_thomson",
            detail: "run adept-1 is not a Thomson analysis run.",
          },
        }),
      })),
    );
    renderDetail();

    await waitFor(() => expect(screen.getByText("Not a Thomson run")).toBeInTheDocument());
    expect(screen.getByText("run adept-1 is not a Thomson analysis run.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("a failed loss history does not take down the rest of the page", async () => {
    stubApi({ failMetric: true });
    renderDetail();

    await waitFor(() => expect(screen.getByText("shot-101675-scan")).toBeInTheDocument());
    expect(screen.getByText("Plots and files")).toBeInTheDocument();
  });
});
