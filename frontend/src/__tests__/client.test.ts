import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api, runQueryParams } from "../api/client";

afterEach(() => {
  vi.unstubAllGlobals();
});

function stubFetch(response: { status?: number; body?: unknown; ok?: boolean }) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: response.ok ?? (response.status ?? 200) < 400,
    status: response.status ?? 200,
    json: async () => response.body,
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("runQueryParams", () => {
  it("maps UI state onto the backend's parameter names", () => {
    const params = runQueryParams({
      experiment: "inverse-thomson-scattering",
      shot: "101675",
      status: "FINISHED",
      stage: "completed",
      user: "archis",
      q: "scan",
      sort: "-loss",
      pageSize: 50,
      pageToken: "cursor-1",
    });
    expect(params.get("shot")).toBe("101675");
    expect(params.get("page_size")).toBe("50");
    // Snake case, and a *token* rather than a page number.
    expect(params.get("page_token")).toBe("cursor-1");
    expect(params.has("page")).toBe(false);
  });

  it("omits empty values instead of sending blank filters", () => {
    expect(runQueryParams({ shot: "", user: undefined }).toString()).toBe("");
  });
});

describe("api.runs", () => {
  it("requests /api/runs with the query string", async () => {
    const fetchMock = stubFetch({ body: { runs: [], page_size: 50, next_page_token: null } });
    await api.runs({ shot: "101675" });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/runs?shot=101675",
      expect.objectContaining({ headers: { Accept: "application/json" } }),
    );
  });

  it("returns the cursor so pagination can continue", async () => {
    stubFetch({ body: { runs: [], page_size: 50, next_page_token: "cursor-2" } });
    await expect(api.runs({})).resolves.toMatchObject({ next_page_token: "cursor-2" });
  });
});

describe("error handling", () => {
  it("surfaces a plain string detail", async () => {
    stubFetch({ status: 400, body: { detail: "cannot sort by nonsense" } });
    await expect(api.runs({ sort: "nonsense" })).rejects.toThrow("cannot sort by nonsense");
  });

  it("extracts reason and detail from the dataset endpoints' error shape", async () => {
    // The dataset endpoints (#30) return {reason, detail} so the UI can tell an
    // out-of-scope view apart from missing data.
    stubFetch({
      status: 409,
      body: { detail: { reason: "angular_not_supported", detail: "angular run" } },
    });
    const error = await api.run("run-angular").catch((cause: unknown) => cause);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).reason).toBe("angular_not_supported");
    expect((error as ApiError).status).toBe(409);
  });

  it("still throws when the error body is not JSON", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 502,
      json: async () => {
        throw new Error("not json");
      },
    });
    vi.stubGlobal("fetch", fetchMock);
    await expect(api.runs({})).rejects.toThrow("request failed (502)");
  });

  it("percent-encodes the run id", async () => {
    const fetchMock = stubFetch({ body: {} });
    await api.run("run/../secret");
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/runs/run%2F..%2Fsecret");
  });
});
