import { describe, expect, it } from "vitest";

import {
  filtersFromSearch,
  hasActiveFilters,
  searchFromFilters,
  sortDirection,
  toggleSort,
} from "../lib/urlState";

describe("filters round-trip through the URL", () => {
  it("reads every filter key from the query string", () => {
    const search = new URLSearchParams(
      "experiment=inverse-thomson-scattering&shot=101675&status=FINISHED&stage=completed&user=archis&q=scan&sort=-loss",
    );
    expect(filtersFromSearch(search)).toEqual({
      experiment: "inverse-thomson-scattering",
      shot: "101675",
      status: "FINISHED",
      stage: "completed",
      user: "archis",
      q: "scan",
      sort: "-loss",
    });
  });

  it("survives a round trip so a pasted link reproduces the view", () => {
    const filters = { shot: "101675", status: "FAILED", sort: "name" };
    expect(filtersFromSearch(searchFromFilters(filters))).toEqual(filters);
  });

  it("ignores blank values rather than sending empty filters to the API", () => {
    expect(filtersFromSearch(new URLSearchParams("shot=&user=%20%20"))).toEqual({});
    expect(searchFromFilters({ shot: "", user: "  " }).toString()).toBe("");
  });

  it("ignores unknown query parameters", () => {
    expect(filtersFromSearch(new URLSearchParams("shot=1&page_token=abc&nonsense=1"))).toEqual({
      shot: "1",
    });
  });
});

describe("hasActiveFilters", () => {
  it("is false when only a sort is set, so the empty state reads correctly", () => {
    // Sorting is not filtering: "no runs match your filters" would be wrong copy.
    expect(hasActiveFilters({ sort: "-created" })).toBe(false);
  });

  it("is true when any real filter is set", () => {
    expect(hasActiveFilters({ shot: "101675" })).toBe(true);
  });

  it("is false when nothing is set", () => {
    expect(hasActiveFilters({})).toBe(false);
  });
});

describe("sort toggling", () => {
  it("cycles ascending, descending, ascending", () => {
    expect(toggleSort(undefined, "shot")).toBe("shot");
    expect(toggleSort("shot", "shot")).toBe("-shot");
    expect(toggleSort("-shot", "shot")).toBe("shot");
  });

  it("switching column starts ascending", () => {
    expect(toggleSort("-shot", "name")).toBe("name");
  });

  it("reports direction for header indicators", () => {
    expect(sortDirection("shot", "shot")).toBe("asc");
    expect(sortDirection("-shot", "shot")).toBe("desc");
    expect(sortDirection("-shot", "name")).toBeNull();
  });
});
