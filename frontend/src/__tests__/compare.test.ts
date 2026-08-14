import { describe, expect, it } from "vitest";

import {
  axisLabels,
  comparableRuns,
  diffAcrossRuns,
  exclusionReason,
  mixedAxisWarning,
  parseRunIds,
  runLabel,
  runsWithSeries,
  sharedSeriesNames,
  type ComparisonRun,
} from "../lib/compare";
import { angularAvailability, availability, profiles, runDetail } from "./fixtures";

function comparisonRun(overrides: Partial<ComparisonRun> & { runId: string }): ComparisonRun {
  const detail = (overrides.detail ?? runDetail({ run_id: overrides.runId })) as ComparisonRun["detail"];
  const probe = (overrides.availability ?? availability()) as ComparisonRun["availability"];
  const runProfiles = overrides.profiles === undefined ? (profiles() as ComparisonRun["profiles"]) : overrides.profiles;
  return {
    runId: overrides.runId,
    detail,
    availability: probe,
    profiles: runProfiles,
    excluded: overrides.excluded ?? exclusionReason(probe, runProfiles),
  };
}

describe("parseRunIds", () => {
  it("splits the URL parameter and preserves order", () => {
    expect(parseRunIds("a,b,c")).toEqual(["a", "b", "c"]);
  });

  it("drops blanks, whitespace and duplicates", () => {
    // Arbitrary user input: /compare?runs= can be hand-edited.
    expect(parseRunIds("a, ,b,,a, b ")).toEqual(["a", "b"]);
  });

  it("handles an absent parameter", () => {
    expect(parseRunIds(null)).toEqual([]);
    expect(parseRunIds("")).toEqual([]);
  });
});

describe("exclusionReason", () => {
  it("excludes an angular run and explains why", () => {
    const reason = exclusionReason(angularAvailability() as never, null);
    expect(reason).toMatch(/scattering angle/);
    expect(reason).toMatch(/config still appears/);
  });

  it("excludes a run with no readable datasets, using the backend's message", () => {
    const probe = availability({
      kind: "unknown",
      supported: false,
      message: "This run has no readable fit/data datasets.",
      spectra: [],
    });
    expect(exclusionReason(probe as never, null)).toBe("This run has no readable fit/data datasets.");
  });

  it("excludes a supported run that logged no profiles", () => {
    expect(exclusionReason(availability() as never, null)).toMatch(/no fitted-parameter profiles/i);
  });

  it("includes a normal 1D run", () => {
    expect(exclusionReason(availability() as never, profiles() as never)).toBeNull();
  });
});

describe("axis compatibility", () => {
  it("warns when runs do not share a lineout axis", () => {
    // Radius (μm) and Time (ps) are both 1D but are not the same axis.
    const runs = [
      comparisonRun({ runId: "a" }),
      comparisonRun({
        runId: "b",
        profiles: profiles({ x_label: "Radius (\\mum)" }) as never,
      }),
    ];
    const warning = mixedAxisWarning(runs);
    expect(warning).toMatch(/do not share a lineout axis/);
    expect(warning).toContain("Time (ps)");
  });

  it("is silent when the axes agree", () => {
    const runs = [comparisonRun({ runId: "a" }), comparisonRun({ runId: "b" })];
    expect(mixedAxisWarning(runs)).toBeNull();
  });

  it("ignores excluded runs when deciding, since they are not overlaid", () => {
    const runs = [
      comparisonRun({ runId: "a" }),
      comparisonRun({ runId: "angular", availability: angularAvailability() as never, profiles: null }),
    ];
    expect(axisLabels(runs)).toEqual(["Time (ps)"]);
    expect(mixedAxisWarning(runs)).toBeNull();
  });
});

describe("series union across runs", () => {
  it("takes the union so a parameter one run lacks is still plotted", () => {
    // An ele-only run has no ion parameters; dropping Ti_ion would hide data the
    // other run does have.
    const runs = [
      comparisonRun({ runId: "a" }),
      comparisonRun({
        runId: "b",
        profiles: profiles({
          series: [
            { name: "Te_electron", values: [1, 2, 3, 4, 5, 6], sigma: null },
            { name: "Ti_ion", values: [1, 2, 3, 4, 5, 6], sigma: null },
          ],
        }) as never,
      }),
    ];
    expect(sharedSeriesNames(runs)).toEqual(["Te_electron", "Ti_ion", "ne_electron"]);
  });

  it("reports which runs actually have a series", () => {
    const runs = [
      comparisonRun({ runId: "a" }),
      comparisonRun({
        runId: "b",
        profiles: profiles({ series: [{ name: "Te_electron", values: [1], sigma: null }] }) as never,
      }),
    ];
    expect(runsWithSeries(runs, "Te_electron").map((run) => run.runId)).toEqual(["a", "b"]);
    expect(runsWithSeries(runs, "ne_electron").map((run) => run.runId)).toEqual(["a"]);
  });

  it("excludes non-overlayable runs from the union", () => {
    const runs = [
      comparisonRun({ runId: "angular", availability: angularAvailability() as never, profiles: null }),
    ];
    expect(comparableRuns(runs)).toEqual([]);
    expect(sharedSeriesNames(runs)).toEqual([]);
  });
});

describe("diffAcrossRuns", () => {
  it("puts one column per run keyed by flattened param", () => {
    const runs = [
      comparisonRun({
        runId: "a",
        detail: runDetail({ config_flat: { "data.shotnum": "1", "other.refit": "False" } }) as never,
      }),
      comparisonRun({
        runId: "b",
        detail: runDetail({ config_flat: { "data.shotnum": "2", "other.refit": "False" } }) as never,
      }),
    ];
    const byKey = new Map(diffAcrossRuns(runs).map((row) => [row.key, row]));

    expect(byKey.get("data.shotnum")?.values).toEqual(["1", "2"]);
    expect(byKey.get("data.shotnum")?.varies).toBe(true);
    expect(byKey.get("other.refit")?.varies).toBe(false);
  });

  it("treats a key absent from one run as varying, and marks it undefined", () => {
    // Different decks: absent is not the same as a different value.
    const runs = [
      comparisonRun({ runId: "a", detail: runDetail({ config_flat: { "a.b": "1" } }) as never }),
      comparisonRun({ runId: "b", detail: runDetail({ config_flat: {} }) as never }),
    ];
    const row = diffAcrossRuns(runs)[0]!;
    expect(row.values).toEqual(["1", undefined]);
    expect(row.varies).toBe(true);
  });

  it("includes angular runs, whose config is still comparable", () => {
    const runs = [
      comparisonRun({ runId: "a", detail: runDetail({ config_flat: { "data.shotnum": "1" } }) as never }),
      comparisonRun({
        runId: "angular",
        availability: angularAvailability() as never,
        profiles: null,
        detail: runDetail({ config_flat: { "data.shotnum": "9" } }) as never,
      }),
    ];
    expect(diffAcrossRuns(runs)[0]?.values).toEqual(["1", "9"]);
  });

  it("is sorted by key for a stable table", () => {
    const runs = [
      comparisonRun({
        runId: "a",
        detail: runDetail({ config_flat: { "z.z": "1", "a.a": "2" } }) as never,
      }),
    ];
    expect(diffAcrossRuns(runs).map((row) => row.key)).toEqual(["a.a", "z.z"]);
  });
});

describe("runLabel", () => {
  it("prefers the run name", () => {
    expect(runLabel(comparisonRun({ runId: "a" }))).toBe("shot-101675-scan");
  });

  it("falls back to the shot, then a short id", () => {
    expect(
      runLabel(comparisonRun({ runId: "a", detail: runDetail({ run_name: null }) as never })),
    ).toBe("shot 101675");
    expect(
      runLabel(
        comparisonRun({
          runId: "abcdefghijkl",
          detail: runDetail({ run_name: null, shot: null, run_id: "abcdefghijkl" }) as never,
        }),
      ),
    ).toBe("abcdefgh");
  });
});
