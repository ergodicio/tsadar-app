import { describe, expect, it } from "vitest";

import { configSources, diffConfigs, displayValue, flattenConfig } from "../lib/config";

describe("flattenConfig", () => {
  it("uses dotted keys, matching how tsadar logs params", () => {
    expect(flattenConfig({ data: { shotnum: 101675, lineouts: { start: 800 } } })).toEqual({
      "data.shotnum": 101675,
      "data.lineouts.start": 800,
    });
  });

  it("treats arrays as leaves rather than indexing into them", () => {
    // other.lamrangE is [400, 700] in the real config; splitting it into
    // lamrangE.0 / lamrangE.1 would make the diff unreadable.
    expect(flattenConfig({ other: { lamrangE: [400, 700] } })).toEqual({
      "other.lamrangE": [400, 700],
    });
  });

  it("keeps nulls as leaves", () => {
    expect(flattenConfig({ a: { b: null } })).toEqual({ "a.b": null });
  });
});

describe("diffConfigs", () => {
  const defaults = { data: { shotnum: 1, lineouts: { start: 800 } }, other: { refit: false } };
  const inputs = { data: { shotnum: 101675, lineouts: { start: 800 } }, extra: { thing: 2 } };

  it("classifies changed, same, added and removed keys", () => {
    const byKey = new Map(diffConfigs(defaults, inputs).map((row) => [row.key, row]));

    expect(byKey.get("data.shotnum")?.status).toBe("changed");
    expect(byKey.get("data.lineouts.start")?.status).toBe("same");
    expect(byKey.get("extra.thing")?.status).toBe("added");
    expect(byKey.get("other.refit")?.status).toBe("removed");
  });

  it("carries both sides so the table can show them", () => {
    const row = diffConfigs(defaults, inputs).find((entry) => entry.key === "data.shotnum")!;
    expect(row.base).toBe(1);
    expect(row.override).toBe(101675);
  });

  it("compares structurally, so an equal array is not a change", () => {
    const rows = diffConfigs({ a: [1, 2] }, { a: [1, 2] });
    expect(rows[0]?.status).toBe("same");
  });

  it("is sorted by key so the table order is stable", () => {
    const keys = diffConfigs(defaults, inputs).map((row) => row.key);
    expect(keys).toEqual([...keys].sort());
  });
});

describe("displayValue", () => {
  it("distinguishes null from absent", () => {
    expect(displayValue(null)).toBe("null");
    expect(displayValue(undefined)).toBe("—");
  });

  it("renders booleans and numbers plainly and objects as JSON", () => {
    expect(displayValue(false)).toBe("false");
    expect(displayValue(5.0)).toBe("5");
    expect(displayValue([400, 700])).toBe("[400,700]");
  });
});

describe("configSources", () => {
  it("is diffable only when both defaults and inputs are present", () => {
    // NERSC-queued: two files, so "what did this run change?" is answerable.
    expect(configSources(["defaults.yaml", "inputs.yaml"]).diffable).toBe(true);
    // App-queued: one merged file, nothing to diff against.
    expect(configSources(["config.yaml"]).diffable).toBe(false);
    expect(configSources(["defaults.yaml"]).diffable).toBe(false);
  });

  it("reports which config artifacts exist", () => {
    const sources = configSources(["config.yaml", "plots/a.png"]);
    expect(sources.hasMerged).toBe(true);
    expect(sources.hasDefaults).toBe(false);
  });
});
