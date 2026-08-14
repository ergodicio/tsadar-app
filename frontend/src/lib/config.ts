/**
 * Config tree flattening and diffing.
 *
 * Two ways a run records its config, which is why the diff is conditional:
 *
 * - **App-queued runs** log a single `config.yaml` — already the merged result,
 *   so there is nothing to diff against.
 * - **NERSC-queued runs** log `defaults.yaml` *and* `inputs.yaml`, where inputs
 *   overrides defaults. Diffing those answers "what did this run actually
 *   change?", which is the useful question.
 *
 * The run detail response also carries `config`, the tree rebuilt from logged
 * params. That is the authoritative merged view and is always shown; the diff is
 * an extra when both YAML files are present.
 */

export type ConfigValue = unknown;

/** Flatten a nested config to dotted keys, matching how tsadar logs params
 *  (`flatten_dict` with a dot reducer in `misc.log_mlflow`). */
export function flattenConfig(value: ConfigValue, prefix = ""): Record<string, ConfigValue> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return prefix ? { [prefix]: value } : {};
  }

  const flat: Record<string, ConfigValue> = {};
  for (const [key, child] of Object.entries(value as Record<string, ConfigValue>)) {
    const path = prefix ? `${prefix}.${key}` : key;
    Object.assign(flat, flattenConfig(child, path));
  }
  return flat;
}

export type DiffStatus = "changed" | "added" | "removed" | "same";

export interface ConfigDiffRow {
  key: string;
  status: DiffStatus;
  base: ConfigValue;
  override: ConfigValue;
}

/** Render a value for comparison and display.
 *
 *  Compared as JSON so nested arrays and objects diff structurally rather than
 *  by reference. */
export function displayValue(value: ConfigValue): string {
  if (value === undefined) return "—";
  if (value === null) return "null";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

/** Diff an override config against a base, keyed by flattened path. */
export function diffConfigs(base: ConfigValue, override: ConfigValue): ConfigDiffRow[] {
  const flatBase = flattenConfig(base);
  const flatOverride = flattenConfig(override);
  const keys = [...new Set([...Object.keys(flatBase), ...Object.keys(flatOverride)])].sort();

  return keys.map((key) => {
    const inBase = key in flatBase;
    const inOverride = key in flatOverride;
    const baseValue = flatBase[key];
    const overrideValue = flatOverride[key];

    let status: DiffStatus;
    if (!inBase) status = "added";
    else if (!inOverride) status = "removed";
    else status = displayValue(baseValue) === displayValue(overrideValue) ? "same" : "changed";

    return { key, status, base: baseValue, override: overrideValue };
  });
}

/** Artifact names that carry config, in the two shapes runs use. */
export const CONFIG_ARTIFACTS = {
  merged: "config.yaml",
  defaults: "defaults.yaml",
  inputs: "inputs.yaml",
} as const;

export interface ConfigSources {
  hasMerged: boolean;
  hasDefaults: boolean;
  hasInputs: boolean;
  /** A diff is only meaningful when both halves exist. */
  diffable: boolean;
}

export function configSources(artifactPaths: readonly string[]): ConfigSources {
  const paths = new Set(artifactPaths);
  const hasDefaults = paths.has(CONFIG_ARTIFACTS.defaults);
  const hasInputs = paths.has(CONFIG_ARTIFACTS.inputs);
  return {
    hasMerged: paths.has(CONFIG_ARTIFACTS.merged),
    hasDefaults,
    hasInputs,
    diffable: hasDefaults && hasInputs,
  };
}
