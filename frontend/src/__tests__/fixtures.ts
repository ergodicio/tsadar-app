/** Response fixtures shaped like the real API. */

export function runDetail(overrides: Record<string, unknown> = {}) {
  return {
    run_id: "run-abc",
    run_name: "shot-101675-scan",
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
    artifact_uri: "s3://public-ergodic-continuum/1/run-abc/artifacts",
    mlflow_run_url: "https://continuum.ergodic.io/experiments/#/experiments/1/runs/run-abc",
    config: { data: { shotnum: 101675, lineouts: { start: 800, end: 940 } } },
    config_flat: { "data.shotnum": "101675" },
    config_unflatten_error: null,
    tags: {},
    metrics: [
      { key: "overall loss", value: 12.5 },
      { key: "epoch loss", value: 13.1 },
      { key: "fit_time", value: 42 },
    ],
    artifacts: [
      { path: "binary", is_dir: true, size: null },
      { path: "binary/ele_fit_and_data.nc", is_dir: false, size: 4096 },
      { path: "plots/fit_and_data.png", is_dir: false, size: 2048 },
      { path: "csv/learned_parameters.csv", is_dir: false, size: 512 },
      // App-queued runs log a single merged config, so there is nothing to diff.
      { path: "config.yaml", is_dir: false, size: 900 },
    ],
    manifest: null,
    ...overrides,
  };
}

export function availability(overrides: Record<string, unknown> = {}) {
  return {
    kind: "one_d",
    supported: true,
    reason: null,
    message: null,
    spectra: [
      {
        which: "ele",
        path: "binary/ele_fit_and_data.nc",
        x_label: "Time (ps)",
        y_label: "Wavelength",
        lineout_count: 6,
        wavelength_count: 32,
        fields: ["data", "fit", "residual"],
      },
    ],
    profiles_available: true,
    sigmas_available: false,
    unavailable_fields: { irf: "IRF and noise components are not written to the netCDF datasets" },
    ...overrides,
  };
}

export const angularAvailability = () =>
  availability({
    kind: "angular",
    supported: false,
    reason: "angular_not_supported",
    message:
      "This is an angularly-resolved run. Interactive spectrogram, lineout and profile views are limited to 1D (time- or space-resolved) Thomson; the plot gallery for this run is still available.",
    spectra: [],
    profiles_available: false,
  });

export function spectrogram(overrides: Record<string, unknown> = {}) {
  return {
    which: "ele",
    field: "data",
    x_label: "Time (ps)",
    y_label: "Wavelength",
    x: [-100, -60, -20, 20, 60, 100],
    y: [520, 525, 530, 535],
    values: [
      [1, 2, 3, 4, 5, 6],
      [2, 3, 4, 5, 6, 7],
      [3, 4, 5, 6, 7, 8],
      [4, 5, 6, 7, 8, 9],
    ],
    full_shape: [6, 32],
    returned_shape: [6, 4],
    downsample_factors: { x: 1, wavelength: 8 },
    downsample_method: "mean",
    ...overrides,
  };
}

export function lineout(overrides: Record<string, unknown> = {}) {
  return {
    which: "ele",
    index: 0,
    lineout_count: 6,
    x_label: "Time (ps)",
    x_value: -100,
    y_label: "Wavelength",
    wavelength: [520, 525, 530, 535],
    data: [1, 2, 3, 4],
    fit: [0.9, 1.9, 2.9, 3.9],
    residual: [0.1, 0.1, 0.1, 0.1],
    components: {},
    components_unavailable: "IRF and noise components are not written to the netCDF datasets",
    ...overrides,
  };
}

export function profiles(overrides: Record<string, unknown> = {}) {
  return {
    x_label: "Time (ps)",
    x: [-100, -60, -20, 20, 60, 100],
    lineout_pixels: [800, 801, 802, 803, 804, 805],
    series: [
      { name: "Te_electron", values: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], sigma: null },
      { name: "ne_electron", values: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35], sigma: null },
    ],
    sigmas_available: false,
    ...overrides,
  };
}

export function metricHistory(key = "epoch loss") {
  return {
    key,
    points: [
      { step: 0, value: 20, timestamp: 1 },
      { step: 1, value: 15, timestamp: 2 },
      { step: 2, value: 12.5, timestamp: 3 },
    ],
  };
}
