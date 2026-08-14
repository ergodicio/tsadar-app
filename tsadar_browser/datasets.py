"""Reading and slicing the ``binary/*.nc`` analysis datasets.

This is the layer that lets the browser render from the real arrays instead of
the pre-rendered PNGs. It reads with xarray, downsamples server-side, and refuses
in a structured way whenever a run cannot honestly be served.

Scope: **1D Thomson only** (temporal and spatial/imaging). Angular runs write a
differently-shaped dataset and are out of scope -- see issue #37. Detecting them
matters more than it looks: an angular ``binary/fit_and_data.nc`` holds the same
two variables with the same dimensionality as a 1D ``ele_fit_and_data.nc``, so
treating one as a fallback for the other would quietly serve
angle-versus-wavelength data to a UI labelling its x-axis as time.

What the datasets actually contain, from ``tsadar/utils/plotting/plotters.py``:

- ``binary/ele_fit_and_data.nc`` / ``binary/ion_fit_and_data.nc`` -- variables
  ``fit`` and ``data`` only, dims ``(<x_label>, "Wavelength")``. The x dimension
  name is dynamic: ``Time (ps)`` or ``Radius (\\mum)``.
- ``binary/fit_and_data.nc`` -- the angular equivalent, x dim
  ``Scattering angle (degrees)``.
- ``sigmas.nc`` at the artifact **root** (not under ``binary/``), variables named
  ``<param>_<species>``.
- ``csv/learned_parameters.csv`` -- an unnamed index column, ``lineout pixel``,
  the x axis column, then one column per fitted parameter.

Residual is derived as ``data - fit``. IRF and noise components are genuinely
**not** in these files -- they exist only baked into the ``lineouts/``, ``best/``
and ``worst/`` PNGs -- so they are reported as unavailable rather than invented.
"""

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .gateway import MlflowGateway
from .schemas import (
    DatasetAvailability,
    DatasetKind,
    Lineout,
    ProfileSeries,
    Profiles,
    Spectrogram,
    SpectrumInfo,
    UnavailableReason,
)

logger = logging.getLogger(__name__)

#: 1D spectrum datasets, keyed by the ``which`` query value.
ONE_D_DATASETS = {
    "ele": "binary/ele_fit_and_data.nc",
    "ion": "binary/ion_fit_and_data.nc",
}

#: The angular dataset. Recognized so it can be refused, never served.
ANGULAR_DATASET = "binary/fit_and_data.nc"

LEARNED_PARAMS_CSV = "csv/learned_parameters.csv"

#: ``save_sigmas_params`` writes this to the artifact root, not under binary/.
SIGMAS_PATHS = ("sigmas.nc", "binary/sigma-params.nc")

WAVELENGTH_DIM = "Wavelength"
LINEOUT_PIXEL_COLUMN = "lineout pixel"

#: Substring identifying an angular x axis, as a second line of defense behind
#: the artifact-shape check.
ANGULAR_AXIS_MARKERS = ("angle", "degrees")

#: Fields the spectrogram endpoint can serve. ``residual`` is derived.
SPECTROGRAM_FIELDS = ("data", "fit", "residual")

#: Why IRF is not offered. Reported rather than silently omitted.
IRF_UNAVAILABLE = (
    "IRF and noise components are not written to the netCDF datasets -- plotters.py "
    "stores only 'fit' and 'data'. They exist only inside the pre-rendered lineout PNGs. "
    "See ergodicio/tsadar#116."
)

#: Default pixel budget. Deliberately below a realistic full-resolution
#: spectrogram (60 lineouts x 1024 wavelength pixels = 61440) so that, per #30,
#: full resolution is never shipped unless a client asks for it by raising
#: max_px. A heatmap panel is a few hundred pixels tall, so 1024 spectral points
#: are already oversampled for display.
DEFAULT_MAX_PX = 20_000

#: Significant digits kept on the wire. Full float repr costs roughly twice the
#: bytes for precision no plot can show; 6 digits is far beyond display
#: resolution but keeps a 60x256 spectrogram near 130 KB instead of 290 KB.
#: Clients needing exact stored values should fetch the .nc through the artifact
#: passthrough instead.
WIRE_SIGNIFICANT_DIGITS = 6


class DatasetUnavailable(Exception):
    """A run cannot be served, for a reason the frontend should act on.

    Carries a machine-readable ``reason`` so #32 can distinguish "angular run,
    interactive views not supported" from "no data found" -- the second reads as
    a bug when shown for the first.
    """

    def __init__(self, reason: UnavailableReason, message: str, status_code: int = 404):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.status_code = status_code


def _round_significant(array: np.ndarray, digits: int) -> np.ndarray:
    """Round to a number of significant digits, leaving zeros and non-finites alone.

    Significant digits rather than decimal places because spectra and fitted
    parameters span very different magnitudes.
    """
    result = np.array(array, dtype=float, copy=True)
    adjustable = np.isfinite(result) & (result != 0)
    if not adjustable.any():
        return result

    magnitude = np.floor(np.log10(np.abs(result[adjustable])))
    factor = np.power(10.0, digits - 1 - magnitude)
    result[adjustable] = np.round(result[adjustable] * factor) / factor
    return result


def _json_safe(values: np.ndarray, digits: int = WIRE_SIGNIFICANT_DIGITS) -> Any:
    """Convert an array to nested lists, JSON-safe and trimmed to ``digits``.

    JSON has no NaN or Infinity. Python's ``json.dumps`` emits bare ``NaN``,
    which is invalid JSON and makes browser ``JSON.parse`` throw, so gaps in the
    fit have to become ``null`` before serialization.
    """
    array = _round_significant(np.asarray(values, dtype=float), digits)
    return np.where(np.isfinite(array), array, None).tolist()


def _is_angular_axis(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in ANGULAR_AXIS_MARKERS)


class DatasetService:
    """Serves slices of a run's analysis datasets."""

    def __init__(self, gateway: MlflowGateway):
        self.gateway = gateway

    # -- discovery ------------------------------------------------------------

    def _artifact_paths(self, run_id: str) -> set[str]:
        return {entry.path for entry in self.gateway.list_artifacts(run_id) if not entry.is_dir}

    def classify(self, run_id: str, paths: set[str] | None = None) -> tuple[DatasetKind, list[str]]:
        """Return the run's dataset kind and which 1D spectra it has.

        Artifact shape is the authority here, not the logged ``spectype`` param --
        that is written before the fit runs and can disagree with reality.
        """
        paths = self._artifact_paths(run_id) if paths is None else paths

        present = [which for which, path in ONE_D_DATASETS.items() if path in paths]
        if present:
            return DatasetKind.one_d, present
        if ANGULAR_DATASET in paths:
            return DatasetKind.angular, []
        return DatasetKind.unknown, []

    def describe(self, run_id: str) -> DatasetAvailability:
        """Report what can be rendered for this run, and why not when it can't.

        #32 calls this to choose a layout, so it must answer for every run --
        including old ones and angular ones -- rather than erroring.
        """
        paths = self._artifact_paths(run_id)
        kind, present = self.classify(run_id, paths)

        if kind is DatasetKind.angular:
            return DatasetAvailability(
                kind=kind,
                supported=False,
                reason=UnavailableReason.angular_not_supported,
                message=(
                    "This is an angularly-resolved run. Interactive spectrogram, lineout and "
                    "profile views are limited to 1D (time- or space-resolved) Thomson; the "
                    "plot gallery for this run is still available."
                ),
                profiles_available=False,
                sigmas_available=False,
                unavailable_fields={"irf": IRF_UNAVAILABLE},
            )

        if kind is DatasetKind.unknown:
            return DatasetAvailability(
                kind=kind,
                supported=False,
                reason=UnavailableReason.dataset_missing,
                message=(
                    "This run has no readable fit/data datasets, which is expected for runs "
                    "predating the artifact contract. Use the plot gallery instead."
                ),
                profiles_available=LEARNED_PARAMS_CSV in paths,
                sigmas_available=any(path in paths for path in SIGMAS_PATHS),
                unavailable_fields={"irf": IRF_UNAVAILABLE},
            )

        spectra: list[SpectrumInfo] = []
        for which in present:
            try:
                spectra.append(self._describe_spectrum(run_id, which))
            except DatasetUnavailable as exc:
                logger.warning("could not describe %s spectrum for run %s: %s", which, run_id, exc)

        return DatasetAvailability(
            kind=kind,
            supported=bool(spectra),
            reason=None if spectra else UnavailableReason.dataset_unreadable,
            message=None if spectra else "The datasets are present but could not be opened.",
            spectra=spectra,
            profiles_available=LEARNED_PARAMS_CSV in paths,
            sigmas_available=any(path in paths for path in SIGMAS_PATHS),
            unavailable_fields={"irf": IRF_UNAVAILABLE},
        )

    def _describe_spectrum(self, run_id: str, which: str) -> SpectrumInfo:
        with self._open(run_id, which) as dataset:
            x_label, y_label = self._axes(dataset)
            return SpectrumInfo(
                which=which,
                path=ONE_D_DATASETS[which],
                x_label=x_label,
                y_label=y_label,
                lineout_count=int(dataset.sizes[x_label]),
                wavelength_count=int(dataset.sizes[y_label]),
                fields=list(SPECTROGRAM_FIELDS),
            )

    # -- opening --------------------------------------------------------------

    def _open(self, run_id: str, which: str) -> xr.Dataset:
        if which not in ONE_D_DATASETS:
            raise DatasetUnavailable(
                UnavailableReason.dataset_missing,
                f"unknown spectrum {which!r}; expected one of {sorted(ONE_D_DATASETS)}",
                status_code=400,
            )

        path = ONE_D_DATASETS[which]
        try:
            local = self.gateway.download_artifact(run_id, path)
        except Exception as exc:  # noqa: BLE001 - absent artifact is an expected outcome
            raise DatasetUnavailable(
                UnavailableReason.dataset_missing,
                f"run has no {path}: {exc}",
            ) from exc

        try:
            dataset = xr.open_dataset(local)
        except Exception as exc:  # noqa: BLE001 - corrupt or non-netCDF payload
            raise DatasetUnavailable(
                UnavailableReason.dataset_unreadable,
                f"could not open {path} as a netCDF dataset: {exc}",
            ) from exc

        missing = {"fit", "data"} - set(dataset.data_vars)
        if missing:
            dataset.close()
            raise DatasetUnavailable(
                UnavailableReason.unexpected_schema,
                f"{path} is missing expected variables: {sorted(missing)}",
            )
        return dataset

    @staticmethod
    def _axes(dataset: xr.Dataset) -> tuple[str, str]:
        """Return ``(x_dim, wavelength_dim)``, rejecting angular data.

        The x dimension name is dynamic, so it is whichever dim is not the
        wavelength axis rather than a fixed string.
        """
        dims = list(dataset["data"].dims)
        if len(dims) != 2:
            raise DatasetUnavailable(
                UnavailableReason.unexpected_schema,
                f"expected a 2D fit/data array, got dims {dims}",
            )

        y_label = WAVELENGTH_DIM if WAVELENGTH_DIM in dims else str(dims[1])
        x_label = str(next(dim for dim in dims if dim != y_label))

        if _is_angular_axis(x_label):
            # Reached only if a 1D-named file carries an angular axis; the
            # artifact-shape check normally catches this first.
            raise DatasetUnavailable(
                UnavailableReason.angular_not_supported,
                f"dataset x axis {x_label!r} is angular; interactive views are 1D only",
                status_code=409,
            )
        return x_label, y_label

    @staticmethod
    def _field(dataset: xr.Dataset, field: str) -> xr.DataArray:
        if field == "residual":
            # Not stored: plotters.py writes only fit and data.
            return dataset["data"] - dataset["fit"]
        if field not in SPECTROGRAM_FIELDS:
            raise DatasetUnavailable(
                UnavailableReason.field_unavailable,
                f"unknown field {field!r}; expected one of {list(SPECTROGRAM_FIELDS)}",
                status_code=400,
            )
        return dataset[field]

    # -- spectrogram ----------------------------------------------------------

    def spectrogram(self, run_id: str, which: str, field: str, max_px: int | None) -> Spectrogram:
        self._require_one_d(run_id)

        with self._open(run_id, which) as dataset:
            x_label, y_label = self._axes(dataset)
            array = self._field(dataset, field)

            full_shape = [int(array.sizes[x_label]), int(array.sizes[y_label])]
            reduced, factors = self._downsample(array, x_label, y_label, max_px)

            # Transpose to (wavelength, x) so `values` is directly usable as a
            # Plotly heatmap `z`, which is indexed [y][x].
            oriented = reduced.transpose(y_label, x_label)

            return Spectrogram(
                which=which,
                field=field,
                x_label=x_label,
                y_label=y_label,
                x=_json_safe(reduced.coords[x_label].values),
                y=_json_safe(reduced.coords[y_label].values),
                values=_json_safe(oriented.values),
                full_shape=full_shape,
                returned_shape=[int(reduced.sizes[x_label]), int(reduced.sizes[y_label])],
                downsample_factors={"x": factors[0], "wavelength": factors[1]},
                downsample_method="mean" if max(factors) > 1 else "none",
            )

    @staticmethod
    def _downsample(
        array: xr.DataArray, x_label: str, y_label: str, max_px: int | None
    ) -> tuple[xr.DataArray, tuple[int, int]]:
        """Block-average down to a pixel budget, sparing the lineout axis.

        Wavelength is coarsened first and the lineout axis only if that is not
        enough. Each lineout is a distinct fit with its own parameters and is what
        the scrubber in #32 steps through, so throwing lineouts away to save
        bytes would cost far more than reducing spectral resolution. Mean rather
        than decimation, so a narrow spectral feature is attenuated rather than
        skipped entirely.
        """
        if max_px is None or array.size <= max_px:
            return array, (1, 1)

        n_x = int(array.sizes[x_label])
        n_y = int(array.sizes[y_label])

        y_factor = min(n_y, max(1, math.ceil(n_x * n_y / max_px)))

        # Integer division means the first estimate can land just over budget
        # (60x1024 to 2000 gives 60x34 = 2040). Tighten wavelength the rest of
        # the way before giving up any lineouts, since one more step of spectral
        # averaging is far cheaper than losing whole fits.
        while y_factor < n_y and n_x * math.ceil(n_y / y_factor) > max_px:
            y_factor += 1

        x_factor = 1
        if n_x * math.ceil(n_y / y_factor) > max_px:
            x_factor = max(1, math.ceil(n_x * math.ceil(n_y / y_factor) / max_px))

        windows = {dim: factor for dim, factor in ((x_label, x_factor), (y_label, y_factor)) if factor > 1}
        if not windows:
            return array, (1, 1)

        # boundary="trim" would silently drop a partial trailing block; "pad"
        # keeps it and averages over the values that exist.
        reduced = array.coarsen(windows, boundary="pad").mean()
        return reduced, (x_factor, y_factor)

    # -- lineout --------------------------------------------------------------

    def lineout(self, run_id: str, which: str, index: int) -> Lineout:
        self._require_one_d(run_id)

        with self._open(run_id, which) as dataset:
            x_label, y_label = self._axes(dataset)
            count = int(dataset.sizes[x_label])
            if not -count <= index < count:
                raise DatasetUnavailable(
                    UnavailableReason.index_out_of_range,
                    f"lineout index {index} out of range; run has {count} lineouts",
                    status_code=400,
                )

            data = dataset["data"].isel({x_label: index})
            fit = dataset["fit"].isel({x_label: index})

            return Lineout(
                which=which,
                index=index if index >= 0 else count + index,
                lineout_count=count,
                x_label=x_label,
                x_value=float(dataset.coords[x_label].values[index]),
                y_label=y_label,
                wavelength=_json_safe(data.coords[y_label].values),
                data=_json_safe(data.values),
                fit=_json_safe(fit.values),
                residual=_json_safe((data - fit).values),
                components={},
                components_unavailable=IRF_UNAVAILABLE,
            )

    # -- profiles -------------------------------------------------------------

    def profiles(self, run_id: str) -> Profiles:
        self._require_one_d(run_id)

        try:
            local = self.gateway.download_artifact(run_id, LEARNED_PARAMS_CSV)
        except Exception as exc:  # noqa: BLE001
            raise DatasetUnavailable(
                UnavailableReason.dataset_missing,
                f"run has no {LEARNED_PARAMS_CSV}: {exc}",
            ) from exc

        try:
            frame = pd.read_csv(local)
        except Exception as exc:  # noqa: BLE001
            raise DatasetUnavailable(
                UnavailableReason.dataset_unreadable,
                f"could not parse {LEARNED_PARAMS_CSV}: {exc}",
            ) from exc

        # to_csv writes the DataFrame index as a leading unnamed column.
        frame = frame.drop(columns=[column for column in frame.columns if str(column).startswith("Unnamed")])

        lineout_pixels = None
        if LINEOUT_PIXEL_COLUMN in frame.columns:
            lineout_pixels = [int(value) for value in frame.pop(LINEOUT_PIXEL_COLUMN).to_numpy()]

        x_label = self._profile_axis_column(frame)
        if x_label is None:
            # Angular runs skip the axis insert entirely, so there is nothing to
            # plot profiles against. _require_one_d should have caught it already.
            raise DatasetUnavailable(
                UnavailableReason.unexpected_schema,
                f"{LEARNED_PARAMS_CSV} has no lineout axis column, so profiles have no x axis",
            )

        x_values = frame.pop(x_label).to_numpy()
        sigmas = self._read_sigmas(run_id)

        series = [
            ProfileSeries(
                name=str(column),
                values=_json_safe(frame[column].to_numpy()),
                sigma=_json_safe(sigmas[str(column)]) if str(column) in sigmas else None,
            )
            for column in frame.columns
        ]

        return Profiles(
            x_label=x_label,
            x=_json_safe(x_values),
            lineout_pixels=lineout_pixels,
            series=series,
            sigmas_available=bool(sigmas),
        )

    @staticmethod
    def _profile_axis_column(frame: pd.DataFrame) -> str | None:
        """Find the lineout-axis column among the parameter columns.

        Axis labels carry units in parentheses (``Time (ps)``, ``Radius (\\mum)``)
        while parameter columns are ``<param>_<species>``, so the parenthesis is
        the discriminator.
        """
        return next((str(column) for column in frame.columns if "(" in str(column)), None)

    def _read_sigmas(self, run_id: str) -> dict[str, np.ndarray]:
        """Read per-parameter uncertainties, if the run computed any.

        ``calc_sigmas`` is off by default, so absence is normal, not an error.
        """
        for path in SIGMAS_PATHS:
            try:
                local = self.gateway.download_artifact(run_id, path)
            except Exception:  # noqa: BLE001 - try the next candidate location
                continue
            try:
                with xr.open_dataset(local) as dataset:
                    return {str(name): array.values for name, array in dataset.data_vars.items()}
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not read sigmas from %s for run %s: %s", path, run_id, exc)
        return {}

    # -- guards ---------------------------------------------------------------

    def _require_one_d(self, run_id: str) -> None:
        """Refuse angular and pre-contract runs before touching any dataset."""
        kind, present = self.classify(run_id)

        if kind is DatasetKind.angular:
            raise DatasetUnavailable(
                UnavailableReason.angular_not_supported,
                (
                    "This is an angularly-resolved run. Interactive views are limited to 1D "
                    "(time- or space-resolved) Thomson; use the plot gallery for this run."
                ),
                status_code=409,
            )
        if kind is DatasetKind.unknown and not present:
            raise DatasetUnavailable(
                UnavailableReason.dataset_missing,
                "This run has no readable fit/data datasets; use the plot gallery instead.",
            )


def local_dataset_path(gateway: MlflowGateway, run_id: str, which: str) -> Path:  # pragma: no cover
    """Convenience for debugging: where a spectrum dataset landed on disk."""
    return gateway.download_artifact(run_id, ONE_D_DATASETS[which])
