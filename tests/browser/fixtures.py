"""Builders for realistic tsadar analysis artifacts.

These mirror what ``tsadar/utils/plotting/plotters.py`` actually writes, because
the whole point of #30 is reading those files. In particular:

- ``fit`` and ``data`` are the ONLY variables; residual is derived and IRF is
  absent (``plotters.py`` lines 430, 477, 506).
- dims are ``(<x_label>, "Wavelength")`` with a dynamic x name.
- angular runs write ``binary/fit_and_data.nc`` with an angular x axis, and are
  otherwise indistinguishable in shape from a 1D dataset.
- ``learned_parameters.csv`` gets an unnamed index column from ``to_csv``, then
  ``lineout pixel``, then the axis column, then the parameters.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

TEMPORAL_AXIS = "Time (ps)"
SPATIAL_AXIS = r"Radius (\mum)"
ANGULAR_AXIS = "Scattering angle (degrees)"
WAVELENGTH = "Wavelength"


def spectrum_dataset(
    n_lineouts: int = 6,
    n_wavelength: int = 32,
    x_label: str = TEMPORAL_AXIS,
    with_nan: bool = False,
) -> xr.Dataset:
    """A fit/data dataset shaped exactly like plotters.plot_ts_data writes."""
    x = np.linspace(-100.0, 100.0, n_lineouts)
    wavelength = np.linspace(520.0, 540.0, n_wavelength)
    coords = (x_label, x), (WAVELENGTH, wavelength)

    rng = np.random.default_rng(1234)
    data = rng.random((n_lineouts, n_wavelength)) + 1.0
    fit = data * 0.9

    if with_nan:
        # A real fit can leave gaps; these must serialize as null, not NaN.
        data[0, 0] = np.nan
        fit[1, 2] = np.inf

    return xr.Dataset(
        {
            "fit": xr.DataArray(fit, coords=coords),
            "data": xr.DataArray(data, coords=coords),
        }
    )


def write_spectrum(directory: Path, name: str, **kwargs) -> bytes:
    """Serialize a spectrum dataset to netCDF bytes."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    spectrum_dataset(**kwargs).to_netcdf(path)
    return path.read_bytes()


def learned_parameters_csv(
    n_lineouts: int = 6,
    x_label: str = TEMPORAL_AXIS,
    include_axis: bool = True,
) -> bytes:
    """learned_parameters.csv as get_final_params writes it.

    ``include_axis=False`` reproduces the angular case, where the
    ``spectype != "angular_full"`` branch skips both inserts so there is no
    lineout axis to plot profiles against.
    """
    frame = pd.DataFrame(
        {
            "Te_electron": np.linspace(0.4, 0.9, n_lineouts),
            "ne_electron": np.linspace(0.1, 0.3, n_lineouts),
            "amp1_general": np.linspace(1.0, 1.4, n_lineouts),
        }
    )
    if include_axis:
        frame.insert(0, x_label, np.linspace(-100.0, 100.0, n_lineouts))
        frame.insert(0, "lineout pixel", np.arange(800, 800 + n_lineouts))

    return frame.to_csv(index=True).encode()


def sigmas_netcdf(directory: Path, n_lineouts: int = 6, x_label: str = TEMPORAL_AXIS) -> bytes:
    """sigmas.nc as save_sigmas_params writes it -- at the artifact ROOT.

    Variables are named ``<param>_<species>``, matching the CSV columns.
    """
    directory.mkdir(parents=True, exist_ok=True)
    coords = ((x_label, np.linspace(-100.0, 100.0, n_lineouts)),)
    dataset = xr.Dataset(
        {
            "Te_electron": xr.DataArray(np.full(n_lineouts, 0.05), coords=coords),
            "ne_electron": xr.DataArray(np.full(n_lineouts, 0.01), coords=coords),
        }
    )
    path = directory / "sigmas.nc"
    dataset.to_netcdf(path)
    return path.read_bytes()
