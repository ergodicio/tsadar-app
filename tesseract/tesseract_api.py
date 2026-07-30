from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from typing import Any

import equinox as eqx
import numpy as np
import yaml
from flatten_dict import flatten, unflatten
from jax import numpy as jnp
from pydantic import BaseModel, Field
from tsadar import ThomsonParams, ThomsonScatteringDiagnostic, get_scattering_angles
from tsadar.core.modules.ts_params import get_act_and_inv_act

from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.jax_recipes import (
    jax_abstract_eval,
    jax_apply,
    jax_jacobian,
    jax_jvp,
    jax_vjp,
)

with open("1d-defaults.yaml") as fi:
    defaults = yaml.safe_load(fi)

with open("1d-inputs.yaml") as fi:
    inputs = yaml.safe_load(fi)

defaults = flatten(defaults)
defaults.update(flatten(inputs))
config = unflatten(defaults)

# TODO: this block is duplicated at ~16 sites in tsadar (forward/calc_series.py, the
# forward/inverse test suites, bench_*.py). It belongs upstream as
# tsadar.forward.prepare_forward_config(config) -> (config, sas).
# Note lamrangE/lamrangI have a different source in inverse mode, where
# utils/process/prepare.py takes them from the calibrated data axis instead.
config["other"]["lamrangE"] = [
    config["data"]["fit_rng"]["forward_epw_start"],
    config["data"]["fit_rng"]["forward_epw_end"],
]
config["other"]["lamrangI"] = [
    config["data"]["fit_rng"]["forward_iaw_start"],
    config["data"]["fit_rng"]["forward_iaw_end"],
]
config["other"]["npts"] = int(config["other"]["CCDsize"][1] * config["other"]["points_per_pixel"])

sas = get_scattering_angles(config)
ts_params = ThomsonParams(config["parameters"], num_params=1, batch=True, activate=True)
ts_diag = ThomsonScatteringDiagnostic(config, sas)

# The forward model is evaluated without data, so the batch only has to carry shapes.
dummy_batch = {
    "i_data": np.array([1]),
    "e_data": np.array([1]),
    "noise_e": np.array([0]),
    "noise_i": np.array([0]),
    "e_amps": np.array([1]),
    "i_amps": np.array([1]),
}

# Which ThomsonParams submodule owns each parameter we expose.
PARAMS = {"ne": "electron", "Te": "electron", "amp1": "general", "amp2": "general", "lam": "general"}

# tsadar stores parameters in a rescaled, optionally logit-transformed space and undoes
# that as `act_fun(normed) * scale + shift` (ts_params.ElectronParams.get_unnormed_params).
# Writing to `normed_*` directly would make this Tesseract's inputs depend on the deck's
# lb/ub *and* on its `active` flags, which decide whether act_fun is a sigmoid or the
# identity. So we invert the transform here and take physical units on the wire instead.
#
# We deliberately do NOT reuse tsadar's `inv_act_fun`. Its stabilized form,
# log(1e-2 + x / (1 - x + 1e-2)), is not the exact inverse of `sigmoid`: round-tripping
# the midpoint gives 0.49759 rather than 0.5, an error of ~0.24% of the parameter's
# span. (tsadar carries that error itself -- the physical value it uses is not quite the
# deck's `val` -- but a Tesseract whose inputs are declared physical should honour them
# exactly, so we use the true inverse instead.)
def _exact_inverse(act_fun):
    """Exact inverse of the activation `get_act_and_inv_act` paired with this config."""
    if act_fun(0.0) == 0.0:  # identity: parameter is not activated
        return lambda x: x
    return lambda x: jnp.log(x) - jnp.log1p(-x)  # logit, the true inverse of sigmoid


_norm = {}
for _name, _submodule in PARAMS.items():
    _cfg = config["parameters"][_submodule][_name]
    _act_fun, _ = get_act_and_inv_act(_cfg, activate=True)
    _norm[_name] = (_cfg["lb"], _cfg["ub"] - _cfg["lb"], _exact_inverse(_act_fun))


def to_normed(name: str, physical) -> jnp.ndarray:
    """Map a physical parameter value into the normalized space tsadar expects.

    The logit diverges at the bounds and is nan outside them, so a value at or beyond
    lb/ub yields a non-finite spectrum rather than an error. `check_bounds` guards the
    apply endpoint against that.
    """
    shift, scale, inv_act = _norm[name]
    normed = inv_act((jnp.asarray(physical, dtype=jnp.float64) - shift) / scale)
    # ThomsonParams(num_params=1, batch=True) builds these as shape (1,); the diagnostic
    # is evaluated with a trailing singleton axis, matching the previous implementation.
    return normed.reshape(-1, 1)


def check_bounds(inputs: "InputSchema") -> None:
    """Raise if any input maps to a non-finite normalized value.

    Tested via `to_normed` itself rather than by comparing against lb/ub, so the guard
    cannot drift from the transform. Note the bounds are exclusive for an activated
    parameter -- the logit is +-inf exactly at lb/ub -- but inclusive for one that is
    not activated, where the transform is the identity.
    """
    for name, (lb, span, _) in _norm.items():
        value = float(np.asarray(getattr(inputs, name)))
        if not np.all(np.isfinite(np.asarray(to_normed(name, value)))):
            raise ValueError(
                f"{name}={value} does not map to a finite normalized value; it must lie "
                f"within the deck bounds ({lb}, {lb + span})"
            )


def _describe(name: str, what: str) -> str:
    lb, span, _ = _norm[name]
    return f"{what} (physical units; deck bounds [{lb}, {lb + span}])"


class InputSchema(BaseModel):
    ne: Differentiable[Float64] = Field(description=_describe("ne", "electron density"))
    Te: Differentiable[Float64] = Field(description=_describe("Te", "electron temperature"))
    amp1: Differentiable[Float64] = Field(description=_describe("amp1", "amplitude 1"))
    amp2: Differentiable[Float64] = Field(description=_describe("amp2", "amplitude 2"))
    lam: Differentiable[Float64] = Field(description=_describe("lam", "central wavelength"))


class OutputSchema(BaseModel):
    electron_spectrum: Differentiable[Array[(None,), Float64]] = Field(description="electron spectrum")


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    # Scatter the exposed parameters into the parameter tree. Everything else in
    # `ts_params` is closed over as a constant, so no partition/combine is needed and
    # differentiability is decided solely by `Differentiable[...]` in InputSchema.
    params = eqx.tree_at(
        lambda p: (
            p.electron.normed_ne,
            p.electron.normed_Te,
            p.general.normed_amp1,
            p.general.normed_amp2,
            p.general.normed_lam,
        ),
        ts_params,
        replace=tuple(to_normed(name, inputs[name]) for name in PARAMS),
    )
    e_spec, _, _, _ = ts_diag(params, dummy_batch)
    # Squeeze inside the jit so abstract_eval derives the same shape apply returns.
    return {"electron_spectrum": jnp.squeeze(e_spec)}


def apply(inputs: InputSchema) -> OutputSchema:
    check_bounds(inputs)
    return OutputSchema(**jax_apply(apply_jit, inputs))


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    return jax_jacobian(apply_jit, inputs, jac_inputs, jac_outputs)


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jax_jvp(apply_jit, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return jax_vjp(apply_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)


def abstract_eval(abstract_inputs):
    return jax_abstract_eval(apply_jit, abstract_inputs)
