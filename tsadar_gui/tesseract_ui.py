from jax import config as jax_config


jax_config.update("jax_enable_x64", True)

import numpy as np
import os
from matplotlib import pyplot as plt
from numpy import ndarray

# from scipy.optimize import minimize
import tqdm, optax, equinox as eqx, yaml
import plotly.graph_objects as go
import streamlit as st

from tsadar import ThomsonParams
from tsadar.core.modules.ts_params import get_filter_spec
from flatten_dict import flatten, unflatten


from tesseract_core import Tesseract

# if DEBUG:
tesseract_url = "http://localhost:54294"  # os.environ["TESSERACT_URI"]  # "http://localhost:54294"  # Change this to the correct address

tsadaract = Tesseract(url=tesseract_url)

with open("./tesseract/1d-defaults.yaml", "r") as f:
    defaults = yaml.safe_load(f)

with open("./tesseract/1d-inputs.yaml", "r") as f:
    inputs = yaml.safe_load(f)

defaults = flatten(defaults)
defaults.update(flatten(inputs))
config = unflatten(defaults)

diff_wrt_to = ["ne", "Te", "amp1", "amp2", "lam"]
jac_outputs = ["electron_spectrum"]


def to_numpy(x: dict[str, float]) -> np.ndarray:
    """Convert the parameter dictionary to a numpy array."""
    return np.concatenate([[x[k] for k in diff_wrt_to]])


def to_dict(params: ndarray) -> dict:
    """Convert the numpy array to a parameter dictionary."""
    return {k: params[i] for i, k in enumerate(diff_wrt_to)}


def mse(pred: ndarray, true: ndarray) -> float:
    """Mean Squared Error."""
    mse = np.mean(np.square(pred - true))
    return mse


def grad_fn(parameters: np.ndarray, true_electron_spectrum: np.ndarray) -> np.ndarray:
    """Compute the gradient of the MSE loss function with respect to the parameters."""
    # Compute the gradient

    jacobian = tsadaract.jacobian(to_dict(parameters), diff_wrt_to, jac_outputs)["electron_spectrum"]

    # Compute the primal
    electron_spectrum = tsadaract.apply(to_dict(parameters))["electron_spectrum"]

    # Propagate the gradient through the model by differentiating the mse function
    error = electron_spectrum - true_electron_spectrum
    grad = {}
    for k in diff_wrt_to:
        grad[k] = 2 * np.mean(jacobian[k] * error)

    return grad  # to_numpy(grad)


def tesseract_ui():
    # Sample random true parameters
    rng = np.random.default_rng()
    true_ne = rng.uniform(0.1, 0.7)
    true_Te = rng.uniform(0.5, 1.5)
    true_amp1 = rng.uniform(0.5, 2.5)
    true_amp2 = rng.uniform(0.5, 2.5)
    true_lam = rng.uniform(525, 527)

    config["parameters"]["electron"]["ne"]["val"] = true_ne
    config["parameters"]["electron"]["Te"]["val"] = true_Te
    config["parameters"]["general"]["amp1"]["val"] = true_amp1
    config["parameters"]["general"]["amp2"]["val"] = true_amp2
    config["parameters"]["general"]["lam"]["val"] = true_lam
    true_ts_params = ThomsonParams(config["parameters"], num_params=1, batch=True, activate=True)

    true_parameters = {
        "ne": true_ts_params.electron.normed_ne[0],
        "Te": true_ts_params.electron.normed_Te[0],
        "amp1": true_ts_params.general.normed_amp1[0],
        "amp2": true_ts_params.general.normed_amp2[0],
        "lam": true_ts_params.general.normed_lam[0],
    }
    true_electron_spectrum = tsadaract.apply(true_parameters)["electron_spectrum"]

    st.write("True parameters:")
    st.json(true_parameters)
    # st.write(f"ne: {true_ne:.2f}")
    # st.write(f"Te: {true_Te:.2f}")
    # st.write(f"amp1: {true_amp1:.2f}")
    # st.write(f"amp2: {true_amp2:.2f}")
    # st.write(f"lam: {true_lam:.2f}")

    # create an initial guess for the parameters
    this_rng = np.random.default_rng()
    init_ne = this_rng.uniform(0.1, 0.7)
    init_Te = this_rng.uniform(0.5, 1.5)
    init_amp1 = this_rng.uniform(0.5, 2.5)
    init_amp2 = this_rng.uniform(0.5, 2.5)
    init_lam = this_rng.uniform(525, 527)

    config["parameters"]["electron"]["ne"]["val"] = init_ne
    config["parameters"]["electron"]["Te"]["val"] = init_Te
    config["parameters"]["general"]["amp1"]["val"] = init_amp1
    config["parameters"]["general"]["amp2"]["val"] = init_amp2
    config["parameters"]["general"]["lam"]["val"] = init_lam

    fit_ts_params = ThomsonParams(config["parameters"], num_params=1, batch=True, activate=True)

    fit_parameters = {
        "ne": fit_ts_params.electron.normed_ne[0],
        "Te": fit_ts_params.electron.normed_Te[0],
        "amp1": fit_ts_params.general.normed_amp1[0],
        "amp2": fit_ts_params.general.normed_amp2[0],
        "lam": fit_ts_params.general.normed_lam[0],
    }

    parameters_np = to_numpy(fit_parameters)

    electron_spectrum = tsadaract.apply(fit_parameters)["electron_spectrum"]

    # plot true electron spectrum in a plotly chart in streamlit
    fig_holder = st.empty()
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=true_electron_spectrum, mode="lines+markers", name="True Electron Spectrum"))
    fig.add_trace(go.Scatter(y=electron_spectrum, mode="lines+markers", name="Fit Electron Spectrum"))
    print(fig.data)
    fig.update_layout(title="Electron Spectrum", xaxis_title="Wavelength", yaxis_title="Amplitude")
    fig_holder.plotly_chart(fig)

    opt = optax.adam(0.05)
    diff_params, static_params = eqx.partition(
        fit_ts_params, filter_spec=get_filter_spec(cfg_params=config["parameters"], ts_params=fit_ts_params)
    )

    opt_state = opt.init(fit_parameters)

    if st.button("Fit"):
        for i in (pbar := tqdm.tqdm(range(1000))):
            parameters_np = to_numpy(fit_parameters)
            electron_spectrum = tsadaract.apply(fit_parameters)["electron_spectrum"]
            loss, grad_loss = mse(electron_spectrum, true_electron_spectrum), grad_fn(
                parameters_np, true_electron_spectrum
            )

            updates, opt_state = opt.update(grad_loss, opt_state)
            fit_parameters = eqx.apply_updates(fit_parameters, updates)
            pbar.set_description(f"Loss: {loss:.4f}")

            if i % 1 == 0:
                # fig.add_trace(go.Scatter(y=electron_spectrum, mode="lines+markers", name=f"Step {i+1}"))
                fig.data[1].y = electron_spectrum
                fig.data[1].name = f"Step {i+1}"
                # fig.update_layout(title="Electron Spectrum", xaxis_title="Wavelength", yaxis_title="Value")
                fig_holder.plotly_chart(fig)
