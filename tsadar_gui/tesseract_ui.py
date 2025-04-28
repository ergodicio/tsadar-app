from jax import config as jax_config


jax_config.update("jax_enable_x64", True)

import numpy as np
import boto3


import tqdm, optax, equinox as eqx, yaml
import plotly.graph_objects as go
import streamlit as st

from tsadar import ThomsonParams
from tsadar.core.modules.ts_params import get_filter_spec

from flatten_dict import flatten, unflatten


from tesseract_core import Tesseract


with open("./tesseract/1d-defaults.yaml", "r") as f:
    defaults = yaml.safe_load(f)

with open("./tesseract/1d-inputs.yaml", "r") as f:
    inputs = yaml.safe_load(f)

defaults = flatten(defaults)
defaults.update(flatten(inputs))
config = unflatten(defaults)

diff_wrt_to = ["ne", "Te", "amp1", "amp2", "lam"]
jac_outputs = ["electron_spectrum"]


def to_arr(x):
    """Convert a float to a numpy array."""
    return np.array(np.array(x, dtype=np.float64)).reshape(-1, 1)


def to_numpy(x: dict[str, float]) -> np.ndarray:
    """Convert the parameter dictionary to a numpy array."""
    return np.concatenate([[x[k] for k in diff_wrt_to]])


def to_dict(params: np.ndarray) -> dict:
    """Convert the numpy array to a parameter dictionary."""
    return {k: params[i] for i, k in enumerate(diff_wrt_to)}


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Squared Error."""
    mse = np.mean(np.square(pred - true))
    return mse


def grad_fn(parameters: np.ndarray, true_electron_spectrum: np.ndarray, tsadaract: Tesseract) -> np.ndarray:
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


def create_parameter_dict(_ts_params: ThomsonParams) -> dict:
    """Create a dictionary of parameters from the ThomsonParams object."""
    parameters = {
        "ne": _ts_params.electron.normed_ne[0],
        "Te": _ts_params.electron.normed_Te[0],
        "amp1": _ts_params.general.normed_amp1[0],
        "amp2": _ts_params.general.normed_amp2[0],
        "lam": _ts_params.general.normed_lam[0],
    }
    return parameters


def tesseract_ui(tesseract_url):

    # check if ecs service is running using boto3
    ecs = boto3.client("ecs")
    ecs_clusters = ecs.list_clusters()
    services = ecs.list_services(cluster=ecs_clusters["clusterArns"][0])
    for service in services["serviceArns"]:
        if "tess" in service:
            tsadaract_service = service

    # check if the service is running
    service_status = ecs.describe_services(cluster=ecs_clusters["clusterArns"][0], services=[tsadaract_service])
    if service_status["services"][0]["desiredCount"] == 0:
        st.warning("Tesseract service is not running. Please start the service.")

        if st.button("Start Service"):
            ecs.update_service(
                cluster=ecs_clusters["clusterArns"][0],
                service=tsadaract_service,
                desiredCount=1,
            )
            st.success("Service started. Please wait for the service to be ready.")

    elif service_status["services"][0]["desiredCount"] == 1 and service_status["services"][0]["runningCount"] == 0:
        st.warning("Tesseract service is launching. Please wait")

    else:
        if st.button("Stop Service"):
            ecs.update_service(
                cluster=ecs_clusters["clusterArns"][0],
                service=tsadaract_service,
                desiredCount=0,
            )
            st.success("Service stopped. Please refresh the page.")

        tsadaract = Tesseract(url=tesseract_url)
        # Sample random true parameters
        col1, col2 = st.columns(2)

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

        true_parameters = create_parameter_dict(true_ts_params)
        true_electron_spectrum = tsadaract.apply(true_parameters)["electron_spectrum"]

        true_fitted_params, _ = true_ts_params.get_fitted_params(config["parameters"])

        with col1:
            st.write("True parameters:")
            st.json(clean_dict(true_fitted_params))

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
        fit_parameters = create_parameter_dict(fit_ts_params)
        parameters_np = to_numpy(fit_parameters)

        electron_spectrum = tsadaract.apply(fit_parameters)["electron_spectrum"]

        # plot true electron spectrum in a plotly chart in streamlit
        with col2:
            st.write("Fitted parameters:")
            fit_param_holder = st.empty()
        fig_holder = st.empty()

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=true_electron_spectrum, mode="lines+markers", name="True Electron Spectrum"))
        fig.add_trace(go.Scatter(y=electron_spectrum, mode="lines+markers", name="Fit Electron Spectrum"))
        fig.update_layout(title="Electron Spectrum", xaxis_title="Wavelength", yaxis_title="Amplitude")
        fig_holder.plotly_chart(fig)

        learning_rate = st.number_input("Learning Rate", value=0.01, step=0.001, key="learning_rate")
        opt = optax.adam(learning_rate)

        diff_params, static_params = eqx.partition(
            fit_ts_params, filter_spec=get_filter_spec(cfg_params=config["parameters"], ts_params=fit_ts_params)
        )
        fit_parameters = create_parameter_dict(diff_params)
        opt_state = opt.init(fit_parameters)

        updated_fitted_parameters = get_fitted_params_for_ui(fit_parameters, diff_params, static_params)
        fit_param_holder.write("Estimated parameters:")
        fit_param_holder.json(updated_fitted_parameters)

        if st.button("Fit"):

            for i in (pbar := tqdm.tqdm(range(1000))):

                parameters_np = to_numpy(fit_parameters)
                electron_spectrum = tsadaract.apply(fit_parameters)["electron_spectrum"]
                loss, grad_loss = mse(electron_spectrum, true_electron_spectrum), grad_fn(
                    parameters_np, true_electron_spectrum, tsadaract
                )

                updates, opt_state = opt.update(grad_loss, opt_state)
                fit_parameters = eqx.apply_updates(fit_parameters, updates)
                pbar.set_description(f"Loss: {loss:.4f}")

                fig.data[1].y = electron_spectrum
                fig.data[1].name = f"Step {i+1}"
                fig.update_layout(
                    title=f"Electron Spectrum, Loss = {loss:.2e}", xaxis_title="Wavelength", yaxis_title="Amplitude"
                )
                fig_holder.plotly_chart(fig)

                fit_param_holder.write("Estimated parameters:")
                updated_fitted_parameters = get_fitted_params_for_ui(fit_parameters, diff_params, static_params)
                fit_param_holder.json(updated_fitted_parameters)


def get_fitted_params_for_ui(fit_parameters, diff_params, static_params):
    diff_params = eqx.tree_at(lambda x: x.electron.normed_ne, diff_params, to_arr(fit_parameters["ne"]))
    diff_params = eqx.tree_at(lambda x: x.electron.normed_Te, diff_params, to_arr(fit_parameters["Te"]))
    diff_params = eqx.tree_at(lambda x: x.general.normed_amp1, diff_params, to_arr(fit_parameters["amp1"]))
    diff_params = eqx.tree_at(lambda x: x.general.normed_amp2, diff_params, to_arr(fit_parameters["amp2"]))
    diff_params = eqx.tree_at(lambda x: x.general.normed_lam, diff_params, to_arr(fit_parameters["lam"]))
    fit_ts_params = eqx.combine(diff_params, static_params)
    updated_fitted_parameters, _ = fit_ts_params.get_fitted_params(config["parameters"])
    updated_fitted_parameters = clean_dict(updated_fitted_parameters)
    return updated_fitted_parameters


def clean_dict(d: dict) -> dict:
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = clean_dict(v)
        else:
            d[k] = round(float(np.squeeze(v)), 2)
    return d
