import boto3
import numpy as np
import optax
import plotly.graph_objects as go
import requests
import streamlit as st
import tqdm

from tesseract_core import Tesseract


PARAMS = ["ne", "Te", "amp1", "amp2", "lam"]
JAC_OUTPUTS = ["electron_spectrum"]

# Physically realistic OMEGA conditions, used to draw both the synthetic truth and the
# initial guess. These are deliberately *not* the deck bounds: the bounds are validity
# limits for the model, so sampling ne from the deck's (0.001, 1.0) would generate test
# cases no experiment would produce. `check_sampling_ranges` asserts these sit strictly
# inside the bounds the Tesseract declares, so a deck change cannot silently invalidate
# them.
SAMPLING_RANGES = {
    "ne": (0.1, 0.7),  # 1e20 cm^-3
    # Te and lam are inset from the deck's [0.001, 1.5] and [525, 527]. Those bounds are
    # exclusive, and sampling hard against one puts the logit somewhere badly
    # conditioned even when it does not land on the bound exactly.
    "Te": (0.5, 1.4),  # keV
    "amp1": (0.5, 2.5),
    "amp2": (0.5, 2.5),
    "lam": (525.2, 526.8),  # nm
}


@st.cache_data(ttl=3600)
def fetch_bounds(tesseract_url: str) -> dict[str, tuple[float, float]]:
    """Read each parameter's deck bounds off the Tesseract's OpenAPI document.

    The bounds live in the tsadar deck and are declared on `InputSchema`, so this is the
    only place the GUI learns them -- it does not keep its own copy to drift.

    They are read by name rather than as JSON Schema constraints because pydantic puts
    them on tesseract-core's encoded-array wrapper, which serialises them as `gt`/`lt`
    (or `ge`/`le`) rather than as `exclusiveMinimum`/`exclusiveMaximum`.
    """
    response = requests.get(f"{tesseract_url}/openapi.json", timeout=10)
    response.raise_for_status()
    properties = response.json()["components"]["schemas"]["Apply_InputSchema"]["properties"]

    bounds = {}
    for name in PARAMS:
        spec = properties[name]
        lower = spec.get("gt", spec.get("ge"))
        upper = spec.get("lt", spec.get("le"))
        if lower is None or upper is None:
            raise RuntimeError(
                f"The Tesseract at {tesseract_url} does not declare bounds for {name!r}. "
                "It predates the InputSchema that declares them; rebuild it."
            )
        bounds[name] = (lower, upper)
    return bounds


def check_sampling_ranges(bounds: dict[str, tuple[float, float]]) -> None:
    """Fail loudly if the deck has moved out from under `SAMPLING_RANGES`."""
    for name, (low, high) in SAMPLING_RANGES.items():
        lower, upper = bounds[name]
        if not lower < low < high < upper:
            raise RuntimeError(
                f"The sampling range for {name} {(low, high)} is not strictly inside the "
                f"deck bounds {(lower, upper)}. Either the deck changed or the range is "
                "wrong; both need a human."
            )


def sample_parameters(rng: np.random.Generator) -> dict[str, float]:
    """Draw a physically plausible parameter set."""
    return {name: float(rng.uniform(*SAMPLING_RANGES[name])) for name in PARAMS}


# The Tesseract takes physical parameters and rejects anything outside the deck bounds,
# so a plain gradient step on them would need projecting back into the box. We optimize
# in an unconstrained space instead and map through a sigmoid, which keeps every iterate
# strictly inside the bounds for free and is better conditioned near them. This is the
# same reparameterization tsadar applies internally; it lives here now because the
# Tesseract's own interface is physical.
def to_unconstrained(
    physical: dict[str, float], bounds: dict[str, tuple[float, float]]
) -> dict[str, float]:
    """Map physical parameters onto the whole real line via the logit."""
    unconstrained = {}
    for name, value in physical.items():
        lower, upper = bounds[name]
        fraction = (value - lower) / (upper - lower)
        unconstrained[name] = float(np.log(fraction) - np.log1p(-fraction))
    return unconstrained


def to_physical(
    unconstrained: dict[str, float], bounds: dict[str, tuple[float, float]]
) -> dict[str, float]:
    """Inverse of `to_unconstrained`.

    The sigmoid keeps the result inside the bounds mathematically, but it saturates in
    float64 for |u| above roughly 40, landing on the bound exactly -- which the Tesseract
    rejects, since the bounds are exclusive. Pin the result one ULP inside so a long fit
    that drives a parameter hard against a bound degrades instead of erroring.
    """
    physical = {}
    for name, value in unconstrained.items():
        lower, upper = bounds[name]
        mapped = lower + (upper - lower) / (1.0 + np.exp(-value))
        physical[name] = float(np.clip(mapped, np.nextafter(lower, upper), np.nextafter(upper, lower)))
    return physical


def dphysical_dunconstrained(
    unconstrained: dict[str, float], bounds: dict[str, tuple[float, float]]
) -> dict[str, float]:
    """The chain-rule factor `span * sigmoid'(u)` relating the two spaces."""
    factors = {}
    for name, value in unconstrained.items():
        lower, upper = bounds[name]
        sigmoid = 1.0 / (1.0 + np.exp(-value))
        factors[name] = (upper - lower) * sigmoid * (1.0 - sigmoid)
    return factors


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean(np.square(pred - true)))


def evaluate(
    unconstrained: dict[str, float],
    true_electron_spectrum: np.ndarray,
    tsadaract: Tesseract,
    bounds: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, float, dict[str, float]]:
    """One step's worth of work: the spectrum, the loss, and dloss/dunconstrained."""
    physical = to_physical(unconstrained, bounds)

    electron_spectrum = tsadaract.apply(physical)["electron_spectrum"]
    jacobian = tsadaract.jacobian(physical, PARAMS, JAC_OUTPUTS)["electron_spectrum"]

    # Differentiate the MSE through the model, then through the reparameterization.
    error = electron_spectrum - true_electron_spectrum
    chain = dphysical_dunconstrained(unconstrained, bounds)
    grad = {name: 2 * np.mean(jacobian[name] * error) * chain[name] for name in PARAMS}

    return electron_spectrum, mse(electron_spectrum, true_electron_spectrum), grad


def display(parameters: dict[str, float]) -> dict[str, float]:
    """Round parameters for display."""
    return {name: round(float(value), 3) for name, value in parameters.items()}


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
        bounds = fetch_bounds(tesseract_url)
        check_sampling_ranges(bounds)

        col1, col2 = st.columns(2)

        # Sample the true parameters, and generate the synthetic spectrum to fit to.
        rng = np.random.default_rng()
        true_parameters = sample_parameters(rng)
        true_electron_spectrum = tsadaract.apply(true_parameters)["electron_spectrum"]

        with col1:
            st.write("True parameters:")
            st.json(display(true_parameters))

        # Create an independent initial guess.
        fit_parameters = sample_parameters(rng)
        unconstrained = to_unconstrained(fit_parameters, bounds)
        electron_spectrum = tsadaract.apply(fit_parameters)["electron_spectrum"]

        with col2:
            st.write("Estimated parameters:")
            fit_param_holder = st.empty()
        fig_holder = st.empty()

        fit_param_holder.json(display(fit_parameters))

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=true_electron_spectrum, mode="lines+markers", name="True Electron Spectrum"))
        fig.add_trace(go.Scatter(y=electron_spectrum, mode="lines+markers", name="Fit Electron Spectrum"))
        fig.update_layout(title="Electron Spectrum", xaxis_title="Wavelength", yaxis_title="Amplitude")
        fig_holder.plotly_chart(fig)

        learning_rate = st.number_input("Learning Rate", value=0.01, step=0.001, key="learning_rate")
        opt = optax.adam(learning_rate)
        opt_state = opt.init(unconstrained)

        if st.button("Fit"):

            for i in (pbar := tqdm.tqdm(range(1000))):

                electron_spectrum, loss, grad_loss = evaluate(unconstrained, true_electron_spectrum, tsadaract, bounds)

                updates, opt_state = opt.update(grad_loss, opt_state)
                unconstrained = optax.apply_updates(unconstrained, updates)
                pbar.set_description(f"Loss: {loss:.4f}")

                fig.data[1].y = electron_spectrum
                fig.data[1].name = f"Step {i+1}"
                fig.update_layout(
                    title=f"Electron Spectrum, Loss = {loss:.2e}", xaxis_title="Wavelength", yaxis_title="Amplitude"
                )
                fig_holder.plotly_chart(fig)

                fit_param_holder.json(display(to_physical(unconstrained, bounds)))
