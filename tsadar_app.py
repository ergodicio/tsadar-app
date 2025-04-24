import streamlit as st
import yaml, os, mlflow, tempfile, boto3

from tsadar_gui import config, tesseract_ui

DEBUG = False


if __name__ == "__main__":
    # st.title("Thomson Scattering Analysis")
    st.sidebar.title("TSADARapp")

    # select box between forward and fit
    mode = st.sidebar.selectbox("Where to?", ["Home", "Fit", "Tesseract (Experimental)"])

    config_dir = os.path.join(os.getcwd(), "temp")

    os.makedirs(config_dir, exist_ok=True)

    if mode == "Home":
        st.header("Welcome to TSADARapp")
        st.write(
            "This web app is a Streamlit implementation of the [Thomson Scattering Analysis](https://www.github.com/ergodicio/inverse-thomson-scattering/) software [1]."
        )
        st.write(
            "The software is designed to analyze Thomson scattering data to extract electron and ion temperatures and densities."
        )

    elif mode == "Fit":
        st.header("Fit Thomson Scattering Data from OMEGA")
        st.write("Configure and submit a job to the TSADAR cluster for analysis.")
        cfg, files = config.get_config()

        c1, c2 = st.columns(2)

        if st.button("Fit"):
            # run and wait for the results
            mlflow.set_experiment(cfg["mlflow"]["experiment"])

            with tempfile.TemporaryDirectory() as tempdir:
                with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as mlflow_run:
                    # write config yaml to disk
                    with open(os.path.join(tempdir, "config.yaml"), "w") as f:
                        yaml.dump(cfg, f)

                    # write uploaded streamlit file to disk
                    for k, fl in files.items():
                        with open(os.path.join(tempdir, fl.name), "wb") as f:
                            f.write(fl.read())

                    mlflow.log_artifacts(tempdir)
                    run_id = mlflow_run.info.run_id

            # requests.post(
            #     "http://sciapi.continuum/queue-run",
            #     data={"jq_nm": "gpu", "jd_nm": "gpu", "job_name": "thomson", "run_type": "dm", "run_id": run_id},
            # )
            if DEBUG:
                from tsadar import run_for_app

                run_for_app(run_id)
            else:
                client = boto3.client("batch", region_name="us-east-1")

                job_template = {
                    "jobQueue": "gpu",
                    "jobDefinition": "tsadar-gpu",
                    "jobName": "tsadar",
                    "parameters": {"run_id": run_id},
                    "retryStrategy": {
                        "attempts": 10,
                        "evaluateOnExit": [{"action": "RETRY", "onStatusReason": "Host EC2*"}],
                    },
                }
                submissionResult = client.submit_job(**job_template)
                st.write(
                    f"The job is queued. The status and results can be found at https://continuum.ergodic.io/tsadar under the experiment -- {cfg['mlflow']['experiment']} -- and run id -- {run_id} --. Email support@ergodic.io if you have any trouble."
                )

    elif mode == "Tesseract (Experimental)":
        st.header("Tesseract (Experimental)")
        st.write(
            "This is an experimental feature that queries a Tesseract to analyze Thomson scattering data. "
            "For now, functionality is limited to fitting to randomly generated synthetic spectra rather than real data from OMEGA."
        )
        tesseract_ui.tesseract_ui()

    st.sidebar.title("About")
    ## Add attribution
    st.sidebar.markdown(
        "This web app is a Streamlit implementation of the [Thomson Scattering Analysis](https://www.github.com/ergodicio/inverse-thomson-scattering/) software [1]."
    )
    # add ref 1
    st.sidebar.markdown(
        "1. Milder, A. L., Joglekar, A. S., Rozmus, W. & Froula, D. H. Qualitative and quantitative enhancement of parameter estimation for model-based diagnostics using automatic differentiation with an application to inertial fusion. Mach. Learn.: Sci. Technol. 5, 015026 (2024)."
    )
