import streamlit as st
import os


def get_config():
    st.subheader(
        "Provide the configuration options for the experiment as well as analysis using the following dropdowns and input fields."
    )
    st.write("More details can be found in the documentation at < >")

    config, files = create_default_config()
    # inputs = get_inputs()
    return config, files


def optimizer_section():
    optimizer = {
        "method": st.selectbox("Optimization method", ["l-bfgs-b", "adam", "sgd"], index=0),
        "hessian": st.checkbox("Use hessian", value=False),
        "y_norm": st.checkbox("Normalize y", value=True),
        "x_norm": st.checkbox("Normalize x", value=False),
        "grad_method": st.selectbox("Gradient method", ["AD", "FD", "Other"], index=0),
        "batch_size": st.number_input("Batch size", value=2, min_value=1, step=1),
        "num_epochs": st.number_input("Number of epochs", value=500, min_value=1, step=1),
        "learning_rate": st.number_input("Learning rate", value=1.0e-4, format="%.5e"),
        "parameter_norm": st.checkbox("Parameter normalization", value=True),
        "refine_factor": st.number_input("Refine factor", value=0, min_value=0, step=1),
        "num_mins": st.number_input("Number of minima", value=8, min_value=1, step=1),
        "sequential": st.checkbox("Sequential optimization", value=False),
        "moment_loss": st.checkbox("EDF moment loss penalties", value=False),
        "loss_method": st.selectbox("Loss method", ["l2", "l1", "poisson", "log-cosh"], index=0),
    }

    return optimizer


def get_species1():

    # Define variables using Streamlit widgets
    species1_type = "electron"  # st.selectbox("species1 type", ["electron"])
    species1_type_active = False  # st.checkbox("species1 type active", value=False)

    c1, c2 = st.columns(2)

    with c1:
        _c1, _c2 = st.columns(2)
        with _c1:
            st.write("Electron Temperature (keV)")
        with _c2:
            Te_active = st.checkbox("Fit Te?", value=True)
        Te_val = st.number_input("Initial value of Te", value=0.5)
        Te_lb = st.number_input("Lower bound for Te", value=0.01)
        Te_ub = st.number_input("Upper bound for Te", value=1.5)

    with c2:
        _c1, _c2 = st.columns(2)
        with _c1:
            st.write("Electron Density ($10^{20} cm{^-3}$)")
        with _c2:
            ne_active = st.checkbox("Fit ne?", value=True)
        ne_val = st.number_input("Initial value of ne", value=0.2)

        ne_lb = st.number_input("Lower bound for ne", value=0.03)
        ne_ub = st.number_input("Upper bound for ne", value=1.0)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        _c1, _c2 = st.columns(2)
        with _c1:
            st.write("Species Super Gaussian index")
        with _c2:
            m_active = st.checkbox("Fit m?", value=True)

        # st.write("Species Super Gaussian index")
        m_val = st.number_input("Initial m", value=2.0)

        m_lb = st.number_input("Lower bound for m", value=2.0)
        m_ub = st.number_input("Upper bound for m", value=5.0)
        m_matte = st.checkbox("Just use Matte formula?", value=False)
        m_intens = st.number_input("m intens", value=2.0)

    with c2:
        _c1, _c2 = st.columns(2)
        with _c1:

            st.write("Distribution Function")
        with _c2:
            fe_active = st.checkbox("Fit f?", value=False)
        fe_val = st.text_area("fe val", value="[]")
        fe_length = st.number_input("fe length", value=1024)
        fe_type = st.selectbox("fe type", ["DLM"])
        fe_lb = st.number_input("fe lb", value=-100.0)
        fe_ub = st.number_input("fe ub", value=-0.5)
        fe_decrease_strict = st.checkbox("fe decrease strict", value=False)
        fe_symmetric = st.checkbox("fe symmetric", value=False)
        fe_dim = st.number_input("fe dim", value=1)
        fe_v_res = st.number_input("fe v_res", value=0.05)
        fe_temp_asym = st.number_input("fe temp_asym", value=1.0)
        fe_m_theta = st.number_input("fe m_theta", value=0.0)
        fe_m_asym = st.number_input("fe m_asym", value=1.0)

    # Use the variables to populate the dictionary
    species1 = {
        "type": {
            "electron": species1_type,
            "active": species1_type_active,
        },
        "Te": {
            "val": Te_val,
            "active": Te_active,
            "lb": Te_lb,
            "ub": Te_ub,
        },
        "ne": {
            "val": ne_val,
            "active": ne_active,
            "lb": ne_lb,
            "ub": ne_ub,
        },
        "m": {
            "val": m_val,
            "active": m_active,
            "lb": m_lb,
            "ub": m_ub,
            "matte": m_matte,
            "intens": m_intens,
        },
        "fe": {
            "val": fe_val,
            "active": fe_active,
            "length": fe_length,
            "type": {"DLM": fe_type},
            "lb": fe_lb,
            "ub": fe_ub,
            "fe_decrease_strict": fe_decrease_strict,
            "symmetric": fe_symmetric,
            "dim": fe_dim,
            "v_res": fe_v_res,
            "temp_asym": fe_temp_asym,
            "m_theta": fe_m_theta,
            "m_asym": fe_m_asym,
        },
    }

    return species1


def get_species2():

    # extract the inline dictionary creation that follows to individual streamlit calls formatted using the column structure from the previous function
    st.divider()
    num_species = st.number_input("Number of ion species", value=1, min_value=1, max_value=10)
    Ti_same = st.checkbox("Same Ti across all ion species?", value=False)
    ion_dict = {}
    for i in range(num_species):
        st.subheader(f"Ion Species {i+1}")
        c1, c2 = st.columns(2)
        with c1:
            _c1, _c2 = st.columns(2)
            with _c1:
                st.write("Ion Temperature (keV)")
            with _c2:
                Ti_active = st.checkbox("Fit Ti?", value=False, key=f"Ti_active_{i}")
            Ti_val = st.number_input("Initial value of Ti", value=0.2, key=f"Ti_val_{i}")
            Ti_lb = st.number_input("Lower bound for Ti", value=0.01, key=f"Ti_lb_{i}")
            Ti_ub = st.number_input("Upper bound for Ti", value=1.5, key=f"Ti_ub_{i}")

        with c2:
            _c1, _c2 = st.columns(2)
            with _c1:
                st.write("Ion Charge")
            with _c2:
                Z_active = st.checkbox("Fit Z?", value=False, key=f"Z_active_{i}")
            Z_val = st.number_input("Initial value of Z", value=10.0, key=f"Z_val_{i}")
            Z_lb = st.number_input("Lower bound for Z", value=1.0, key=f"Z_lb_{i}")
            Z_ub = st.number_input("Upper bound for Z", value=18.0, key=f"Z_ub_{i}")

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            # _c1, _c2 = st.columns(2)
            # with _c1:
            st.write("Ion Mass")
            # with _c2:
            A_active = False
            A_val = st.number_input("Value of A", value=40.0, key=f"A_val_{i}")
            A_lb = 1.0
            A_ub = 100.0

        with c2:
            _c1, _c2 = st.columns(2)
            with _c1:
                st.write("Ion Fraction")
            with _c2:
                fract_active = st.checkbox("Fit fract?", value=False, key=f"fract_active_{i}")
            fract_val = st.number_input("Initial value of fract", value=1.0, key=f"fract_val_{i}")
            fract_lb = st.number_input("Lower bound for fract", value=0.0, key=f"fract_lb_{i}")
            fract_ub = st.number_input("Upper bound for fract", value=1.0, key=f"fract_ub_{i}")

        st.divider()
        st.divider()

        ion_dict[f"species{i+2}"] = {
            "type": {
                "ion": "ion",
                "active": False,  # st.checkbox("species2 type active", value=False),
            },
            "Ti": {
                "val": Ti_val,
                "active": Ti_active,
                "lb": Ti_lb,
                "ub": Ti_ub,
                "same": Ti_same,
            },
            "Z": {
                "val": Z_val,
                "active": Z_active,
                "lb": Z_lb,
                "ub": Z_ub,
            },
            "A": {
                "val": A_val,
                "active": A_active,
                "lb": A_lb,
                "ub": A_ub,
            },
            "fract": {
                "val": fract_val,
                "active": fract_active,
                "lb": fract_lb,
                "ub": fract_ub,
            },
        }

    return ion_dict


def get_general():
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        _c1, _c2 = st.columns(2)
        with _c1:
            st.write("amp1 - blue-shifted EPW")
        with _c2:
            amp1_active = st.checkbox("Fit amp1?", value=True)
        amp1_val = st.number_input("Initial value of amp1", value=1.0)
        amp1_lb = st.number_input("Lower bound for amp1", value=0.01)
        amp1_ub = st.number_input("Upper bound for amp1", value=3.75)

    with c2:
        _c1, _c2 = st.columns(2)
        with _c1:
            st.write("amp2 - red-shifted EPW")
        with _c2:
            amp2_active = st.checkbox("Fit amp2?", value=True)
        amp2_val = st.number_input("Initial value of amp2", value=1.0)
        amp2_lb = st.number_input("Lower bound for amp2", value=0.01)
        amp2_ub = st.number_input("Upper bound for amp2", value=3.75)

    with c3:
        _c1, _c2 = st.columns(2)
        with _c1:
            st.write("amp3 - IAW")
        with _c2:
            amp3_active = st.checkbox("Fit amp3?", value=False)
        amp3_val = st.number_input("Initial value of amp3", value=1.0)
        amp3_lb = st.number_input("Lower bound for amp3", value=0.01)
        amp3_ub = st.number_input("Upper bound for amp3", value=3.75)

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        _c1, _c2 = st.columns(2)
        with _c1:
            st.write("Lambda (nm)")
        with _c2:
            lam_active = st.checkbox("Fit Lambda?", value=True)
        lam_val = st.number_input("Initial value of Lambda", value=526.5)
        lam_lb = st.number_input("Lower bound for Lambda", value=523.0)
        lam_ub = st.number_input("Upper bound for Lambda", value=528.0)

    with c2:
        _c1, _c2 = st.columns(2)
        with _c1:
            st.write("Drift Velocity? (10^6 cm/s)")
        with _c2:
            ud_active = st.checkbox("Fit ud?", value=False)
        ud_val = st.number_input("Initial value of ud", value=0.0)
        ud_angle = st.number_input("ud angle", value=0.0)
        ud_lb = st.number_input("Lower bound for ud", value=-10.0)
        ud_ub = st.number_input("Upper bound for ud", value=10.0)

    with c3:
        _c1, _c2 = st.columns(2)
        with _c1:
            st.write("Flow velocity? (10^6 cm/s)")
        with _c2:
            Va_active = st.checkbox("Fit Va?", value=False)
        Va_val = st.number_input("Initial value of Va", value=0.0)
        Va_angle = st.number_input("Va angle", value=0.0)
        Va_lb = st.number_input("Lower bound for Va", value=-20.5)
        Va_ub = st.number_input("Upper bound for Va", value=20.5)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        _c1, _c2 = st.columns(2)
        with _c1:
            st.write("Te Gradient (% / scattering volume)")
        with _c2:
            Te_gradient_active = st.checkbox("Fit Te Gradient?", value=False)
        Te_gradient_val = st.number_input("Initial value of Te Gradient", value=0.0)
        Te_gradient_lb = st.number_input("Lower bound for Te Gradient", value=0.0)
        Te_gradient_ub = st.number_input("Upper bound for Te Gradient", value=10.0)
        Te_gradient_num_grad_points = st.number_input("Number of Te gradient points", value=1)

    with c2:
        _c1, _c2 = st.columns(2)
        with _c1:
            st.write("ne Gradient (% / scattering volume)")
        with _c2:
            ne_gradient_active = st.checkbox("Fit ne Gradient?", value=False)
        ne_gradient_val = st.number_input("Initial value of ne Gradient", value=0.0)
        ne_gradient_lb = st.number_input("Lower bound for ne Gradient", value=0.0)
        ne_gradient_ub = st.number_input("Upper bound for ne Gradient", value=15.0)
        ne_gradient_num_grad_points = st.number_input("Number of ne gradient points", value=1)

    general = {
        "type": {
            "general": "general",
            "active": False,  # st.checkbox("general type active", value=False),
        },
        "amp1": {
            "val": amp1_val,
            "active": amp1_active,
            "lb": amp1_lb,
            "ub": amp1_ub,
        },
        "amp2": {
            "val": amp2_val,
            "active": amp2_active,
            "lb": amp2_lb,
            "ub": amp2_ub,
        },
        "amp3": {
            "val": amp3_val,
            "active": amp3_active,
            "lb": amp3_lb,
            "ub": amp3_ub,
        },
        "lam": {
            "val": lam_val,
            "active": lam_active,
            "lb": lam_lb,
            "ub": lam_ub,
        },
        "Te_gradient": {
            "val": Te_gradient_val,
            "active": Te_gradient_active,
            "lb": Te_gradient_lb,
            "ub": Te_gradient_ub,
            "num_grad_points": Te_gradient_num_grad_points,
        },
        "ne_gradient": {
            "val": ne_gradient_val,
            "active": ne_gradient_active,
            "lb": ne_gradient_lb,
            "ub": ne_gradient_ub,
            "num_grad_points": ne_gradient_num_grad_points,
        },
        "ud": {
            "val": ud_val,
            "angle": ud_angle,
            "active": ud_active,
            "lb": ud_lb,
            "ub": ud_ub,
        },
        "Va": {
            "val": Va_val,
            "angle": Va_angle,
            "active": Va_active,
            "lb": Va_lb,
            "ub": Va_ub,
        },
    }

    return general


# Function to create an interactive dictionary structure for the merged YAML
def create_default_config():

    username = st.text_input("Please input your username (must match the username you used to log in)", value="")

    if username.casefold() in os.environ["USERNAMES"].split(","):
        with st.expander(
            "Mlflow - This will help you find your run on the mlflow server and helps keeps things organized"
        ):
            c1, c2 = st.columns(2)
            with c1:
                exp = st.text_input("experiment name", value=f"{username}-experiments")

            with c2:
                run = st.text_input("run name", value=f"fit")

            mlflow = {"experiment": exp, "run": run}

        with st.expander("Electron parameters"):
            species1 = get_species1()

        with st.expander("Ion parameters"):
            ion_dict = get_species2()

        with st.expander("General/Misc parameters"):
            general = get_general()

        parameters = {
            "species1": species1,
            # "species2": species2,
            "general": general,
        }
        parameters = parameters | ion_dict

        with st.expander("Data"):
            # extract the inline dictionary creation from the previous code to individual streamlit calls formatted using the column structure from the previous function
            c1, c2 = st.columns(2)
            with c1:
                shotnum = st.number_input("shotnum", value=101675)
                # file upload
                files = {}
                epw_file = st.file_uploader("Upload the EPW file", type=["hdf"])
                if epw_file:
                    files["epw"] = epw_file
                iaw_file = st.file_uploader("Upload the IAW file", type=["hdf"])
                if iaw_file:
                    files["iaw"] = iaw_file

                shotDay = st.checkbox("shotDay", value=False)
                st.divider()
                st.write("Which lineouts?")
                lineout_type = st.selectbox("lineouts type", ["pixel"])
                lineout_start = st.number_input("lineouts start", value=400)
                lineout_end = st.number_input("lineouts end", value=600)
                lineout_skip = st.number_input("lineouts skip", value=5)
                st.divider()
                probe_beam = st.selectbox("Probe Beam", ["P9"])
            with c2:
                st.write("Background?")
                background_type = st.selectbox("background type", ["pixel"])
                background_slice = st.number_input("background slice", value=900)
                ele_t0 = st.number_input("ele_t0", value=1500.0)
                ion_t0_shift = st.number_input("ion_t0_shift", value=900.0)
                dpixel = st.number_input("dpixel", value=3)
                bgscaleE = st.number_input("bgscaleE", value=1.0)
                bgscaleI = st.number_input("bgscaleI", value=0.1)
                launch_data_visualizer = False  # st.checkbox("launch_data_visualizer", value=True)
                ele_lam_shift = st.number_input("ele_lam_shift", value=0.0)
                ion_loss_scale = st.number_input("ion_loss_scale", value=2.0)

            st.divider()
            st.write("Spectral Fitting Range")
            c1, c2 = st.columns(2)
            with c1:
                blue_min = st.number_input("Blue Min", min_value=400, max_value=510, value=430)
                blue_max = st.number_input("Blue Max", min_value=400, max_value=510, value=510)
                red_min = st.number_input("Red Min", min_value=500, max_value=660, value=545)
                red_max = st.number_input("Red Max", min_value=500, max_value=660, value=660)
                iaw_min = st.number_input("IAW Min", min_value=500.0, max_value=530.0, value=525.5)
                iaw_max = st.number_input("IAW Max", min_value=500.0, max_value=530.0, value=527.5)

            with c2:
                iaw_cf_min = st.number_input("IAW CF Min", min_value=526.0, max_value=527.0, value=526.49)
                iaw_cf_max = st.number_input("IAW CF Max", min_value=526.0, max_value=527.0, value=526.51)
                forward_epw_start = st.number_input("Forward EPW Start", min_value=300, max_value=800, value=400)
                forward_epw_end = st.number_input("Forward EPW End", min_value=300, max_value=800, value=700)
                forward_iaw_start = st.number_input("Forward IAW Start", min_value=500.0, max_value=530.0, value=525.75)
                forward_iaw_end = st.number_input("Forward IAW End", min_value=500.0, max_value=530.0, value=527.25)

            data = {
                "shotnum": shotnum,
                "shotDay": shotDay,
                "filenames": {},
                "lineouts": {
                    "type": lineout_type,
                    "start": lineout_start,
                    "end": lineout_end,
                    "skip": lineout_skip,
                },
                "background": {
                    "type": background_type,
                    "slice": background_slice,
                },
                "probe_beam": probe_beam,
                "ele_t0": ele_t0,
                "ion_t0_shift": ion_t0_shift,
                "dpixel": dpixel,
                "bgscaleE": bgscaleE,
                "bgscaleI": bgscaleI,
                "launch_data_visualizer": launch_data_visualizer,
                "ele_lam_shift": ele_lam_shift,
                "ion_loss_scale": ion_loss_scale,
                "fit_rng": {
                    "blue_min": blue_min,
                    "blue_max": blue_max,
                    "red_min": red_min,
                    "red_max": red_max,
                    "iaw_min": iaw_min,
                    "iaw_max": iaw_max,
                    "iaw_cf_min": iaw_cf_min,
                    "iaw_cf_max": iaw_cf_max,
                    "forward_epw_start": forward_epw_start,
                    "forward_epw_end": forward_epw_end,
                    "forward_iaw_start": forward_iaw_start,
                    "forward_iaw_end": forward_iaw_end,
                },
            }

        with st.expander("Other"):
            other = {
                "extraoptions": {
                    "load_ion_spec": st.checkbox("load_ion_spec", value=False),
                    "load_ele_spec": st.checkbox("load_ele_spec", value=True),
                    "fit_IAW": st.checkbox("fit_IAW", value=False),
                    "fit_EPWb": st.checkbox("fit_EPWb", value=True),
                    "fit_EPWr": st.checkbox("fit_EPWr", value=True),
                    "absolute_timing": st.checkbox("absolute_timing", value=False),
                    "spectype": st.selectbox("spectype", ["temporal"]),
                },
                "PhysParams": {
                    "widIRF": {
                        "spect_stddev_ion": st.number_input("spect_stddev_ion", value=0.015),
                        "spect_stddev_ele": st.number_input("spect_stddev_ele", value=0.1),
                        "spect_FWHM_ele": st.number_input("spect_FWHM_ele", value=0.9),
                        "ang_FWHM_ele": st.number_input("ang_FWHM_ele", value=1.0),
                    }
                },
                "refit": st.checkbox("refit", value=False),
                "refit_thresh": st.number_input("refit_thresh", value=5.0),
                "calc_sigmas": st.checkbox("calc_sigmas", value=False),
                "CCDsize": st.number_input("CCD Size (uses same value for both dimensions)", value=1024),
                "flatbg": st.checkbox("flatbg", value=False),
                "gain": st.number_input("gain", value=1.0),
                "points_per_pixel": st.number_input("points_per_pixel", value=1),
                "iawoff": st.number_input("iawoff", value=0.0),
            }
            # Input as a comma-separated string
            iawfilter_input = st.text_input("IAW Filter (comma-separated)", value="1, 4, 24, 528")

            # Convert the input string into a list of integers
            iawfilter = [int(i) for i in iawfilter_input.split(",")]
            other["iawfilter"] = iawfilter

            background_input = st.text_input("Background (comma-separated)", value="0, 0")
            background = [int(i) for i in background_input.split(",")]

            # Input for 'norm' as an integer value
            norm = st.number_input("Norm", min_value=0, value=0)

            # Creating the PhysParams dictionary
            other["PhysParams"] = {"background": background, "norm": norm}

            other["CCDsize"] = [other["CCDsize"], other["CCDsize"]]
            other["username"] = username

        with st.expander("Plotting"):
            # Input for n_sigmas (integer)
            n_sigmas = st.number_input("n_sigmas", min_value=0, value=3)

            # Input for rolling_std_width (integer)
            rolling_std_width = st.number_input("rolling_std_width", min_value=1, value=5)

            # Input for data_cbar_u (string)
            data_cbar_u = st.text_input("data_cbar_u", value="data")

            # Input for data_cbar_l (integer)
            data_cbar_l = st.number_input("data_cbar_l", min_value=0, value=0)

            # Input for ion_window_start (integer)
            ion_window_start = st.number_input("ion_window_start", min_value=500, max_value=530, value=525)

            # Input for ion_window_end (integer)
            ion_window_end = st.number_input("ion_window_end", min_value=500, max_value=530, value=528)

            # Input for ele_window_start (integer)
            ele_window_start = st.number_input("ele_window_start", min_value=400, max_value=700, value=425)

            # Input for ele_window_end (integer)
            ele_window_end = st.number_input("ele_window_end", min_value=400, max_value=700, value=660)

            # Creating the plotting dictionary
            plotting = {
                "n_sigmas": n_sigmas,
                "rolling_std_width": rolling_std_width,
                "data_cbar_u": data_cbar_u,
                "data_cbar_l": data_cbar_l,
                "ion_window_start": ion_window_start,
                "ion_window_end": ion_window_end,
                "ele_window_start": ele_window_start,
                "ele_window_end": ele_window_end,
            }

        with st.expander("Optimizer"):
            optimizer = optimizer_section()

        config = {
            "parameters": parameters,
            "data": data,
            "other": other,
            "mlflow": mlflow,
            "optimizer": optimizer,
            "plotting": plotting,
        }

        if epw_file is not None:
            if str(config["data"]["shotnum"]) in epw_file.name:
                config["data"]["filenames"]["epw"] = epw_file.name
            else:
                st.error("The EPW file name does not match the shot number")

        if iaw_file is not None:
            if str(config["data"]["shotnum"]) in iaw_file.name:
                config["data"]["filenames"]["iaw"] = iaw_file.name
            else:
                st.error("The IAW file name does not match the shot number")

    else:
        config = {}
        files = []
        st.error("The username you entered does not match the list of valid usernames")

    return config, files
