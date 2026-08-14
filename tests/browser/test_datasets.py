"""Dataset slicing endpoints (issue #30).

The tests that matter most here are the ones asserting angular runs are refused
distinguishably, and that a 1D dataset is never substituted for an angular one --
the two files have identical variables and dimensionality, so nothing but an
explicit check separates them.
"""

import numpy as np
import pytest
import xarray as xr

from tsadar_browser.datasets import DatasetUnavailable
from tsadar_browser.schemas import DatasetKind, UnavailableReason

from .conftest import install_artifacts, make_run
from .fixtures import ANGULAR_AXIS, SPATIAL_AXIS, TEMPORAL_AXIS, write_spectrum


class TestAvailability:
    def test_one_d_run_is_supported(self, dataset_client, one_d_run):
        body = dataset_client.get(f"/api/runs/{one_d_run}/datasets").json()
        assert body["kind"] == "one_d"
        assert body["supported"] is True
        assert body["reason"] is None
        assert {s["which"] for s in body["spectra"]} == {"ele", "ion"}
        assert body["profiles_available"] is True
        assert body["sigmas_available"] is True

    def test_spectrum_info_reports_shape_and_axis(self, dataset_client, one_d_run):
        body = dataset_client.get(f"/api/runs/{one_d_run}/datasets").json()
        ele = next(s for s in body["spectra"] if s["which"] == "ele")
        assert ele["x_label"] == TEMPORAL_AXIS
        assert ele["y_label"] == "Wavelength"
        assert (ele["lineout_count"], ele["wavelength_count"]) == (6, 32)
        assert ele["fields"] == ["data", "fit", "residual"]

    def test_irf_is_reported_unavailable_not_omitted(self, dataset_client, one_d_run):
        """IRF genuinely isn't in the netCDFs; say so rather than staying silent."""
        body = dataset_client.get(f"/api/runs/{one_d_run}/datasets").json()
        assert "irf" in body["unavailable_fields"]
        assert "not written to the netCDF" in body["unavailable_fields"]["irf"]

    def test_angular_run_is_unsupported_with_a_reason(self, dataset_client, angular_run):
        """Must answer 200 with a reason, so #32 can pick the gallery layout."""
        response = dataset_client.get(f"/api/runs/{angular_run}/datasets")
        assert response.status_code == 200
        body = response.json()
        assert body["kind"] == "angular"
        assert body["supported"] is False
        assert body["reason"] == "angular_not_supported"
        assert "angularly-resolved" in body["message"]
        assert body["spectra"] == []

    def test_pre_contract_run_is_unsupported_but_not_an_error(self, dataset_client, bare_run):
        body = dataset_client.get(f"/api/runs/{bare_run}/datasets").json()
        assert body["kind"] == "unknown"
        assert body["supported"] is False
        assert body["reason"] == "dataset_missing"

    def test_run_with_only_ele_reports_only_ele(self, fake_client, dataset_client, tmp_path):
        install_artifacts(
            fake_client,
            "run-abc",
            {"binary/ele_fit_and_data.nc": write_spectrum(tmp_path / "e", "ele_fit_and_data.nc")},
        )
        body = dataset_client.get("/api/runs/run-abc/datasets").json()
        assert [s["which"] for s in body["spectra"]] == ["ele"]
        assert body["profiles_available"] is False


class TestClassification:
    def test_angular_dataset_is_never_used_as_a_fallback(self, dataset_service, angular_run):
        """The trap: fit_and_data.nc has the same vars and rank as ele_fit_and_data.nc.

        A "try the other file if ele/ion are absent" fallback would serve
        angle-vs-wavelength data to a UI labelling its x-axis as time.
        """
        with pytest.raises(DatasetUnavailable) as caught:
            dataset_service.spectrogram(angular_run, which="ele", field="data", max_px=None)
        assert caught.value.reason is UnavailableReason.angular_not_supported
        assert caught.value.status_code == 409

    def test_classification_uses_artifacts_not_the_spectype_param(
        self, fake_client, dataset_service, tmp_path
    ):
        """spectype is logged before the fit runs, so it can lie; artifacts can't."""
        fake_client.runs["run-liar"] = make_run(
            run_id="run-liar", params={"other.extraoptions.spectype": "angular_full"}
        )
        install_artifacts(
            fake_client,
            "run-liar",
            {"binary/ele_fit_and_data.nc": write_spectrum(tmp_path / "l", "ele_fit_and_data.nc")},
        )
        kind, present = dataset_service.classify("run-liar")
        assert kind is DatasetKind.one_d, "artifact shape should win over a stale param"
        assert present == ["ele"]

    def test_angular_axis_inside_a_1d_named_file_is_still_refused(
        self, fake_client, dataset_service, tmp_path
    ):
        """Defense in depth: the axis name is checked even past the shape check."""
        install_artifacts(
            fake_client,
            "run-abc",
            {
                "binary/ele_fit_and_data.nc": write_spectrum(
                    tmp_path / "sneaky", "ele_fit_and_data.nc", x_label=ANGULAR_AXIS
                )
            },
        )
        with pytest.raises(DatasetUnavailable) as caught:
            dataset_service.spectrogram("run-abc", which="ele", field="data", max_px=None)
        assert caught.value.reason is UnavailableReason.angular_not_supported

    def test_spatial_axis_is_supported(self, fake_client, dataset_service, tmp_path):
        """Imaging runs are in scope; only angular is out."""
        install_artifacts(
            fake_client,
            "run-abc",
            {
                "binary/ele_fit_and_data.nc": write_spectrum(
                    tmp_path / "sp", "ele_fit_and_data.nc", x_label=SPATIAL_AXIS
                )
            },
        )
        result = dataset_service.spectrogram("run-abc", which="ele", field="data", max_px=None)
        assert result.x_label == SPATIAL_AXIS


class TestSpectrogram:
    def test_returns_plotly_ready_orientation(self, dataset_client, one_d_run):
        """values must be [y][x] so it drops straight into a heatmap z."""
        body = dataset_client.get(f"/api/runs/{one_d_run}/spectrogram?which=ele&field=data").json()
        assert len(body["y"]) == 32
        assert len(body["x"]) == 6
        assert len(body["values"]) == len(body["y"])
        assert len(body["values"][0]) == len(body["x"])

    def test_full_resolution_when_under_budget(self, dataset_client, one_d_run):
        body = dataset_client.get(f"/api/runs/{one_d_run}/spectrogram?max_px=100000").json()
        assert body["downsample_method"] == "none"
        assert body["downsample_factors"] == {"x": 1, "wavelength": 1}
        assert body["returned_shape"] == body["full_shape"] == [6, 32]

    def test_downsamples_wavelength_and_spares_lineouts(self, dataset_client, one_d_run):
        """Each lineout is a separate fit and the scrubber axis, so it is preserved."""
        body = dataset_client.get(f"/api/runs/{one_d_run}/spectrogram?max_px=48").json()
        assert body["returned_shape"][0] == 6, "lineout axis should be spared"
        assert body["downsample_factors"]["x"] == 1
        assert body["downsample_factors"]["wavelength"] > 1
        assert body["returned_shape"][0] * body["returned_shape"][1] <= 48
        assert body["downsample_method"] == "mean"

    def test_coarsens_lineouts_only_when_wavelength_is_not_enough(self, dataset_service, one_d_run):
        result = dataset_service.spectrogram(one_d_run, which="ele", field="data", max_px=3)
        assert result.downsample_factors["x"] > 1
        assert result.returned_shape[0] * result.returned_shape[1] <= 6

    @pytest.mark.parametrize(
        ("n_lineouts", "n_wavelength", "budget"),
        [(60, 1024, 20_000), (60, 1024, 2_000), (300, 1024, 5_000)],
    )
    def test_realistic_detector_shapes_keep_every_lineout(
        self, dataset_service, n_lineouts, n_wavelength, budget
    ):
        """Wavelength is exhausted before any lineout is given up.

        Integer division can leave the first factor estimate just over budget
        (60x1024 to 2000 lands on 60x34 = 2040); tightening wavelength one more
        step gets under it without dropping fits.
        """
        array = xr.DataArray(
            np.zeros((n_lineouts, n_wavelength)),
            coords=(
                (TEMPORAL_AXIS, np.arange(n_lineouts, dtype=float)),
                ("Wavelength", np.arange(n_wavelength, dtype=float)),
            ),
        )
        reduced, (x_factor, _) = dataset_service._downsample(
            array, TEMPORAL_AXIS, "Wavelength", budget
        )
        assert x_factor == 1, "no lineout should be lost at a realistic budget"
        assert reduced.sizes[TEMPORAL_AXIS] == n_lineouts
        assert reduced.size <= budget

    def test_residual_is_derived_from_data_minus_fit(self, dataset_service, one_d_run):
        """Residual is not stored, so it must be computed as data - fit.

        Compared with a tolerance, not exactly: the residual is differenced at
        full precision server-side and only then trimmed to the wire's
        significant digits, so it is not the difference of the two rounded
        arrays this test also fetches.
        """
        data = dataset_service.spectrogram(one_d_run, which="ele", field="data", max_px=None)
        fit = dataset_service.spectrogram(one_d_run, which="ele", field="fit", max_px=None)
        residual = dataset_service.spectrogram(one_d_run, which="ele", field="residual", max_px=None)

        expected = np.array(data.values, dtype=float) - np.array(fit.values, dtype=float)
        # Measured deviation is ~9e-6 absolute / 7e-5 relative, the double-rounding
        # floor at magnitude ~1. A genuinely wrong residual would be off by ~0.1.
        assert np.allclose(np.array(residual.values, dtype=float), expected, rtol=1e-3, atol=1e-5)

    def test_ion_spectrum_is_servable(self, dataset_client, one_d_run):
        assert dataset_client.get(f"/api/runs/{one_d_run}/spectrogram?which=ion").status_code == 200

    def test_unknown_field_is_a_400_with_a_reason(self, dataset_client, one_d_run):
        response = dataset_client.get(f"/api/runs/{one_d_run}/spectrogram?field=irf")
        assert response.status_code == 400
        assert response.json()["detail"]["reason"] == "field_unavailable"

    def test_unknown_spectrum_is_a_400_with_a_reason(self, dataset_client, one_d_run):
        response = dataset_client.get(f"/api/runs/{one_d_run}/spectrogram?which=nope")
        assert response.status_code == 400
        assert response.json()["detail"]["reason"] == "dataset_missing"

    def test_angular_run_is_a_409_with_a_reason(self, dataset_client, angular_run):
        response = dataset_client.get(f"/api/runs/{angular_run}/spectrogram")
        assert response.status_code == 409
        assert response.json()["detail"]["reason"] == "angular_not_supported"

    def test_missing_dataset_is_a_404_with_a_reason(self, dataset_client, bare_run):
        response = dataset_client.get(f"/api/runs/{bare_run}/spectrogram")
        assert response.status_code == 404
        assert response.json()["detail"]["reason"] == "dataset_missing"

    def test_absent_requested_spectrum_is_a_404(self, fake_client, dataset_client, tmp_path):
        install_artifacts(
            fake_client,
            "run-abc",
            {"binary/ele_fit_and_data.nc": write_spectrum(tmp_path / "e", "ele_fit_and_data.nc")},
        )
        response = dataset_client.get("/api/runs/run-abc/spectrogram?which=ion")
        assert response.status_code == 404


class TestNonFiniteValues:
    def test_nan_and_inf_serialize_as_null(self, fake_client, dataset_client, tmp_path):
        """JSON has no NaN; a bare NaN makes browser JSON.parse throw."""
        install_artifacts(
            fake_client,
            "run-abc",
            {
                "binary/ele_fit_and_data.nc": write_spectrum(
                    tmp_path / "nan", "ele_fit_and_data.nc", with_nan=True
                )
            },
        )
        response = dataset_client.get("/api/runs/run-abc/spectrogram?field=data&max_px=100000")
        assert response.status_code == 200
        assert "NaN" not in response.text and "Infinity" not in response.text
        assert response.json()["values"][0][0] is None

    def test_inf_in_the_fit_also_becomes_null(self, fake_client, dataset_client, tmp_path):
        install_artifacts(
            fake_client,
            "run-abc",
            {
                "binary/ele_fit_and_data.nc": write_spectrum(
                    tmp_path / "inf", "ele_fit_and_data.nc", with_nan=True
                )
            },
        )
        body = dataset_client.get("/api/runs/run-abc/lineout?index=1&which=ele").json()
        assert body["fit"][2] is None


class TestLineout:
    def test_returns_measured_and_fitted_spectrum(self, dataset_client, one_d_run):
        body = dataset_client.get(f"/api/runs/{one_d_run}/lineout?which=ele&index=2").json()
        assert body["index"] == 2
        assert body["lineout_count"] == 6
        assert body["x_label"] == TEMPORAL_AXIS
        assert len(body["wavelength"]) == 32
        assert len(body["data"]) == len(body["fit"]) == len(body["residual"]) == 32

    def test_x_value_locates_the_lineout_on_its_axis(self, dataset_client, one_d_run):
        body = dataset_client.get(f"/api/runs/{one_d_run}/lineout?index=0").json()
        assert body["x_value"] == pytest.approx(-100.0)

    def test_residual_matches_data_minus_fit(self, dataset_client, one_d_run):
        # Tolerance rather than equality: see the spectrogram equivalent -- the
        # residual is differenced before the wire rounding, not after it.
        body = dataset_client.get(f"/api/runs/{one_d_run}/lineout?index=3").json()
        expected = np.array(body["data"], dtype=float) - np.array(body["fit"], dtype=float)
        assert np.allclose(np.array(body["residual"], dtype=float), expected, rtol=1e-3, atol=1e-5)

    def test_negative_index_counts_from_the_end(self, dataset_client, one_d_run):
        body = dataset_client.get(f"/api/runs/{one_d_run}/lineout?index=-1").json()
        assert body["index"] == 5

    def test_components_are_empty_and_explained(self, dataset_client, one_d_run):
        """IRF/noise aren't in the datasets, so report why instead of an empty dict."""
        body = dataset_client.get(f"/api/runs/{one_d_run}/lineout?index=0").json()
        assert body["components"] == {}
        assert "not written to the netCDF" in body["components_unavailable"]

    def test_out_of_range_index_is_a_400_with_a_reason(self, dataset_client, one_d_run):
        response = dataset_client.get(f"/api/runs/{one_d_run}/lineout?index=99")
        assert response.status_code == 400
        assert response.json()["detail"]["reason"] == "index_out_of_range"

    def test_angular_run_is_refused(self, dataset_client, angular_run):
        response = dataset_client.get(f"/api/runs/{angular_run}/lineout")
        assert response.status_code == 409
        assert response.json()["detail"]["reason"] == "angular_not_supported"


class TestProfiles:
    def test_returns_one_series_per_fitted_parameter(self, dataset_client, one_d_run):
        body = dataset_client.get(f"/api/runs/{one_d_run}/profiles").json()
        assert body["x_label"] == TEMPORAL_AXIS
        assert len(body["x"]) == 6
        assert {s["name"] for s in body["series"]} == {"Te_electron", "ne_electron", "amp1_general"}

    def test_index_column_from_to_csv_is_not_mistaken_for_a_parameter(self, dataset_client, one_d_run):
        """to_csv writes the DataFrame index as a leading unnamed column."""
        names = {s["name"] for s in dataset_client.get(f"/api/runs/{one_d_run}/profiles").json()["series"]}
        assert not any(name.startswith("Unnamed") for name in names)

    def test_lineout_pixels_are_surfaced_separately(self, dataset_client, one_d_run):
        body = dataset_client.get(f"/api/runs/{one_d_run}/profiles").json()
        assert body["lineout_pixels"] == [800, 801, 802, 803, 804, 805]
        assert "lineout pixel" not in {s["name"] for s in body["series"]}

    def test_sigmas_are_attached_where_available(self, dataset_client, one_d_run):
        """sigmas.nc lives at the artifact root, not under binary/."""
        body = dataset_client.get(f"/api/runs/{one_d_run}/profiles").json()
        assert body["sigmas_available"] is True
        by_name = {s["name"]: s for s in body["series"]}
        assert by_name["Te_electron"]["sigma"] == [pytest.approx(0.05)] * 6
        assert by_name["amp1_general"]["sigma"] is None, "no sigma logged for this parameter"

    def test_absent_sigmas_are_not_an_error(self, fake_client, dataset_client, tmp_path):
        """calc_sigmas is off by default, so most runs have none."""
        from .fixtures import learned_parameters_csv

        install_artifacts(
            fake_client,
            "run-abc",
            {
                "binary/ele_fit_and_data.nc": write_spectrum(tmp_path / "e", "ele_fit_and_data.nc"),
                "csv/learned_parameters.csv": learned_parameters_csv(),
            },
        )
        body = dataset_client.get("/api/runs/run-abc/profiles").json()
        assert body["sigmas_available"] is False
        assert all(series["sigma"] is None for series in body["series"])

    def test_missing_csv_is_a_404_with_a_reason(self, fake_client, dataset_client, tmp_path):
        install_artifacts(
            fake_client,
            "run-abc",
            {"binary/ele_fit_and_data.nc": write_spectrum(tmp_path / "e", "ele_fit_and_data.nc")},
        )
        response = dataset_client.get("/api/runs/run-abc/profiles")
        assert response.status_code == 404
        assert response.json()["detail"]["reason"] == "dataset_missing"

    def test_angular_run_is_refused_before_reading_its_axis_less_csv(self, dataset_client, angular_run):
        """Angular skips the axis insert, so profiles have no x axis at all."""
        response = dataset_client.get(f"/api/runs/{angular_run}/profiles")
        assert response.status_code == 409
        assert response.json()["detail"]["reason"] == "angular_not_supported"


class TestOpenApi:
    def test_dataset_endpoints_are_in_the_schema(self, dataset_client):
        paths = dataset_client.get("/api/openapi.json").json()["paths"]
        assert {
            "/api/runs/{run_id}/datasets",
            "/api/runs/{run_id}/spectrogram",
            "/api/runs/{run_id}/lineout",
            "/api/runs/{run_id}/profiles",
        } <= set(paths)

    def test_error_bodies_document_the_reason_code(self, dataset_client):
        schema = dataset_client.get("/api/openapi.json").json()
        responses = schema["paths"]["/api/runs/{run_id}/spectrogram"]["get"]["responses"]
        assert {"400", "404", "409"} <= set(responses)
        assert "DatasetUnavailableResponse" in str(responses)


class TestWireFormat:
    def test_default_budget_does_not_ship_full_resolution(self, fake_client, dataset_client, tmp_path):
        """#30: 'Never ship full resolution by default.'

        A realistic spectrogram is 60 x 1024 = 61440 points, so the default
        budget has to sit below that rather than merely capping pathological
        cases.
        """
        install_artifacts(
            fake_client,
            "run-abc",
            {
                "binary/ele_fit_and_data.nc": write_spectrum(
                    tmp_path / "big", "ele_fit_and_data.nc", n_lineouts=60, n_wavelength=1024
                )
            },
        )
        body = dataset_client.get("/api/runs/run-abc/spectrogram").json()
        assert body["full_shape"] == [60, 1024]
        assert body["downsample_method"] == "mean"
        assert body["returned_shape"][0] == 60, "still without sacrificing lineouts"
        assert body["returned_shape"][0] * body["returned_shape"][1] < 60 * 1024

    def test_values_are_trimmed_to_significant_digits(self, dataset_client, one_d_run):
        """Full float repr roughly doubles the payload for precision no plot shows."""
        body = dataset_client.get(f"/api/runs/{one_d_run}/spectrogram?max_px=100000").json()
        flat = [value for row in body["values"] for value in row if value is not None]
        assert flat, "expected some values"
        assert all(len(repr(value).split(".")[-1].rstrip("0")) <= 8 for value in flat)

    def test_rounding_preserves_zero_and_sign(self):
        from tsadar_browser.datasets import _round_significant

        result = _round_significant(np.array([0.0, -1234.56789, 1e-30]), 6)
        assert result[0] == 0.0
        assert result[1] == pytest.approx(-1234.57)
        assert result[2] == pytest.approx(1e-30)


class TestReviewRegressions:
    """Cases from the review on #40."""

    def test_declared_error_schema_matches_the_wire_body(self, dataset_client, angular_run):
        """The generated client reads the schema, so a flat declaration lies.

        FastAPI nests whatever an HTTPException carries under `detail`, so a
        consumer trusting a flat schema would read err.reason and get undefined
        on exactly the angular-vs-missing distinction these endpoints exist for.
        """
        response = dataset_client.get(f"/api/runs/{angular_run}/spectrogram")
        assert response.status_code == 409
        body = response.json()

        schema = dataset_client.get("/api/openapi.json").json()
        ref = schema["paths"]["/api/runs/{run_id}/spectrogram"]["get"]["responses"]["409"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        declared = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]

        # Declared shape must be the envelope, not its contents.
        assert list(declared["properties"]) == ["detail"]
        inner_ref = declared["properties"]["detail"]["$ref"].rsplit("/", 1)[-1]
        inner = schema["components"]["schemas"][inner_ref]
        assert set(inner["properties"]) == {"reason", "detail"}

        # And the wire body must actually match it.
        assert set(body) == {"detail"}
        assert set(body["detail"]) == {"reason", "detail"}
        assert body["detail"]["reason"] == "angular_not_supported"

    def test_serving_paths_list_only_the_binary_directory(self, fake_client, dataset_client, tmp_path):
        """Classification must not walk the whole artifact tree per request.

        A real run has binary/, csv/, plots/, lineouts/, best/ and worst/, and the
        lineout scrubber steps interactively -- paying a round trip per directory
        on every step before reading any data.
        """
        from .fixtures import learned_parameters_csv

        install_artifacts(
            fake_client,
            "run-abc",
            {
                "binary/ele_fit_and_data.nc": write_spectrum(tmp_path / "e", "ele_fit_and_data.nc"),
                "csv/learned_parameters.csv": learned_parameters_csv(),
                "plots/a.png": b"x",
                "lineouts/a.png": b"x",
                "best/a.png": b"x",
                "worst/a.png": b"x",
            },
        )

        original = fake_client.list_artifacts
        listed: list[str] = []

        def counting(run_id, path=""):
            listed.append(path)
            return original(run_id, path)

        fake_client.list_artifacts = counting

        for endpoint in ("spectrogram", "lineout?index=0", "lineout?index=1"):
            listed.clear()
            assert dataset_client.get(f"/api/runs/run-abc/{endpoint}").status_code == 200
            assert listed == ["binary"], f"{endpoint} walked {listed} instead of just binary/"

    def test_describe_still_reports_profiles_and_sigmas(self, dataset_client, one_d_run):
        """The narrower listing must not cost /datasets its fuller answer."""
        body = dataset_client.get(f"/api/runs/{one_d_run}/datasets").json()
        assert body["profiles_available"] is True
        assert body["sigmas_available"] is True

    def test_axis_column_is_matched_by_name_not_position(self, dataset_service):
        """A parameter carrying a unit must not be mistaken for the lineout axis."""
        import pandas as pd

        frame = pd.DataFrame(
            {
                # Deliberately ahead of the real axis, which is what the
                # positional heuristic alone would fall for.
                "Te_electron (keV)": np.linspace(5.0, 1.0, 6),
                TEMPORAL_AXIS: np.linspace(-100.0, 100.0, 6),
                "ne_electron": np.linspace(0.1, 0.3, 6),
            }
        )
        assert dataset_service._profile_axis_column(frame, TEMPORAL_AXIS) == TEMPORAL_AXIS

    def test_axis_column_falls_back_to_a_monotonic_candidate(self, dataset_service):
        """With no hint, prefer monotonicity: a lineout axis always is."""
        import pandas as pd

        frame = pd.DataFrame(
            {
                "Te_electron (keV)": np.array([5.0, 1.0, 4.0, 2.0, 3.0, 0.5]),
                TEMPORAL_AXIS: np.linspace(-100.0, 100.0, 6),
            }
        )
        assert dataset_service._profile_axis_column(frame, None) == TEMPORAL_AXIS

    def test_profiles_uses_the_dataset_axis_name(self, dataset_client, one_d_run):
        body = dataset_client.get(f"/api/runs/{one_d_run}/profiles").json()
        assert body["x_label"] == TEMPORAL_AXIS
        assert TEMPORAL_AXIS not in {series["name"] for series in body["series"]}
