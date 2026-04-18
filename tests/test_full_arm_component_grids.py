from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _real_package(name: str) -> bool:
    try:
        module = importlib.import_module(name)
    except Exception:
        return False
    return hasattr(module, "__path__")


def _deps_are_real() -> bool:
    return all(_real_package(name) for name in ("jax", "astropy", "exojax"))


def _fake_component_bundle(name: str, *, mode: str = "transmission", region_name: str = "terminator"):
    observation_config = SimpleNamespace(
        name=name,
        region_name=region_name,
        mode=mode,
        radial_velocity_mode="orbital",
        sample_prefix=name,
        opa_mols={},
        opa_atoms={},
        opa_cias={},
    )
    return SimpleNamespace(
        name=name,
        observation_config=observation_config,
        observation_inputs=SimpleNamespace(),
        opa_mols={},
        opa_atoms={},
        opa_cias={},
    )


def _fake_retrieval_component_bundle(
    name: str,
    *,
    wav_obs: np.ndarray,
    mode: str = "transmission",
    region_name: str = "terminator",
):
    data = np.asarray(
        [
            np.linspace(0.1, 0.3, wav_obs.size, dtype=float),
            np.linspace(0.4, 0.6, wav_obs.size, dtype=float),
        ],
        dtype=float,
    )
    sigma = np.full_like(data, 0.01)
    phase = np.asarray([-0.01, 0.01], dtype=float)
    observation_config = SimpleNamespace(
        name=name,
        region_name=region_name,
        mode=mode,
        radial_velocity_mode="orbital",
        sample_prefix=name,
        beta_inst=1.0,
        likelihood_kind="matched_filter",
        Tstar=None,
        subtract_per_exposure_mean=False,
        apply_sysrem=False,
    )
    return SimpleNamespace(
        name=name,
        wav_obs=np.asarray(wav_obs, dtype=float),
        data=data,
        sigma=sigma,
        phase=phase,
        sysrem=None,
        inst_nus=np.linspace(1.0, 2.0, wav_obs.size, dtype=float),
        nu_grid=np.linspace(10.0, 10.0 + wav_obs.size - 1, wav_obs.size, dtype=float),
        sop_rot=None,
        sop_inst=None,
        instrument_resolution=130000.0,
        opa_cias={},
        opa_mols={"Fe": object()},
        opa_atoms={},
        observation_config=observation_config,
        observation_inputs=SimpleNamespace(),
    )


class FullArmComponentGridTests(unittest.TestCase):
    def setUp(self):
        if not _deps_are_real():
            self.skipTest(
                "real jax/astropy/exojax packages are not available in sys.modules; "
                "run tests/test_full_arm_component_grids.py in the retrieval environment."
            )

    def test_run_retrieval_full_arm_uses_joint_loader_for_primary_and_skips_global_grid(self):
        import pipeline.retrieval as retrieval

        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            red_wav = np.asarray([7400.0, 7350.0, 7300.0], dtype=float)
            red_data = np.asarray([[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]], dtype=float)
            red_sigma = np.full_like(red_data, 0.01)
            red_phase = np.asarray([-0.01, 0.01], dtype=float)
            sysrem = retrieval.SysremInputBundle(
                U=np.zeros((2, 1), dtype=float),
                V=np.eye(2, dtype=float),
            )
            model_params = {
                "Kp": 150.0,
                "Kp_err": 5.0,
                "Kp_low": None,
                "Kp_high": None,
                "RV_abs": 0.0,
                "RV_abs_err": 1.0,
                "R_p": 1.0,
                "R_p_err": 0.1,
                "M_p": 1.0,
                "M_p_err": 0.1,
                "M_p_upper_3sigma": None,
                "M_star": 1.0,
                "R_star": 1.0,
                "R_star_err": 0.1,
                "T_star": 8000.0,
                "logg_star": 4.0,
                "Fe_H": 0.0,
                "T_eq": None,
                "Tirr_mean": None,
                "Tirr_std": None,
                "a": 0.05,
                "period": 3.0,
            }
            region = SimpleNamespace(
                name="terminator",
                sample_prefix=None,
                pt_profile="guillot",
            )
            load_joint_specs: list[dict[str, object]] = []
            shared_system_calls: list[dict[str, object]] = []
            build_regions_calls: list[list[str]] = []

            def fake_load_joint(spec, **_kwargs):
                load_joint_specs.append(dict(spec))
                return _fake_component_bundle(spec["name"])

            def fake_build_shared_system_config(**kwargs):
                shared_system_calls.append(dict(kwargs))
                return SimpleNamespace()

            def fake_build_atmosphere_regions(**kwargs):
                build_regions_calls.append(
                    [cfg.name for cfg in kwargs["observation_configs"]]
                )
                return (tuple([region]), {"terminator": region})

            with ExitStack() as stack:
                for manager in (
                    patch.object(retrieval.config, "OBSERVING_MODE", "full"),
                    patch.object(retrieval.config, "DIR_SAVE", str(temp_path)),
                    patch.object(
                        retrieval.config,
                        "create_timestamped_dir",
                        return_value=str(temp_path / "run"),
                    ),
                    patch.object(retrieval.config, "save_run_config", return_value=None),
                    patch.object(
                        retrieval.config,
                        "get_params",
                        return_value={"period": 3.0, "R_p": 1.0, "M_p": 1.0, "R_star": 1.0},
                    ),
                    patch.object(
                        retrieval.config,
                        "get_full_arm_data_dirs",
                        return_value={
                            "red": temp_path / "red",
                            "blue": temp_path / "blue",
                        },
                    ),
                    patch.object(retrieval.config, "get_resolution", return_value=130000),
                    patch.object(
                        retrieval,
                        "load_timeseries_data",
                        return_value=(red_wav, red_data, red_sigma, red_phase),
                    ),
                    patch.object(retrieval, "_normalize_phase", side_effect=lambda phase: phase),
                    patch.object(retrieval, "_load_sysrem_inputs", return_value={"dummy": True}),
                    patch.object(retrieval, "_validate_sysrem_inputs", return_value=sysrem),
                    patch.object(retrieval, "_coerce_model_params", return_value=model_params),
                    patch.object(retrieval, "_build_art_for_mode", return_value=object()),
                    patch.object(
                        retrieval,
                        "setup_wavenumber_grid",
                        side_effect=AssertionError("full-arm retrieval should not build a global HRS grid"),
                    ),
                    patch.object(
                        retrieval,
                        "_build_primary_spectroscopic_component",
                        side_effect=AssertionError("full-arm retrieval should not use the legacy primary builder"),
                    ),
                    patch.object(
                        retrieval,
                        "_load_joint_spectroscopic_component",
                        side_effect=fake_load_joint,
                    ),
                    patch.object(
                        retrieval,
                        "build_shared_system_config",
                        side_effect=fake_build_shared_system_config,
                    ),
                    patch.object(
                        retrieval,
                        "_build_atmosphere_regions",
                        side_effect=fake_build_atmosphere_regions,
                    ),
                    patch.object(retrieval, "create_joint_retrieval_model", return_value=object()),
                    patch.object(
                        retrieval,
                        "run_svi",
                        return_value=({}, np.asarray([0.0]), None, None, None),
                    ),
                ):
                    stack.enter_context(manager)
                retrieval.run_retrieval(
                    mode="transmission",
                    epoch="20200101",
                    data_format="timeseries",
                    pt_profile="guillot",
                    chemistry_model="free",
                    skip_svi=False,
                    svi_only=True,
                    no_plots=True,
                )

            self.assertEqual([spec["name"] for spec in load_joint_specs], ["spectroscopy_red", "spectroscopy_blue"])
            self.assertIn("wav_obs", load_joint_specs[0])
            self.assertIn("data", load_joint_specs[0])
            self.assertIn("sysrem", load_joint_specs[0])
            self.assertEqual(load_joint_specs[1]["data_dir"], str(temp_path / "blue"))
            self.assertEqual(
                shared_system_calls[0]["shared_velocity_component_names"],
                ("spectroscopy_red", "spectroscopy_blue"),
            )
            self.assertEqual(
                build_regions_calls[0],
                ["spectroscopy_red", "spectroscopy_blue"],
            )

    def test_load_joint_spectroscopic_component_uses_arm_specific_grid_bounds(self):
        import pipeline.retrieval as retrieval

        red_dir = Path("/tmp/red")
        blue_dir = Path("/tmp/blue")
        red_wav = np.asarray([7405.0, 7390.0, 7360.0], dtype=float)
        blue_wav = np.asarray([5410.0, 5395.0, 5380.0], dtype=float)
        data = np.asarray([[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]], dtype=float)
        sigma = np.full_like(data, 0.02)
        phase = np.asarray([-0.01, 0.01], dtype=float)
        sysrem = retrieval.SysremInputBundle(
            U=np.zeros((2, 1), dtype=float),
            V=np.eye(2, dtype=float),
        )
        grid_calls: list[tuple[float, float, int, str]] = []

        def fake_load_timeseries_data(data_dir):
            if Path(data_dir) == red_dir:
                return red_wav, data, sigma, phase
            if Path(data_dir) == blue_dir:
                return blue_wav, data, sigma, phase
            raise AssertionError(f"unexpected data_dir {data_dir}")

        def fake_setup_wavenumber_grid(start, stop, N, unit="AA"):
            grid_calls.append((float(start), float(stop), int(N), str(unit)))
            return np.asarray([1.0, 2.0]), np.asarray([5000.0, 5001.0]), 123456.0

        def fake_build_obs_config(**kwargs):
            return SimpleNamespace(
                name=kwargs["name"],
                region_name=kwargs["region_name"],
                mode=kwargs["mode"],
                radial_velocity_mode=kwargs["radial_velocity_mode"],
                sample_prefix=kwargs["sample_prefix"],
                opa_mols=kwargs["opa_mols"],
                opa_atoms=kwargs["opa_atoms"],
                opa_cias=kwargs["opa_cias"],
            )

        with ExitStack() as stack:
            for manager in (
                patch.object(retrieval, "load_timeseries_data", side_effect=fake_load_timeseries_data),
                patch.object(retrieval, "_normalize_phase", side_effect=lambda phase_arr: phase_arr),
                patch.object(retrieval, "_load_sysrem_inputs", return_value={"dummy": True}),
                patch.object(retrieval, "_validate_sysrem_inputs", return_value=sysrem),
                patch.object(retrieval, "setup_wavenumber_grid", side_effect=fake_setup_wavenumber_grid),
                patch.object(retrieval, "_preflight_grid_checks", return_value=None),
                patch.object(retrieval, "setup_spectral_operators", return_value=("rot", "inst", 0.1)),
                patch.object(retrieval, "_load_opacity_bundle", return_value=({}, {}, {})),
                patch.object(retrieval, "_load_phoenix_surface_flux_on_grid", return_value=None),
                patch.object(retrieval, "build_spectroscopic_observation_config", side_effect=fake_build_obs_config),
                patch.object(retrieval, "_build_spectroscopic_observation_inputs", return_value=SimpleNamespace()),
            ):
                stack.enter_context(manager)
            retrieval._load_joint_spectroscopic_component(
                {
                    "name": "spectroscopy_red",
                    "mode": "transmission",
                    "data_format": "timeseries",
                    "data_dir": str(red_dir),
                    "apply_sysrem": True,
                    "phase_mode": "global",
                    "radial_velocity_mode": "orbital",
                    "likelihood_kind": "matched_filter",
                    "subtract_per_exposure_mean": True,
                    "instrument_resolution": 130000,
                },
                default_mode="transmission",
                default_tstar=8000.0,
                default_logg_star=4.0,
                default_metallicity=0.0,
                default_mstar=1.0,
                default_rstar=1.0,
                default_phoenix_spectrum_path=None,
                default_phoenix_cache_dir=None,
            )
            retrieval._load_joint_spectroscopic_component(
                {
                    "name": "spectroscopy_blue",
                    "mode": "transmission",
                    "data_format": "timeseries",
                    "data_dir": str(blue_dir),
                    "apply_sysrem": True,
                    "phase_mode": "global",
                    "radial_velocity_mode": "orbital",
                    "likelihood_kind": "matched_filter",
                    "subtract_per_exposure_mean": True,
                    "instrument_resolution": 130000,
                },
                default_mode="transmission",
                default_tstar=8000.0,
                default_logg_star=4.0,
                default_metallicity=0.0,
                default_mstar=1.0,
                default_rstar=1.0,
                default_phoenix_spectrum_path=None,
                default_phoenix_cache_dir=None,
            )

        self.assertEqual(len(grid_calls), 2)
        self.assertEqual(
            grid_calls[0],
            (
                float(np.min(red_wav)) - retrieval.config.WAV_MIN_OFFSET,
                float(np.max(red_wav)) + retrieval.config.WAV_MAX_OFFSET,
                retrieval.config.N_SPECTRAL_POINTS,
                "AA",
            ),
        )
        self.assertEqual(
            grid_calls[1],
            (
                float(np.min(blue_wav)) - retrieval.config.WAV_MIN_OFFSET,
                float(np.max(blue_wav)) + retrieval.config.WAV_MAX_OFFSET,
                retrieval.config.N_SPECTRAL_POINTS,
                "AA",
            ),
        )

    def test_build_diagnostic_context_supports_full_arm_mode_and_is_symmetric(self):
        import pipeline.diagnostics as diagnostics

        model_params = {
            "Kp": 150.0,
            "Kp_err": 5.0,
            "Kp_low": None,
            "Kp_high": None,
            "RV_abs": 0.0,
            "RV_abs_err": 1.0,
            "R_p": 1.0,
            "R_p_err": 0.1,
            "M_p": 1.0,
            "M_p_err": 0.1,
            "M_p_upper_3sigma": None,
            "M_star": 1.0,
            "R_star": 1.0,
            "R_star_err": 0.1,
            "T_star": 8000.0,
            "logg_star": 4.0,
            "Fe_H": 0.0,
            "T_eq": None,
            "Tirr_mean": None,
            "Tirr_std": None,
            "a": 0.05,
            "period": 3.0,
        }
        region = SimpleNamespace(name="terminator", sample_prefix=None)
        load_joint_specs: list[dict[str, object]] = []
        shared_system_calls: list[dict[str, object]] = []

        def fake_load_joint(spec, **_kwargs):
            load_joint_specs.append(dict(spec))
            return _fake_component_bundle(spec["name"])

        def fake_build_shared_system_config(**kwargs):
            shared_system_calls.append(dict(kwargs))
            return SimpleNamespace()

        with ExitStack() as stack:
            for manager in (
                patch.object(diagnostics, "temporary_runtime_config", side_effect=lambda _overrides: nullcontext()),
                patch.object(
                    diagnostics.config,
                    "get_params",
                    return_value={"period": 3.0, "R_p": 1.0, "M_p": 1.0, "R_star": 1.0},
                ),
                patch.object(
                    diagnostics.config,
                    "get_full_arm_data_dirs",
                    return_value={
                        "red": Path("/tmp/red"),
                        "blue": Path("/tmp/blue"),
                    },
                ),
                patch.object(diagnostics.config, "get_resolution", return_value=130000),
                patch.object(diagnostics._retrieval, "_coerce_model_params", return_value=model_params),
                patch.object(diagnostics._retrieval, "_build_art_for_mode", return_value=object()),
                patch.object(
                    diagnostics._retrieval,
                    "_load_joint_spectroscopic_component",
                    side_effect=fake_load_joint,
                ),
                patch.object(
                    diagnostics._retrieval,
                    "build_shared_system_config",
                    side_effect=fake_build_shared_system_config,
                ),
                patch.object(
                    diagnostics._retrieval,
                    "_build_atmosphere_regions",
                    return_value=((region,), {"terminator": region}),
                ),
                patch.object(diagnostics._retrieval, "create_joint_retrieval_model", return_value=object()),
            ):
                stack.enter_context(manager)
            context = diagnostics.build_diagnostic_context(
                planet="KELT-20b",
                ephemeris="DefaultEphem",
                epoch="20200101",
                mode="transmission",
                pt_profile="guillot",
                chemistry_model="free",
                observing_mode="full",
                resolution_mode="hr",
                nlayer=10,
                n_spectral_points=50000,
            )

        self.assertEqual([spec["name"] for spec in load_joint_specs], ["spectroscopy_red", "spectroscopy_blue"])
        self.assertEqual(load_joint_specs[0]["data_dir"], "/tmp/red")
        self.assertEqual(load_joint_specs[1]["data_dir"], "/tmp/blue")
        self.assertEqual(
            shared_system_calls[0]["shared_velocity_component_names"],
            ("spectroscopy_red", "spectroscopy_blue"),
        )
        self.assertEqual(
            context.spectroscopic_component_names,
            ("spectroscopy_red", "spectroscopy_blue"),
        )
        self.assertEqual(
            set(context.spectroscopic_components.keys()),
            {"spectroscopy_red", "spectroscopy_blue"},
        )
        self.assertFalse(hasattr(context, "primary_component"))
        self.assertIs(context.shared_region_config, region)
        self.assertEqual(len(context.observation_configs), 2)
        self.assertEqual(
            set(context.model_inputs["observations"].keys()),
            {"spectroscopy_red", "spectroscopy_blue"},
        )

    def test_synthesize_processed_model_timeseries_uses_explicit_component_name(self):
        import pipeline.diagnostics as diagnostics

        shared_region = SimpleNamespace(sample_prefix="terminator")
        red_component = _fake_retrieval_component_bundle(
            "spectroscopy_red",
            wav_obs=np.asarray([7300.0, 7350.0, 7400.0], dtype=float),
        )
        blue_component = _fake_retrieval_component_bundle(
            "spectroscopy_blue",
            wav_obs=np.asarray([5300.0, 5350.0, 5400.0], dtype=float),
        )
        context = diagnostics.DiagnosticContext(
            planet="KELT-20b",
            ephemeris="DefaultEphem",
            epoch="20200101",
            mode="transmission",
            chemistry_model="free",
            pt_profile="guillot",
            model_params={"Kp": 150.0, "M_p": 1.0, "R_star": 1.0, "R_p": 1.0},
            shared_region_config=shared_region,
            shared_region_sample_prefix="terminator",
            shared_system=SimpleNamespace(),
            atmosphere_region_configs=(shared_region,),
            observation_configs=(
                red_component.observation_config,
                blue_component.observation_config,
            ),
            spectroscopic_component_names=("spectroscopy_red", "spectroscopy_blue"),
            spectroscopic_components={
                "spectroscopy_red": red_component,
                "spectroscopy_blue": blue_component,
            },
            component_sample_prefixes={
                "spectroscopy_red": "spectroscopy_red",
                "spectroscopy_blue": "spectroscopy_blue",
            },
            model_c=object(),
            model_inputs={"observations": {}},
        )
        compute_calls: list[dict[str, object]] = []
        synth_calls: list[dict[str, object]] = []

        def fake_compute(**kwargs):
            compute_calls.append(dict(kwargs))
            return {
                "params": {"Kp": 150.0},
                "dtau": np.ones((2, 3), dtype=float),
                "Tarr": np.ones((2,), dtype=float),
                "mmw": np.ones((2,), dtype=float),
            }

        def fake_synthesize(**kwargs):
            synth_calls.append(dict(kwargs))
            component = kwargs["component"]
            return np.asarray(component.data, dtype=float)

        with ExitStack() as stack:
            for manager in (
                patch.object(
                    diagnostics,
                    "merge_named_params",
                    side_effect=lambda _context, overrides=None: dict(overrides or {}),
                ),
                patch.object(
                    diagnostics,
                    "compute_atmospheric_state_from_posterior",
                    side_effect=fake_compute,
                ),
                patch.object(
                    diagnostics._retrieval,
                    "_synthesize_timeseries_from_atmospheric_state",
                    side_effect=fake_synthesize,
                ),
            ):
                stack.enter_context(manager)
            model_ts, atmo_state = diagnostics.synthesize_processed_model_timeseries(
                context,
                {"Kp": 155.0},
                component_name="spectroscopy_blue",
            )

        self.assertEqual(len(compute_calls), 1)
        self.assertTrue(np.array_equal(compute_calls[0]["nu_grid"], blue_component.nu_grid))
        self.assertEqual(compute_calls[0]["sample_prefix"], "terminator")
        self.assertEqual(len(synth_calls), 1)
        self.assertEqual(synth_calls[0]["component"].name, "spectroscopy_blue")
        self.assertEqual(synth_calls[0]["component_sample_prefix"], "spectroscopy_blue")
        self.assertTrue(np.array_equal(model_ts, blue_component.data))
        self.assertIn("params", atmo_state)

    def test_bandpass_constraint_uses_dedicated_grid(self):
        import pipeline.retrieval as retrieval

        wavelength_m = np.asarray([8.0e-7, 9.0e-7, 1.0e-6], dtype=float)
        response = np.asarray([0.1, 0.8, 0.2], dtype=float)
        grid_calls: list[tuple[float, float, int, str]] = []

        def fake_setup_wavenumber_grid(start, stop, N, unit="AA"):
            grid_calls.append((float(start), float(stop), int(N), str(unit)))
            return np.asarray([1.0, 2.0]), np.asarray([8000.0, 10000.0]), 654321.0

        def fake_build_bandpass_config(**kwargs):
            return SimpleNamespace(
                name=kwargs["name"],
                region_name=kwargs["region_name"],
                mode=kwargs["mode"],
                sample_prefix=kwargs["sample_prefix"],
                nu_grid=kwargs["nu_grid"],
            )

        with ExitStack() as stack:
            for manager in (
                patch.object(retrieval, "setup_wavenumber_grid", side_effect=fake_setup_wavenumber_grid),
                patch.object(retrieval, "_load_opacity_bundle", return_value=({}, {}, {})),
                patch.object(retrieval, "_load_phoenix_surface_flux_on_grid", return_value=None),
                patch.object(
                    retrieval,
                    "build_bandpass_observation_config",
                    side_effect=fake_build_bandpass_config,
                ),
            ):
                stack.enter_context(manager)
            bundle = retrieval._load_bandpass_constraint(
                {
                    "name": "tess_bandpass",
                    "mode": "transmission",
                    "region_name": "terminator",
                    "observable": "transit_depth",
                    "value": 0.0123,
                    "sigma": 0.0004,
                    "wavelength_m": wavelength_m,
                    "response": response,
                },
                default_mode="transmission",
                default_tstar=8000.0,
                default_logg_star=4.0,
                default_metallicity=0.0,
                default_mstar=1.0,
                default_rstar=1.0,
                default_semi_major_axis_au=0.05,
                default_phoenix_spectrum_path=None,
                default_phoenix_cache_dir=None,
            )

        self.assertEqual(len(grid_calls), 1)
        self.assertEqual(
            grid_calls[0],
            (
                float(np.min(wavelength_m) * 1.0e10) - retrieval.config.WAV_MIN_OFFSET,
                float(np.max(wavelength_m) * 1.0e10) + retrieval.config.WAV_MAX_OFFSET,
                retrieval.config.N_SPECTRAL_POINTS,
                "AA",
            ),
        )
        self.assertEqual(bundle.name, "tess_bandpass")
        self.assertEqual(bundle.observation_config.name, "tess_bandpass")

    def test_run_retrieval_full_arm_emits_per_component_output_products(self):
        import pipeline.retrieval as retrieval

        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            (temp_path / "run").mkdir()
            red_component = _fake_retrieval_component_bundle(
                "spectroscopy_red",
                wav_obs=np.asarray([7300.0, 7350.0, 7400.0], dtype=float),
            )
            blue_component = _fake_retrieval_component_bundle(
                "spectroscopy_blue",
                wav_obs=np.asarray([5300.0, 5350.0, 5400.0], dtype=float),
            )
            model_params = {
                "Kp": 150.0,
                "Kp_err": 5.0,
                "Kp_low": None,
                "Kp_high": None,
                "RV_abs": 0.0,
                "RV_abs_err": 1.0,
                "R_p": 1.0,
                "R_p_err": 0.1,
                "M_p": 1.0,
                "M_p_err": 0.1,
                "M_p_upper_3sigma": None,
                "M_star": 1.0,
                "R_star": 1.0,
                "R_star_err": 0.1,
                "T_star": 8000.0,
                "logg_star": 4.0,
                "Fe_H": 0.0,
                "T_eq": None,
                "Tirr_mean": None,
                "Tirr_std": None,
                "a": 0.05,
                "period": 3.0,
            }
            shared_region = SimpleNamespace(
                name="terminator",
                sample_prefix=None,
                pt_profile="guillot",
                art=object(),
                Tint_fixed=None,
            )
            plotted_spectra: list[str] = []
            contribution_paths: list[str] = []
            atmospheric_state_paths: list[str] = []

            def fake_load_joint(spec, **_kwargs):
                if spec["name"] == "spectroscopy_red":
                    return red_component
                if spec["name"] == "spectroscopy_blue":
                    return blue_component
                raise AssertionError(f"unexpected component spec {spec['name']}")

            class FakeMCMC:
                def __init__(self, *args, **kwargs):
                    self.args = args
                    self.kwargs = kwargs

                def run(self, *args, **kwargs):
                    return None

                def print_summary(self):
                    return None

                def get_samples(self):
                    return {"theta": np.asarray([1.0])}

            def fake_compute_atmo(**kwargs):
                nu_grid = np.asarray(kwargs["nu_grid"], dtype=float)
                return {
                    "dtau": np.ones((2, nu_grid.size), dtype=float),
                    "dtau_per_species": {"Fe": np.ones((2, nu_grid.size), dtype=float)},
                    "Tarr": np.ones((2,), dtype=float),
                    "pressure": np.ones((2,), dtype=float),
                    "dParr": np.ones((2,), dtype=float),
                    "mmw": np.ones((2,), dtype=float),
                    "vmrH2": np.ones((2,), dtype=float),
                    "vmrHe": np.ones((2,), dtype=float),
                }

            def fake_compute_model_timeseries_for_plot(**kwargs):
                component = kwargs["component"]
                atmo_state = kwargs.get("atmo_state")
                if atmo_state is None:
                    atmo_state = fake_compute_atmo(nu_grid=component.nu_grid)
                return np.asarray(component.data, dtype=float), atmo_state

            def fake_plot_transmission_spectrum(**kwargs):
                plotted_spectra.append(str(kwargs["save_path"]))

            def fake_plot_contribution_function(**kwargs):
                contribution_paths.append(str(kwargs["save_path"]))

            def fake_plot_contribution_per_species(**kwargs):
                contribution_paths.append(str(kwargs["save_path"]))

            def fake_plot_contribution_combined(**kwargs):
                contribution_paths.append(str(kwargs["save_path"]))

            def fake_np_savez(path, *args, **kwargs):
                atmospheric_state_paths.append(str(path))

            with ExitStack() as stack:
                for manager in (
                    patch.object(retrieval.config, "OBSERVING_MODE", "full"),
                    patch.object(retrieval.config, "DIR_SAVE", str(temp_path)),
                    patch.object(
                        retrieval.config,
                        "create_timestamped_dir",
                        return_value=str(temp_path / "run"),
                    ),
                    patch.object(retrieval.config, "save_run_config", return_value=None),
                    patch.object(
                        retrieval.config,
                        "get_params",
                        return_value={"period": 3.0, "R_p": 1.0, "M_p": 1.0, "R_star": 1.0},
                    ),
                    patch.object(
                        retrieval.config,
                        "get_full_arm_data_dirs",
                        return_value={
                            "red": temp_path / "red",
                            "blue": temp_path / "blue",
                        },
                    ),
                    patch.object(retrieval.config, "get_resolution", return_value=130000),
                    patch.object(
                        retrieval,
                        "load_timeseries_data",
                        return_value=(
                            red_component.wav_obs,
                            red_component.data,
                            red_component.sigma,
                            red_component.phase,
                        ),
                    ),
                    patch.object(retrieval, "_normalize_phase", side_effect=lambda phase: phase),
                    patch.object(retrieval, "_load_sysrem_inputs", return_value={"dummy": True}),
                    patch.object(
                        retrieval,
                        "_validate_sysrem_inputs",
                        return_value=retrieval.SysremInputBundle(
                            U=np.zeros((2, 1), dtype=float),
                            V=np.eye(2, dtype=float),
                        ),
                    ),
                    patch.object(retrieval, "_coerce_model_params", return_value=model_params),
                    patch.object(retrieval, "_build_art_for_mode", return_value=object()),
                    patch.object(retrieval, "_load_joint_spectroscopic_component", side_effect=fake_load_joint),
                    patch.object(retrieval, "build_shared_system_config", return_value=SimpleNamespace()),
                    patch.object(
                        retrieval,
                        "_build_atmosphere_regions",
                        return_value=((shared_region,), {"terminator": shared_region}),
                    ),
                    patch.object(retrieval, "create_joint_retrieval_model", return_value=object()),
                    patch.object(retrieval, "_validate_mcmc_device_layout", return_value=None),
                    patch.object(retrieval, "NUTS", return_value=SimpleNamespace()),
                    patch.object(retrieval, "MCMC", side_effect=lambda *a, **k: FakeMCMC(*a, **k)),
                    patch.object(retrieval, "save_retrieval_corner_plots", return_value=None),
                    patch.object(retrieval, "plot_svi_loss", return_value=None),
                    patch.object(retrieval, "plot_temperature_profile", return_value=None),
                    patch.object(
                        retrieval,
                        "_compute_model_timeseries_for_plot",
                        side_effect=fake_compute_model_timeseries_for_plot,
                    ),
                    patch.object(
                        retrieval,
                        "compute_atmospheric_state_from_posterior",
                        side_effect=fake_compute_atmo,
                    ),
                    patch.object(retrieval, "plot_transmission_spectrum", side_effect=fake_plot_transmission_spectrum),
                    patch.object(retrieval, "plot_contribution_function", side_effect=fake_plot_contribution_function),
                    patch.object(
                        retrieval,
                        "plot_contribution_per_species",
                        side_effect=fake_plot_contribution_per_species,
                    ),
                    patch.object(
                        retrieval,
                        "plot_contribution_combined",
                        side_effect=fake_plot_contribution_combined,
                    ),
                    patch.object(retrieval.np, "savez", side_effect=fake_np_savez),
                    patch.object(retrieval.jnp, "savez", return_value=None),
                ):
                    stack.enter_context(manager)
                retrieval.run_retrieval(
                    mode="transmission",
                    epoch="20200101",
                    data_format="timeseries",
                    pt_profile="guillot",
                    chemistry_model="free",
                    skip_svi=True,
                    no_plots=False,
                    compute_contribution=True,
                )

        self.assertEqual(
            sorted(Path(path).name for path in plotted_spectra),
            [
                "transmission_spectrum_spectroscopy_blue.png",
                "transmission_spectrum_spectroscopy_red.png",
            ],
        )
        self.assertEqual(
            sorted(Path(path).name for path in atmospheric_state_paths),
            [
                "atmospheric_state_spectroscopy_blue.npz",
                "atmospheric_state_spectroscopy_red.npz",
            ],
        )
        self.assertEqual(
            sorted(Path(path).name for path in contribution_paths),
            [
                "contribution_combined_spectroscopy_blue.pdf",
                "contribution_combined_spectroscopy_red.pdf",
                "contribution_function_spectroscopy_blue.pdf",
                "contribution_function_spectroscopy_red.pdf",
                "contribution_per_species_spectroscopy_blue.pdf",
                "contribution_per_species_spectroscopy_red.pdf",
            ],
        )

    def test_run_memory_profile_full_mode_aggregates_per_arm_results(self):
        import pipeline.memory_profile as memory_profile

        calls: list[dict[str, object]] = []

        def fake_single(**kwargs):
            calls.append(dict(kwargs))
            if kwargs["wav_min"] == 6200.0:
                return memory_profile.ProfileResult(
                    est_art_bytes=10.0,
                    est_opa_bytes=100.0,
                    est_total_bytes=110.0,
                    peak_gpu_used_bytes=1.0,
                    peak_gpu_label="After opacities",
                )
            return memory_profile.ProfileResult(
                est_art_bytes=20.0,
                est_opa_bytes=200.0,
                est_total_bytes=220.0,
                peak_gpu_used_bytes=2.0,
                peak_gpu_label="After opacities",
            )

        def fake_get_wavelength_range(mode=None):
            if mode == "red":
                return 6200.0, 7400.0
            if mode == "blue":
                return 4800.0, 5400.0
            return 4752.0, 7427.0

        with ExitStack() as stack:
            for manager in (
                patch.object(memory_profile.config, "OBSERVING_MODE", "full"),
                patch.object(memory_profile.config, "get_wavelength_range", side_effect=fake_get_wavelength_range),
                patch.object(
                    memory_profile,
                    "_run_single_memory_profile_for_range",
                    side_effect=fake_single,
                ),
            ):
                stack.enter_context(manager)
            result = memory_profile.run_memory_profile(mode="transmission")

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["wav_min"], 6200.0)
        self.assertEqual(calls[0]["wav_max"], 7400.0)
        self.assertEqual(calls[1]["wav_min"], 4800.0)
        self.assertEqual(calls[1]["wav_max"], 5400.0)
        self.assertEqual(result.est_art_bytes, 30.0)
        self.assertEqual(result.est_opa_bytes, 300.0)
        self.assertEqual(result.est_total_bytes, 330.0)
        self.assertEqual(result.peak_gpu_used_bytes, 2.0)
        self.assertEqual(set(result.component_results.keys()), {"red", "blue"})

    def test_run_memory_sweep_full_mode_uses_arm_ranges_only(self):
        import pipeline.memory_profile as memory_profile

        wavelength_calls: list[str | None] = []
        profile_calls: list[dict[str, object]] = []

        def fake_get_wavelength_range(mode=None):
            wavelength_calls.append(mode)
            if mode == "red":
                return 6200.0, 7400.0
            if mode == "blue":
                return 4800.0, 5400.0
            raise AssertionError("full-arm memory sweep should not request a global wavelength range")

        def fake_run_memory_profile(**kwargs):
            profile_calls.append(dict(kwargs))
            return memory_profile.ProfileResult()

        with ExitStack() as stack:
            for manager in (
                patch.object(memory_profile.config, "OBSERVING_MODE", "full"),
                patch.object(memory_profile.config, "get_wavelength_range", side_effect=fake_get_wavelength_range),
                patch.object(memory_profile, "run_memory_profile", side_effect=fake_run_memory_profile),
            ):
                stack.enter_context(manager)
            memory_profile.run_memory_sweep(
                mode="transmission",
                nfree_values=[8],
                nlayer_values=[50],
                nspec_values=[150000],
            )

        self.assertEqual(wavelength_calls, ["red", "blue"])
        self.assertEqual(
            profile_calls,
            [
                {"mode": "transmission", "nfree": 8},
                {"mode": "transmission", "nlayer": 50},
                {"mode": "transmission", "n_spectral_points": 150000},
            ],
        )


if __name__ == "__main__":
    unittest.main()
