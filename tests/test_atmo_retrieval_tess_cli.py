import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class _FakeTessTransitFitConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if not hasattr(self, "sectors"):
            self.sectors = None
        if not hasattr(self, "exptime_s"):
            self.exptime_s = 120


class AtmoRetrievalTessCliTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_root = Path(__file__).resolve().parents[1]

    def tearDown(self):
        self.tempdir.cleanup()

    def _fake_modules(self):
        temp_root = Path(self.tempdir.name)

        config_mod = types.ModuleType("config")
        config_mod.CONFIG_PROFILE_ENVVAR = "ATMO_RETRIEVAL_CONFIG_PROFILE"
        config_mod.DEFAULT_DATA_FORMAT = "timeseries"
        config_mod.RESOLUTION_MODE = "standard"
        config_mod.PT_PROFILE_DEFAULT = "guillot"
        config_mod.CHEMISTRY_MODEL_DEFAULT = "free"
        config_mod.DEFAULT_PHASE_MODE = "global"
        config_mod.NLAYER = 50
        config_mod.N_SPECTRAL_POINTS = 2048
        config_mod.PRESSURE_TOP = 1.0e-6
        config_mod.PRESSURE_BTM = 1.0
        config_mod.T_LOW = 500
        config_mod.T_HIGH = 5000
        config_mod.QUICK_SVI_STEPS = 10
        config_mod.QUICK_MCMC_WARMUP = 5
        config_mod.QUICK_MCMC_SAMPLES = 5
        config_mod.QUICK_MCMC_CHAINS = 1
        config_mod.USE_DEFAULT_SPECIES = False
        config_mod.DEFAULT_SPECIES = {"atoms": [], "molecules": []}
        config_mod.ATOMIC_SPECIES = {"Fe I": object()}
        config_mod.MOLPATH_HITEMP = {"H2O": "dummy"}
        config_mod.MOLPATH_EXOMOL = {}
        config_mod.PLANET = "KELT-20b"
        config_mod.EPHEMERIS = "DefaultEphem"
        config_mod.RETRIEVAL_MODE = "transmission"
        config_mod.OBSERVING_MODE = "red"
        config_mod.DIR_SAVE = str(temp_root / "output")
        config_mod.SVI_NUM_STEPS = 100
        config_mod.SVI_LEARNING_RATE = 1.0e-2
        config_mod.SVI_LR_DECAY_STEPS = None
        config_mod.SVI_LR_DECAY_RATE = None
        config_mod.MCMC_NUM_WARMUP = 20
        config_mod.MCMC_NUM_SAMPLES = 30
        config_mod.MCMC_NUM_CHAINS = 1
        config_mod.MCMC_CHAIN_METHOD = "sequential"
        config_mod.MCMC_REQUIRE_GPU_PER_CHAIN = False
        config_mod.FASTCHEM_PARAMETER_FILE = None
        config_mod.DATA_DIR = str(temp_root / "data")
        config_mod.TRANSMISSION_DATA = {}
        config_mod.EMISSION_DATA = {}
        config_mod._params = {
            "period": 3.4741,
            "epoch": 2459757.811176,
            "duration": 0.147565,
            "rp_rs": 0.111,
            "b": 0.5,
            "R_p": 1.7,
            "T_star": 8700,
            "RV_abs": -22.8,
        }

        def _set_runtime_config(name, value):
            setattr(config_mod, name, value)

        config_mod.set_runtime_config = _set_runtime_config
        config_mod.list_runtime_profiles = lambda: ["default"]
        config_mod.get_runtime_profile_name = lambda: "default"
        config_mod.apply_runtime_profile = lambda profile: None
        config_mod.get_output_dir = lambda: str(temp_root / "output")
        config_mod.get_data_dir = lambda epoch: str(temp_root / "data" / str(epoch))
        config_mod.get_full_arm_data_dirs = lambda epoch, mode=None: {
            "red": temp_root / "data" / str(epoch) / "red",
            "blue": temp_root / "data" / str(epoch) / "blue",
        }
        config_mod.get_transmission_paths = lambda epoch: {"epoch": epoch, "mode": "transmission"}
        config_mod.get_emission_paths = lambda epoch: {"epoch": epoch, "mode": "emission"}
        config_mod.get_wavelength_range = lambda: (4800.0, 6800.0)
        config_mod.get_resolution = lambda: 130000
        config_mod.get_params = lambda planet=None, ephemeris=None: dict(config_mod._params)

        pipeline_pkg = types.ModuleType("pipeline")
        pipeline_pkg.__path__ = []

        retrieval_mod = types.ModuleType("pipeline.retrieval")
        retrieval_mod.run_calls = []
        retrieval_mod.make_bandpass_constraints_from_tbl = lambda path: []
        retrieval_mod.make_joint_spectrum_component_from_tbl = lambda path: {"tbl_path": path}

        def _run_retrieval(**kwargs):
            retrieval_mod.run_calls.append(kwargs)

        retrieval_mod.run_retrieval = _run_retrieval

        retrieval_binned_mod = types.ModuleType("pipeline.retrieval_binned")
        retrieval_binned_mod.phase_calls = []
        retrieval_binned_mod.run_phase_binned_retrieval = lambda **kwargs: retrieval_binned_mod.phase_calls.append(kwargs)

        dataio_pkg = types.ModuleType("dataio")
        dataio_pkg.__path__ = []

        tess_mod = types.ModuleType("dataio.tess_photometry")
        tess_mod.fit_calls = []
        tess_mod.TessTransitFitConfig = _FakeTessTransitFitConfig

        def _fit_tess_transit_to_bandpass_constraint(config_obj, **kwargs):
            tess_mod.fit_calls.append((config_obj, kwargs))
            return types.SimpleNamespace(
                bandpass_constraint={
                    "name": kwargs["constraint_name"],
                    "mode": "transmission",
                    "observable": kwargs["observable"],
                    "value": 0.0123,
                    "sigma": 0.0004,
                }
            )

        tess_mod.fit_tess_transit_to_bandpass_constraint = _fit_tess_transit_to_bandpass_constraint

        return {
            "config": config_mod,
            "pipeline": pipeline_pkg,
            "pipeline.retrieval": retrieval_mod,
            "pipeline.retrieval_binned": retrieval_binned_mod,
            "dataio": dataio_pkg,
            "dataio.tess_photometry": tess_mod,
        }

    def _load_module(self, fake_modules):
        module_name = "atmo_retrieval_test_module"
        spec = importlib.util.spec_from_file_location(
            module_name,
            self.repo_root / "atmo_retrieval.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_main_fits_tess_transit_and_injects_constraint(self):
        fake_modules = self._fake_modules()
        with patch.dict(sys.modules, fake_modules, clear=False):
            module = self._load_module(fake_modules)
            retrieval_mod = fake_modules["pipeline.retrieval"]
            tess_mod = fake_modules["dataio.tess_photometry"]

            argv = [
                "atmo_retrieval.py",
                "--planet",
                "KELT-20b",
                "--mode",
                "transmission",
                "--epoch",
                "20200101",
                "--fit-tess-transit",
                "--tess-target",
                "TIC 123456789",
                "--tess-sector",
                "14",
                "--tess-sector",
                "15",
                "--tess-t0-bjd",
                "2458000.25",
                "--tess-quality-bitmask",
                "hard",
                "--tess-flux-column",
                "sap_flux",
                "--tess-observable",
                "transit_depth",
                "--tess-photon-weighted",
                "--tess-bandpass-tbl-output",
                str(Path(self.tempdir.name) / "tess.tbl"),
                "--svi-only",
                "--no-plots",
            ]

            with patch.object(sys, "argv", argv):
                rc = module.main()

        self.assertEqual(rc, 0)
        self.assertEqual(len(tess_mod.fit_calls), 1)
        fit_config, fit_kwargs = tess_mod.fit_calls[0]
        self.assertEqual(fit_config.target, "TIC 123456789")
        self.assertEqual(fit_config.sectors, (14, 15))
        self.assertEqual(fit_config.quality_bitmask, "hard")
        self.assertEqual(fit_config.flux_column, "sap_flux")
        self.assertAlmostEqual(fit_config.period_d, fake_modules["config"]._params["period"])
        self.assertAlmostEqual(
            fit_config.transit_duration_d,
            fake_modules["config"]._params["duration"],
        )
        self.assertAlmostEqual(
            fit_config.radius_ratio_guess,
            fake_modules["config"]._params["rp_rs"],
        )
        self.assertAlmostEqual(fit_config.impact_guess, fake_modules["config"]._params["b"])
        self.assertAlmostEqual(fit_config.t0_btjd, 1000.25)
        self.assertEqual(fit_kwargs["observable"], "transit_depth")
        self.assertEqual(fit_kwargs["constraint_name"], "tess_transit")
        self.assertTrue(fit_kwargs["photon_weighted"])

        self.assertEqual(len(retrieval_mod.run_calls), 1)
        retrieval_kwargs = retrieval_mod.run_calls[0]
        self.assertEqual(retrieval_kwargs["mode"], "transmission")
        self.assertEqual(len(retrieval_kwargs["bandpass_constraints"]), 1)
        self.assertEqual(
            retrieval_kwargs["bandpass_constraints"][0]["observable"],
            "transit_depth",
        )

    def test_main_defaults_tess_observable_to_transit_depth(self):
        fake_modules = self._fake_modules()
        with patch.dict(sys.modules, fake_modules, clear=False):
            module = self._load_module(fake_modules)
            retrieval_mod = fake_modules["pipeline.retrieval"]
            tess_mod = fake_modules["dataio.tess_photometry"]

            argv = [
                "atmo_retrieval.py",
                "--planet",
                "KELT-20b",
                "--mode",
                "transmission",
                "--epoch",
                "20200101",
                "--fit-tess-transit",
                "--tess-target",
                "TIC 123456789",
                "--tess-sector",
                "14",
                "--tess-t0-bjd",
                "2458000.25",
                "--svi-only",
                "--no-plots",
            ]

            with patch.object(sys, "argv", argv):
                rc = module.main()

        self.assertEqual(rc, 0)
        self.assertEqual(len(tess_mod.fit_calls), 1)
        _fit_config, fit_kwargs = tess_mod.fit_calls[0]
        self.assertEqual(fit_kwargs["observable"], "transit_depth")

        self.assertEqual(len(retrieval_mod.run_calls), 1)
        retrieval_kwargs = retrieval_mod.run_calls[0]
        self.assertEqual(
            retrieval_kwargs["bandpass_constraints"][0]["observable"],
            "transit_depth",
        )

    def test_main_builds_joint_hrs_components_for_extra_epochs(self):
        fake_modules = self._fake_modules()
        with patch.dict(sys.modules, fake_modules, clear=False):
            module = self._load_module(fake_modules)
            retrieval_mod = fake_modules["pipeline.retrieval"]

            argv = [
                "atmo_retrieval.py",
                "--planet",
                "KELT-20b",
                "--mode",
                "transmission",
                "--epoch",
                "20200101",
                "20200102",
                "20200103",
                "--svi-only",
                "--no-plots",
            ]

            with patch.object(sys, "argv", argv):
                rc = module.main()

        self.assertEqual(rc, 0)
        self.assertEqual(len(retrieval_mod.run_calls), 1)
        retrieval_kwargs = retrieval_mod.run_calls[0]
        self.assertEqual(retrieval_kwargs["epoch"], ["20200101", "20200102", "20200103"])

        joint_spectra = retrieval_kwargs["joint_spectra"]
        self.assertEqual(len(joint_spectra), 2)
        self.assertEqual(joint_spectra[0]["name"], "spectroscopy_red_20200102")
        self.assertEqual(
            joint_spectra[0]["data_dir"],
            str(Path(self.tempdir.name) / "data" / "20200102"),
        )
        self.assertEqual(joint_spectra[0]["data_format"], "timeseries")
        self.assertEqual(joint_spectra[0]["radial_velocity_mode"], "orbital")
        self.assertEqual(joint_spectra[0]["likelihood_kind"], "matched_filter")
        self.assertEqual(joint_spectra[1]["name"], "spectroscopy_red_20200103")

    def test_main_builds_full_arm_joint_hrs_components_for_extra_epochs(self):
        fake_modules = self._fake_modules()
        fake_modules["config"].OBSERVING_MODE = "full"
        with patch.dict(sys.modules, fake_modules, clear=False):
            module = self._load_module(fake_modules)
            retrieval_mod = fake_modules["pipeline.retrieval"]

            argv = [
                "atmo_retrieval.py",
                "--planet",
                "KELT-20b",
                "--mode",
                "emission",
                "--wavelength-range",
                "full",
                "--epoch",
                "20200101",
                "20200102",
                "--svi-only",
                "--no-plots",
            ]

            with patch.object(sys, "argv", argv):
                rc = module.main()

        self.assertEqual(rc, 0)
        self.assertEqual(len(retrieval_mod.run_calls), 1)
        retrieval_kwargs = retrieval_mod.run_calls[0]
        self.assertEqual(retrieval_kwargs["epoch"], ["20200101", "20200102"])

        joint_spectra = retrieval_kwargs["joint_spectra"]
        self.assertEqual(len(joint_spectra), 2)
        self.assertEqual(joint_spectra[0]["name"], "spectroscopy_red_20200102")
        self.assertEqual(
            joint_spectra[0]["data_dir"],
            str(Path(self.tempdir.name) / "data" / "20200102" / "red"),
        )
        self.assertEqual(joint_spectra[1]["name"], "spectroscopy_blue_20200102")
        self.assertEqual(
            joint_spectra[1]["data_dir"],
            str(Path(self.tempdir.name) / "data" / "20200102" / "blue"),
        )

    def test_quality_bitmask_parser_accepts_integer_bitmask(self):
        fake_modules = self._fake_modules()
        with patch.dict(sys.modules, fake_modules, clear=False):
            module = self._load_module(fake_modules)

        self.assertEqual(module._parse_tess_quality_bitmask("hard"), "hard")
        self.assertEqual(module._parse_tess_quality_bitmask("175"), 175)

        with self.assertRaisesRegex(ValueError, "must be one of"):
            module._parse_tess_quality_bitmask("badmask")

    def test_tess_fit_rejects_emission_mode(self):
        fake_modules = self._fake_modules()
        with patch.dict(sys.modules, fake_modules, clear=False):
            module = self._load_module(fake_modules)
            parser = module.create_parser()
            args = parser.parse_args(
                [
                    "--planet",
                    "KELT-20b",
                    "--mode",
                    "emission",
                    "--epoch",
                    "20200101",
                    "--fit-tess-transit",
                ]
            )

            with self.assertRaisesRegex(ValueError, "only supported for transmission"):
                module._fit_tess_transit_constraint(args, fake_modules["config"].get_params())


if __name__ == "__main__":
    unittest.main()
