import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np


class _DummyTime:
    def __init__(self, *args, **kwargs):
        pass


class _DummySkyCoord:
    def __init__(self, *args, **kwargs):
        pass


class _DummyEarthLocation:
    @classmethod
    def of_site(cls, *args, **kwargs):
        return cls()


sys.modules.setdefault(
    "jax",
    types.SimpleNamespace(
        __version__="0.0-test",
        default_backend=lambda: "cpu",
        devices=lambda: [],
    ),
)
sys.modules.setdefault("astropy", types.ModuleType("astropy"))
sys.modules.setdefault("astropy.io", types.ModuleType("astropy.io"))
sys.modules.setdefault("astropy.io.fits", types.ModuleType("astropy.io.fits"))
sys.modules.setdefault("astropy.time", types.SimpleNamespace(Time=_DummyTime))
sys.modules.setdefault(
    "astropy.coordinates",
    types.SimpleNamespace(SkyCoord=_DummySkyCoord, EarthLocation=_DummyEarthLocation),
)
sys.modules.setdefault(
    "astropy.units",
    types.SimpleNamespace(hourangle="hourangle", deg="deg"),
)

from dataio import prepare_retrieval_timeseries as prep


def _planet_cfg(**overrides):
    cfg = {
        "period": 3.0,
        "duration": 0.12,
        "epoch": 2450000.0,
        "tau": 0.02,
        "RA": "00h00m00.00s",
        "Dec": "+00d00m00.00s",
        "rp_rs": 0.1,
        "b": 0.2,
        "lambda_angle": 5.0,
        "a_rs": 6.0,
        "v_sini_star": 100.0,
        "gamma1": 0.3,
        "gamma2": 0.2,
    }
    cfg.update(overrides)
    return cfg


def _load_result(data=None):
    if data is None:
        data = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.5, 2.5, 3.5, 4.5],
            ],
            dtype=float,
        )
    wave = np.asarray(
        [
            [5000.0, 5001.0, 5002.0, 5003.0],
            [5000.0, 5001.0, 5002.0, 5003.0],
        ],
        dtype=float,
    )
    sigma = np.full_like(data, 0.1)
    jd = np.asarray([2450000.10, 2450000.20], dtype=float)
    snr = np.asarray([100.0, 110.0], dtype=float)
    exptime = np.asarray([600.0, 600.0], dtype=float)
    airmass = np.asarray([1.1, 1.2], dtype=float)
    return (
        wave,
        data,
        sigma,
        jd,
        snr,
        exptime,
        airmass,
        data.shape[0],
        data.shape[1],
    )


class BuildShadowInputsTests(unittest.TestCase):
    def test_build_shadow_inputs_returns_expected_mapping(self):
        phase = np.asarray([-0.01, 0.0, 0.01], dtype=float)
        shadow_inputs, status = prep._build_shadow_inputs(_planet_cfg(), phase)

        self.assertIsNotNone(shadow_inputs)
        self.assertFalse(status["applied"])
        self.assertIsNone(status["skip_reason"])
        np.testing.assert_allclose(shadow_inputs["phase"], phase)
        self.assertEqual(
            shadow_inputs["planet_params"],
            {
                "rp_rs": 0.1,
                "b": 0.2,
                "lambda_angle": 5.0,
                "a_rs": 6.0,
                "period": 3.0,
            },
        )
        self.assertEqual(
            shadow_inputs["stellar_params"],
            {
                "vsini": 100.0,
                "gamma1": 0.3,
                "gamma2": 0.2,
            },
        )

    def test_build_shadow_inputs_reports_missing_params(self):
        shadow_inputs, status = prep._build_shadow_inputs(
            _planet_cfg(lambda_angle=np.nan),
            np.asarray([-0.01, 0.01], dtype=float),
        )

        self.assertIsNone(shadow_inputs)
        self.assertFalse(status["applied"])
        self.assertIn("lambda_angle", status["skip_reason"])


class PrepareRetrievalTimeseriesMainTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tempdir.name) / "prep"

    def tearDown(self):
        self.tempdir.cleanup()

    def _argv(self, *extra_args):
        return [
            "prepare_retrieval_timeseries.py",
            "--epoch",
            "20200101",
            "--planet",
            "KELT-20b",
            "--arm",
            "red",
            "--phase-bin",
            "all",
            "--output-dir",
            str(self.output_dir),
            *extra_args,
        ]

    def _run_main(self, argv, *, planet_cfg, result, remove_output, phase, chunk_indices=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            planet_mock = stack.enter_context(
                patch("dataio.prepare_retrieval_timeseries._planet_config", return_value=planet_cfg)
            )
            stack.enter_context(patch("dataio.prepare_retrieval_timeseries._load_data", return_value=result))
            stack.enter_context(patch("dataio.prepare_retrieval_timeseries.get_orbital_phase", return_value=phase))
            remove_mock = stack.enter_context(
                patch(
                    "dataio.prepare_retrieval_timeseries.remove_doppler_shadow",
                    return_value=remove_output,
                )
            )
            if chunk_indices is not None:
                stack.enter_context(
                    patch(
                        "dataio.prepare_retrieval_timeseries.get_sysrem_chunk_indices",
                        return_value=chunk_indices,
                    )
                )
            rc = prep.main()
        return rc, remove_mock, planet_mock

    def test_main_applies_shadow_and_records_metadata(self):
        base_result = (_load_result(), {})
        corrected = base_result[0][1] - 0.25

        rc, remove_mock, planet_mock = self._run_main(
            self._argv("--ephemeris", "Singh24"),
            planet_cfg=_planet_cfg(),
            result=base_result,
            remove_output=(corrected, np.full_like(corrected, 0.25), {"scaling": 1.5}),
            phase=np.asarray([-0.01, 0.01], dtype=float),
        )

        self.assertEqual(rc, 0)
        planet_mock.assert_called_once_with("KELT-20b", "Singh24")
        np.testing.assert_allclose(np.load(self.output_dir / "data.npy"), corrected)
        metadata = json.loads((self.output_dir / "timeseries_prep.json").read_text())
        self.assertEqual(metadata["ephemeris"], "Singh24")
        self.assertTrue(metadata["doppler_shadow_applied"])
        self.assertEqual(metadata["doppler_shadow_skip_reason"], None)
        self.assertAlmostEqual(metadata["doppler_shadow_scaling"], 1.5)
        remove_mock.assert_called_once()

    def test_main_skips_shadow_when_required_params_are_missing(self):
        base_result = (_load_result(), {})

        rc, remove_mock, _planet_mock = self._run_main(
            self._argv(),
            planet_cfg=_planet_cfg(lambda_angle=np.nan),
            result=base_result,
            remove_output=(None, None, {"scaling": 99.0}),
            phase=np.asarray([-0.01, 0.01], dtype=float),
        )

        self.assertEqual(rc, 0)
        np.testing.assert_allclose(np.load(self.output_dir / "data.npy"), base_result[0][1])
        metadata = json.loads((self.output_dir / "timeseries_prep.json").read_text())
        self.assertFalse(metadata["doppler_shadow_applied"])
        self.assertIn("lambda_angle", metadata["doppler_shadow_skip_reason"])
        self.assertIsNone(metadata["doppler_shadow_scaling"])
        remove_mock.assert_not_called()

    def test_main_skips_shadow_when_no_subtract_median_is_used(self):
        base_result = (_load_result(), {})

        rc, remove_mock, _planet_mock = self._run_main(
            self._argv("--no-subtract-median"),
            planet_cfg=_planet_cfg(),
            result=base_result,
            remove_output=(None, None, {"scaling": 99.0}),
            phase=np.asarray([-0.01, 0.01], dtype=float),
        )

        self.assertEqual(rc, 0)
        np.testing.assert_allclose(np.load(self.output_dir / "data.npy"), base_result[0][1])
        metadata = json.loads((self.output_dir / "timeseries_prep.json").read_text())
        self.assertFalse(metadata["doppler_shadow_applied"])
        self.assertEqual(metadata["doppler_shadow_skip_reason"], "subtract_median_disabled")
        self.assertIsNone(metadata["doppler_shadow_scaling"])
        remove_mock.assert_not_called()

    def test_main_applies_shadow_with_sysrem_outputs_present(self):
        base = _load_result()
        corrected = base[1] - 0.1
        extras = {
            "U_sysrem": np.ones((2, 1, 1), dtype=float),
        }
        chunk_indices = (
            np.asarray(["all"], dtype="U32"),
            (np.arange(4, dtype=int),),
            np.zeros(4, dtype=bool),
        )

        rc, remove_mock, _planet_mock = self._run_main(
            self._argv("--run-sysrem"),
            planet_cfg=_planet_cfg(),
            result=(base, extras),
            remove_output=(corrected, np.full_like(corrected, 0.1), {"scaling": 0.8}),
            phase=np.asarray([-0.01, 0.01], dtype=float),
            chunk_indices=chunk_indices,
        )

        self.assertEqual(rc, 0)
        np.testing.assert_allclose(np.load(self.output_dir / "data.npy"), corrected)
        self.assertTrue((self.output_dir / "U_sysrem.npz").exists())
        metadata = json.loads((self.output_dir / "timeseries_prep.json").read_text())
        self.assertTrue(metadata["doppler_shadow_applied"])
        self.assertAlmostEqual(metadata["doppler_shadow_scaling"], 0.8)
        remove_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
