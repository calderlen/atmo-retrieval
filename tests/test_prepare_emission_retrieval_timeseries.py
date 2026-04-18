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

from dataio import prepare_emission_retrieval_timeseries as prep


def _planet_cfg(**overrides):
    cfg = {
        "period": 3.0,
        "duration": 0.12,
        "epoch": 2450000.0,
        "RA": "00h00m00.00s",
        "Dec": "+00d00m00.00s",
    }
    cfg.update(overrides)
    return cfg


def _load_result(data=None):
    if data is None:
        data = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.5, 2.5, 3.5, 4.5],
                [2.0, 3.0, 4.0, 5.0],
            ],
            dtype=float,
        )
    wave = np.asarray(
        [
            [5000.0, 5001.0, 5002.0, 5003.0],
            [5000.0, 5001.0, 5002.0, 5003.0],
            [5000.0, 5001.0, 5002.0, 5003.0],
        ],
        dtype=float,
    )
    sigma = np.full_like(data, 0.1)
    jd = np.asarray([2450000.10, 2450000.20, 2450000.30], dtype=float)
    snr = np.asarray([100.0, 110.0, 120.0], dtype=float)
    exptime = np.asarray([600.0, 600.0, 600.0], dtype=float)
    airmass = np.asarray([1.1, 1.2, 1.3], dtype=float)
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


class EmissionPhaseSelectionTests(unittest.TestCase):
    def test_dayside_and_nightside_are_complementary_half_orbits(self):
        phase = np.asarray([-0.40, -0.10, 0.10, 0.40], dtype=float)

        dayside = prep._phase_selection_mask(phase, phase_bin="dayside", planet_params=_planet_cfg())
        nightside = prep._phase_selection_mask(phase, phase_bin="nightside", planet_params=_planet_cfg())

        np.testing.assert_array_equal(dayside, np.asarray([True, False, False, True]))
        np.testing.assert_array_equal(nightside, np.asarray([False, True, True, False]))

    def test_pre_and_post_eclipse_use_phase_modulo_one(self):
        phase = np.asarray([-0.49, -0.01, 0.01, 0.49], dtype=float)

        pre = prep._phase_selection_mask(phase, phase_bin="pre_eclipse", planet_params=_planet_cfg())
        post = prep._phase_selection_mask(phase, phase_bin="post_eclipse", planet_params=_planet_cfg())

        np.testing.assert_array_equal(pre, np.asarray([False, False, True, True]))
        np.testing.assert_array_equal(post, np.asarray([True, True, False, False]))

    def test_eclipse_selection_wraps_across_negative_half_phase(self):
        phase = np.asarray([-0.48, -0.44, 0.10], dtype=float)
        planet_cfg = _planet_cfg(duration=1.2, period=3.0)

        eclipse = prep._phase_selection_mask(phase, phase_bin="eclipse", planet_params=planet_cfg)

        np.testing.assert_array_equal(eclipse, np.asarray([True, True, False]))


class PrepareEmissionRetrievalTimeseriesMainTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tempdir.name) / "prep"

    def tearDown(self):
        self.tempdir.cleanup()

    def _argv(self, *extra_args):
        return [
            "prepare_emission_retrieval_timeseries.py",
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

    def _run_main(self, argv, *, planet_cfg, result, phase, chunk_indices=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            planet_mock = stack.enter_context(
                patch("dataio.prepare_emission_retrieval_timeseries._planet_config", return_value=planet_cfg)
            )
            stack.enter_context(patch("dataio.prepare_emission_retrieval_timeseries._load_data", return_value=result))
            stack.enter_context(patch("dataio.prepare_emission_retrieval_timeseries.get_orbital_phase", return_value=phase))
            if chunk_indices is not None:
                stack.enter_context(
                    patch(
                        "dataio.prepare_emission_retrieval_timeseries.get_sysrem_chunk_indices",
                        return_value=chunk_indices,
                    )
                )
            rc = prep.main()
        return rc, planet_mock

    def test_main_writes_emission_metadata_and_keeps_all_exposures(self):
        base_result = (_load_result(), {})

        rc, planet_mock = self._run_main(
            self._argv("--ephemeris", "Gaudi17"),
            planet_cfg=_planet_cfg(),
            result=base_result,
            phase=np.asarray([-0.49, -0.20, 0.10], dtype=float),
        )

        self.assertEqual(rc, 0)
        planet_mock.assert_called_once_with("KELT-20b", "Gaudi17")
        np.testing.assert_allclose(np.load(self.output_dir / "data.npy"), base_result[0][1])
        metadata = json.loads((self.output_dir / "timeseries_prep.json").read_text())
        self.assertEqual(metadata["mode"], "emission")
        self.assertEqual(metadata["ephemeris"], "Gaudi17")
        self.assertEqual(metadata["phase_convention"], "orbital_transit_zero")
        self.assertEqual(metadata["phase_bin"], "all")
        self.assertIn("all exposures", metadata["phase_bin_definition"])

    def test_main_selects_post_eclipse_and_writes_sysrem_bundle(self):
        base = _load_result()
        extras = {
            "U_sysrem": np.ones((3, 1, 1), dtype=float),
        }
        chunk_indices = (
            np.asarray(["all"], dtype="U32"),
            (np.arange(4, dtype=int),),
            np.zeros(4, dtype=bool),
        )

        rc, _planet_mock = self._run_main(
            self._argv("--phase-bin", "post_eclipse", "--run-sysrem"),
            planet_cfg=_planet_cfg(),
            result=(base, extras),
            phase=np.asarray([-0.49, -0.01, 0.01], dtype=float),
            chunk_indices=chunk_indices,
        )

        self.assertEqual(rc, 0)
        saved_phase = np.load(self.output_dir / "phase.npy")
        np.testing.assert_allclose(saved_phase, np.asarray([-0.49, -0.01], dtype=float))
        self.assertTrue((self.output_dir / "U_sysrem.npz").exists())
        metadata = json.loads((self.output_dir / "timeseries_prep.json").read_text())
        self.assertEqual(metadata["phase_bin"], "post_eclipse")

    def test_main_requires_duration_for_eclipse_selection(self):
        with self.assertRaisesRegex(ValueError, "requires a finite duration"):
            self._run_main(
                self._argv("--phase-bin", "eclipse"),
                planet_cfg=_planet_cfg(duration=np.nan),
                result=(_load_result(), {}),
                phase=np.asarray([-0.49, -0.20, 0.10], dtype=float),
            )


if __name__ == "__main__":
    unittest.main()
