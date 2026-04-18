"""Regression tests for the 'no concatenation' --arm full refactor.

Covers config path helpers and the per-arm loop in the prep scripts.
(Shared-dRV model-level tests live in tests/test_shared_drv_model.py so
they don't clash with the astropy/jax import shims here.)
"""

from __future__ import annotations

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


from config import paths_config

from dataio import prepare_retrieval_timeseries as prep


class GetDataDirFullArmTests(unittest.TestCase):
    def test_get_data_dir_rejects_full_arm(self):
        with self.assertRaises(ValueError) as cm:
            paths_config.get_data_dir(planet="KELT-20b", arm="full", epoch="20200101")
        self.assertIn("full", str(cm.exception))
        self.assertIn("get_full_arm_data_dirs", str(cm.exception))

    def test_get_full_arm_data_dirs_returns_red_and_blue(self):
        dirs = paths_config.get_full_arm_data_dirs(
            planet="KELT-20b", epoch="20200101", mode="transmission"
        )
        self.assertEqual(set(dirs.keys()), {"red", "blue"})
        self.assertTrue(str(dirs["red"]).endswith("20200101/red"))
        self.assertTrue(str(dirs["blue"]).endswith("20200101/blue"))

    def test_full_arm_members_constant(self):
        self.assertEqual(paths_config.FULL_ARM_MEMBERS, ("red", "blue"))


class NoLegacyFullArmSymbolsTests(unittest.TestCase):
    """The refactor must remove the concatenation helpers entirely."""

    def test_combine_full_arms_not_exported(self):
        from dataio import collapse_transmission_timeseries_to_1d as ctt

        self.assertFalse(hasattr(ctt, "combine_full_arms"))
        self.assertFalse(hasattr(ctt, "_align_exposures_by_jd"))


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


def _load_result():
    data = np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]], dtype=float
    )
    wave = np.asarray(
        [[5000.0, 5001.0, 5002.0, 5003.0], [5000.0, 5001.0, 5002.0, 5003.0]],
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


class PrepareRetrievalFullArmMainTests(unittest.TestCase):
    """--arm full should loop per-arm and write red/blue sibling dirs."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self._input_patch = patch.object(paths_config, "INPUT_DIR", self.root)
        self._input_patch.start()

    def tearDown(self):
        self._input_patch.stop()
        self.tempdir.cleanup()

    def _argv(self, *extra):
        return [
            "prepare_retrieval_timeseries.py",
            "--epoch",
            "20200101",
            "--planet",
            "KELT-20b",
            "--arm",
            "full",
            "--phase-bin",
            "all",
            "--ephemeris",
            "Singh24",
            *extra,
        ]

    def test_full_arm_writes_red_and_blue_siblings(self):
        base_result = (_load_result(), {})
        phase = np.asarray([-0.01, 0.01], dtype=float)

        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", self._argv()))
            stack.enter_context(
                patch(
                    "dataio.prepare_retrieval_timeseries._planet_config",
                    return_value=_planet_cfg(),
                )
            )
            stack.enter_context(
                patch(
                    "dataio.prepare_retrieval_timeseries._load_data",
                    return_value=base_result,
                )
            )
            stack.enter_context(
                patch(
                    "dataio.prepare_retrieval_timeseries.get_orbital_phase",
                    return_value=phase,
                )
            )
            stack.enter_context(
                patch(
                    "dataio.prepare_retrieval_timeseries.remove_doppler_shadow",
                    return_value=(
                        base_result[0][1] - 0.1,
                        np.full_like(base_result[0][1], 0.1),
                        {"scaling": 0.8},
                    ),
                )
            )
            rc = prep.main()
        self.assertEqual(rc, 0)

        red_dir = self.root / "hrs/transmission/kelt20b/20200101/red"
        blue_dir = self.root / "hrs/transmission/kelt20b/20200101/blue"
        for arm_dir in (red_dir, blue_dir):
            self.assertTrue(arm_dir.exists(), f"missing {arm_dir}")
            self.assertTrue((arm_dir / "data.npy").exists())
            self.assertTrue((arm_dir / "wavelength.npy").exists())
            meta = json.loads((arm_dir / "timeseries_prep.json").read_text())
            self.assertEqual(meta["ephemeris"], "Singh24")

        # No legacy 'full' sibling directory should be created.
        full_dir = self.root / "hrs/transmission/kelt20b/20200101/full"
        self.assertFalse(full_dir.exists())

    def test_full_arm_rejects_output_dir(self):
        with patch.object(
            sys,
            "argv",
            self._argv("--output-dir", str(self.root / "prep")),
        ):
            with self.assertRaises(ValueError) as cm:
                prep.main()
            self.assertIn("--output-dir", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
