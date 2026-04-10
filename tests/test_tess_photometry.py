import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


sys.modules.setdefault("astropy", types.ModuleType("astropy"))
sys.modules.setdefault("astropy.io", types.ModuleType("astropy.io"))
sys.modules.setdefault("astropy.io.fits", types.ModuleType("astropy.io.fits"))
sys.modules.setdefault("exojax", types.ModuleType("exojax"))
sys.modules.setdefault("exojax.utils", types.ModuleType("exojax.utils"))
sys.modules.setdefault(
    "exojax.utils.grids",
    types.SimpleNamespace(wav2nu=lambda wavelength, unit: np.asarray(wavelength)),
)

from dataio.load import load_nasa_archive_spectrum, parse_nasa_archive_tbl
from dataio.tess_photometry import (
    TessTransitFitConfig,
    TessTransitFitResult,
    download_tess_lightcurves,
    make_tess_bandpass_constraint_from_mlexo,
    serialize_tess_fit_summary,
    summarize_posterior_samples,
    write_tess_bandpass_tbl,
)


def _theta_sample(
    *,
    period_d: float = 3.0,
    t0_btjd: float = 100.0,
    rho_star_solar: float = 0.4,
    radius_ratio: float = 0.1,
    impact_param: float = 0.2,
    q1: float = 0.25,
    q2: float = 0.30,
    n_sectors: int = 1,
) -> np.ndarray:
    impact_scaled = impact_param / (1.0 + radius_ratio)
    return np.concatenate(
        [
            np.array(
                [
                    np.log(period_d),
                    t0_btjd,
                    np.log(rho_star_solar),
                    np.log(radius_ratio),
                    impact_scaled,
                    q1,
                    q2,
                ],
                dtype=float,
            ),
            np.zeros(n_sectors, dtype=float),
            np.full(n_sectors, np.log(1.0e-4), dtype=float),
            np.full(n_sectors, np.log(1.0e-4), dtype=float),
            np.full(n_sectors, np.log(0.05), dtype=float),
        ]
    )


class TessPhotometryTests(unittest.TestCase):
    def test_download_tess_lightcurves_passes_quality_bitmask_and_flux_column(self):
        class _FakeSearchResult:
            def __init__(self):
                self.download_all_kwargs = None

            def __len__(self):
                return 1

            def download_all(self, **kwargs):
                self.download_all_kwargs = kwargs
                return ["lc1", "lc2"]

        class _FakeLightkurve:
            def __init__(self):
                self.search_calls = []
                self.search_result = _FakeSearchResult()

            def search_lightcurve(self, target, **kwargs):
                self.search_calls.append((target, kwargs))
                return self.search_result

        fake_lk = _FakeLightkurve()
        config = TessTransitFitConfig(
            target="TIC 123456789",
            period_d=3.0,
            t0_btjd=100.0,
            transit_duration_d=0.12,
            radius_ratio_guess=0.1,
            impact_guess=0.2,
            quality_bitmask="hard",
            flux_column="sap_flux",
            sectors=(14, 15),
        )

        with patch("dataio.tess_photometry._load_module_or_raise", return_value=fake_lk):
            collection = download_tess_lightcurves(config)

        self.assertEqual(collection, ["lc1", "lc2"])
        self.assertEqual(len(fake_lk.search_calls), 1)
        target, search_kwargs = fake_lk.search_calls[0]
        self.assertEqual(target, "TIC 123456789")
        self.assertEqual(search_kwargs["sector"], [14, 15])
        self.assertEqual(search_kwargs["author"], "SPOC")
        self.assertEqual(fake_lk.search_result.download_all_kwargs["quality_bitmask"], "hard")
        self.assertEqual(fake_lk.search_result.download_all_kwargs["flux_column"], "sap_flux")

    def test_summarize_posterior_samples_is_self_contained(self):
        theta = _theta_sample()
        samples = np.vstack([theta, theta, theta])

        summary = summarize_posterior_samples(samples, n_sectors=1)

        self.assertIsNotNone(summary)
        self.assertAlmostEqual(summary["r"]["median"], 0.1)
        self.assertAlmostEqual(summary["transit_depth_percent"]["median"], 1.0)
        self.assertGreater(summary["a_over_rstar"]["median"], 0.0)
        self.assertGreater(summary["inclination_deg"]["median"], 0.0)

    def test_make_radius_ratio_constraint_from_summary(self):
        summary = {
            "r": {
                "median": 0.104,
                "plus": 0.003,
                "minus": 0.002,
            }
        }

        constraint = make_tess_bandpass_constraint_from_mlexo(
            summary_stats=summary,
            observable="radius_ratio",
            photon_weighted=True,
        )

        self.assertEqual(constraint["mode"], "transmission")
        self.assertEqual(constraint["observable"], "radius_ratio")
        self.assertAlmostEqual(constraint["value"], 0.104)
        self.assertAlmostEqual(constraint["sigma"], 0.003)
        self.assertTrue(constraint["photon_weighted"])

    def test_make_transit_depth_constraint_uses_fractional_depth(self):
        summary = {
            "transit_depth_percent": {
                "median": 1.25,
                "plus": 0.10,
                "minus": 0.05,
            }
        }

        constraint = make_tess_bandpass_constraint_from_mlexo(
            summary_stats=summary,
            observable="transit_depth",
            sigma_mode="mean",
        )

        self.assertEqual(constraint["observable"], "transit_depth")
        self.assertAlmostEqual(constraint["value"], 0.0125)
        self.assertAlmostEqual(constraint["sigma"], 0.00075)

    def test_write_tess_bandpass_tbl_round_trips_through_loader(self):
        constraint = {
            "name": "tess_transit",
            "mode": "transmission",
            "observable": "radius_ratio",
            "value": 0.1,
            "sigma": 0.002,
        }

        with tempfile.TemporaryDirectory() as tempdir:
            tbl_path = write_tess_bandpass_tbl(
                Path(tempdir) / "tess.tbl",
                constraint=constraint,
                planet_name="KELT-20 b",
                reference="unit test",
                note="synthetic",
            )

            metadata, columns, _data_by_col, _units_by_col = parse_nasa_archive_tbl(tbl_path)
            wav_angstrom, spectrum, sigma, loader_metadata = load_nasa_archive_spectrum(
                tbl_path,
                mode="transmission",
            )

        self.assertEqual(metadata["PL_NAME"], "KELT-20 b")
        self.assertEqual(loader_metadata["SPEC_TYPE"], "Transit")
        self.assertIn("SPECTRANSDEP", columns)
        np.testing.assert_allclose(wav_angstrom, np.array([8000.0]))
        np.testing.assert_allclose(spectrum, np.array([0.01]))
        np.testing.assert_allclose(sigma, np.array([0.0004]))

    def test_serialize_tess_fit_summary_returns_json_safe_mapping(self):
        config = TessTransitFitConfig(
            target="KELT-20",
            period_d=3.0,
            t0_btjd=100.0,
            transit_duration_d=0.12,
            radius_ratio_guess=0.1,
            impact_guess=0.2,
        )
        result = TessTransitFitResult(
            config=config,
            dataset={
                "n_sectors": 2,
                "sector_labels": np.array(["TESS Sector 14", "TESS Sector 15"]),
                "sector_counts": np.array([120, 140]),
            },
            best_fit={
                "period": 3.0,
                "t0": 100.0,
                "rho_star_solar": 0.4,
                "a_over_rstar": 7.5,
                "duration": 0.12,
                "r": 0.1,
                "b": 0.2,
                "emcee_acceptance_fraction": 0.35,
                "emcee_parallel": False,
                "emcee_worker_count": 1,
                "summary_stats": {"r": {"median": 0.1, "plus": 0.002, "minus": 0.002}},
            },
            bandpass_constraint={
                "name": "tess_transit",
                "mode": "transmission",
                "observable": "radius_ratio",
                "value": 0.1,
                "sigma": 0.002,
            },
        )

        summary = serialize_tess_fit_summary(result)

        self.assertEqual(summary["target"], "KELT-20")
        self.assertEqual(summary["n_sectors"], 2)
        self.assertEqual(summary["sector_counts"], [120, 140])
        self.assertEqual(summary["constraint"]["observable"], "radius_ratio")
        self.assertAlmostEqual(summary["fit"]["r"], 0.1)


if __name__ == "__main__":
    unittest.main()
