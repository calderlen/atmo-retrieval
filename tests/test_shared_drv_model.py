"""Shared-dRV plumbing tests in physics.model.

These require real JAX / numpyro, so we deliberately do NOT install the
dummy jax shim that lives in sister prep-script tests. Run this file on
its own (or alongside prep tests that only touch the shim via
``setdefault`` after this module's imports have resolved).
"""

from __future__ import annotations

import sys
import unittest

import numpy as np


def _jax_is_real() -> bool:
    """Return True iff ``jax`` in sys.modules is a real package (not a dummy
    shim installed by a sister test file)."""
    import importlib

    try:
        real = importlib.import_module("jax")
    except Exception:
        return False
    return hasattr(real, "__path__")


class SharedDRVModelTests(unittest.TestCase):
    def setUp(self):
        # Skip when another test file has replaced jax/astropy/exojax with
        # dummy shims in sys.modules (we can't safely evict them without
        # breaking sibling tests that rely on those shims). Run this file on
        # its own (``python -m unittest tests.test_shared_drv_model``) to
        # exercise the shared-dRV plumbing.
        if not _jax_is_real():
            self.skipTest(
                "jax dummy shim is installed in sys.modules; "
                "run tests/test_shared_drv_model.py in isolation "
                "(python -m unittest tests.test_shared_drv_model)."
            )

    def _run_once(self, *, shared: bool):
        import jax.numpy as jnp
        import numpyro
        from numpyro.handlers import seed, trace

        from physics.model import (
            SpectroscopicObservationConfig,
            _sample_component_velocity_offset,
            _sample_shared_system_state,
            build_shared_system_config,
        )

        params = dict(
            Kp=180.0,
            Kp_err=5.0,
            RV_abs=-24.0,
            RV_abs_err=0.1,
            R_p=1.8,
            R_p_err=0.1,
            M_p=3.5,
            M_p_err=0.2,
            R_star=1.6,
            R_star_err=0.05,
            period=3.47,
        )
        if shared:
            shared_cfg = build_shared_system_config(
                params=params,
                shared_velocity_phase_mode="linear",
                shared_velocity_component_names=(
                    "spectroscopy_red",
                    "spectroscopy_blue",
                ),
            )
        else:
            shared_cfg = build_shared_system_config(params=params)

        phase_red = jnp.linspace(-0.02, 0.02, 6)
        phase_blue = jnp.linspace(-0.02, 0.02, 4)

        def _cfg(name):
            return SpectroscopicObservationConfig(
                name=name,
                region_name="region",
                mode="transmission",
                opa_mols={},
                opa_atoms={},
                opa_cias={},
                nu_grid=jnp.asarray([1.0]),
                sop_rot=None,
                sop_inst=None,
                inst_nus=jnp.asarray([1.0]),
                beta_inst=1.0,
                radial_velocity_mode="dRV",
                phase_mode="linear",
                likelihood_kind="gaussian",
                subtract_per_exposure_mean=False,
                apply_sysrem=False,
                Tstar=None,
                sample_prefix=name,
            )

        red_cfg = _cfg("spectroscopy_red")
        blue_cfg = _cfg("spectroscopy_blue")

        def model():
            state = _sample_shared_system_state(shared_cfg)
            dRV_red = _sample_component_velocity_offset(
                red_cfg, phase_red, scope_prefix="spectroscopy_red", shared_state=state
            )
            dRV_blue = _sample_component_velocity_offset(
                blue_cfg, phase_blue, scope_prefix="spectroscopy_blue", shared_state=state
            )
            numpyro.deterministic("dRV_red_out", dRV_red)
            numpyro.deterministic("dRV_blue_out", dRV_blue)

        with seed(rng_seed=0):
            tr = trace(model).get_trace()
        return tr

    def test_shared_drv_samples_only_once(self):
        tr = self._run_once(shared=True)
        sample_sites = {
            k
            for k, v in tr.items()
            if v.get("type") == "sample" and not v.get("is_observed")
        }
        self.assertIn("shared_dRV_0", sample_sites)
        self.assertIn("shared_dRV_slope", sample_sites)
        self.assertNotIn("spectroscopy_red/dRV_0", sample_sites)
        self.assertNotIn("spectroscopy_blue/dRV_0", sample_sites)

        dRV_red = np.asarray(tr["dRV_red_out"]["value"])
        dRV_blue = np.asarray(tr["dRV_blue_out"]["value"])
        self.assertTrue(
            np.isclose(dRV_red[0], dRV_blue[0]),
            msg="first-exposure dRV should match between arms (same phase)",
        )
        self.assertTrue(
            np.isclose(dRV_red[-1], dRV_blue[-1]),
            msg="last-exposure dRV should match between arms (same phase)",
        )

    def test_unshared_drv_samples_per_component(self):
        tr = self._run_once(shared=False)
        sample_sites = {
            k
            for k, v in tr.items()
            if v.get("type") == "sample" and not v.get("is_observed")
        }
        self.assertNotIn("shared_dRV_0", sample_sites)
        self.assertNotIn("shared_dRV_slope", sample_sites)
        self.assertIn("spectroscopy_red/dRV_0", sample_sites)
        self.assertIn("spectroscopy_red/dRV_slope", sample_sites)
        self.assertIn("spectroscopy_blue/dRV_0", sample_sites)
        self.assertIn("spectroscopy_blue/dRV_slope", sample_sites)


if __name__ == "__main__":
    unittest.main()
