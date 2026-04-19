from __future__ import annotations

import importlib
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
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
    return all(_real_package(name) for name in ("jax", "numpyro"))


class SVIProgressLoggingTests(unittest.TestCase):
    def setUp(self):
        if not _deps_are_real():
            self.skipTest(
                "real jax/numpyro packages are not available in sys.modules; "
                "run tests/test_inference_progress.py in the retrieval environment."
            )

    def test_run_svi_logs_periodic_progress_and_resumes_from_previous_state(self):
        import pipeline.inference as inference

        run_calls: list[dict[str, object]] = []
        saved_outputs: dict[str, object] = {}
        fake_guide = [
            object(),
            SimpleNamespace(median=lambda params: {"latent": np.asarray(params["theta"])}),
        ]

        class FakeSVI:
            def __init__(self, model, guide, optimizer, loss):
                self.model = model
                self.guide = guide
                self.optimizer = optimizer
                self.loss = loss

            def run(self, rng_key, num_steps, *args, progress_bar=True, init_state=None, **kwargs):
                del rng_key, args, kwargs
                run_calls.append(
                    {
                        "num_steps": int(num_steps),
                        "progress_bar": bool(progress_bar),
                        "init_state": init_state,
                    }
                )
                start_step = 0 if init_state is None else int(init_state["step"])
                end_step = start_step + int(num_steps)
                losses = np.arange(start_step + 1, end_step + 1, dtype=float)
                return SimpleNamespace(
                    params={"theta": np.asarray([end_step], dtype=float)},
                    state={"step": end_step},
                    losses=losses,
                )

        def fake_save_svi_outputs(params, losses, init_values, output_dir):
            saved_outputs["params"] = params
            saved_outputs["losses"] = np.asarray(losses, dtype=float)
            saved_outputs["init_values"] = init_values
            saved_outputs["output_dir"] = output_dir

        with tempfile.TemporaryDirectory() as tempdir:
            stdout = io.StringIO()
            with (
                patch.object(inference, "build_guide", return_value=fake_guide),
                patch.object(inference, "build_svi_optimizer", return_value=object()),
                patch.object(inference, "SVI", FakeSVI),
                patch.object(inference, "_default_svi_report_interval", return_value=4),
                patch.object(inference, "save_svi_outputs", side_effect=fake_save_svi_outputs),
                redirect_stdout(stdout),
            ):
                params, losses, _init_strategy, svi_median, guide = inference.run_svi(
                    model_c=lambda **_kwargs: None,
                    rng_key=np.asarray([0, 1], dtype=np.uint32),
                    model_inputs={"observations": {"dummy": object()}},
                    Mp_mean=2.0,
                    Mp_std=0.1,
                    Mp_upper_3sigma=9.0,
                    Rp_mean=1.5,
                    Rp_std=0.2,
                    Rstar_mean=1.7,
                    Rstar_std=0.3,
                    output_dir=tempdir,
                    num_steps=11,
                    lr=0.001,
                )

        output = stdout.getvalue()

        self.assertEqual([call["num_steps"] for call in run_calls], [1, 4, 4, 2])
        self.assertTrue(all(call["progress_bar"] is False for call in run_calls))
        self.assertIsNone(run_calls[0]["init_state"])
        self.assertEqual(run_calls[1]["init_state"], {"step": 1})
        self.assertEqual(run_calls[2]["init_state"], {"step": 5})
        self.assertEqual(run_calls[3]["init_state"], {"step": 9})

        np.testing.assert_allclose(np.asarray(losses, dtype=float), np.arange(1.0, 12.0))
        np.testing.assert_allclose(saved_outputs["losses"], np.arange(1.0, 12.0))
        np.testing.assert_allclose(params["theta"], np.asarray([11.0]))
        np.testing.assert_allclose(saved_outputs["params"]["theta"], np.asarray([11.0]))

        self.assertEqual(saved_outputs["output_dir"], tempdir)
        self.assertEqual(guide, fake_guide)
        np.testing.assert_allclose(svi_median["latent"], np.asarray([11.0]))
        self.assertEqual(svi_median["Mp"], 3.0)
        self.assertEqual(svi_median["Rp"], 1.5)
        self.assertEqual(svi_median["Rstar"], 1.7)

        self.assertIn("SVI: JAX compile starting; first update may take a while...", output)
        self.assertIn("SVI progress: 1/11 steps", output)
        self.assertIn("SVI progress: 5/11 steps", output)
        self.assertIn("SVI progress: 9/11 steps", output)
        self.assertIn("SVI progress: 11/11 steps", output)
        self.assertIn("SVI warm-up complete:", output)


if __name__ == "__main__":
    unittest.main()
