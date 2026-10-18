"""Regression coverage for the TILDE-Q implementation retained from FYP-TSFM."""

import unittest
from unittest.mock import patch

import numpy as np

from scripts.helpers.forecast_eval import compute_metrics
from taildart.utils.tilde_q import compute_tilde_q


class TildeQTests(unittest.TestCase):
    def test_matches_original_experiment_reference_values(self):
        # Reference values come from the original FYP-TSFM implementation.
        # Its identical-input loss need not be zero; preserve that behavior.
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        for prediction, expected in (
            (actual, 0.5),
            (actual + 0.1, 0.5333795547485352),
        ):
            with self.subTest(prediction=prediction):
                self.assertAlmostEqual(compute_tilde_q(actual, prediction), expected, places=6)
                metrics = compute_metrics(actual, prediction, history=np.arange(5.0))
                self.assertTrue(np.isfinite(metrics["tilde_q"]))
                self.assertAlmostEqual(metrics["tilde_q"], expected, places=6)

    def test_missing_metric_dependency_is_not_silently_suppressed(self):
        with patch(
            "scripts.helpers.forecast_eval._compute_tilde_q_metric",
            side_effect=ModuleNotFoundError("Missing TILDE-Q implementation"),
        ):
            with self.assertRaises(ModuleNotFoundError):
                compute_metrics(np.arange(1.0, 5.0), np.arange(1.0, 5.0))

    def test_undefined_zero_series_retains_nan_result(self):
        metrics = compute_metrics(np.zeros(4), np.zeros(4))
        self.assertTrue(np.isnan(metrics["tilde_q"]))


if __name__ == "__main__":
    unittest.main()
