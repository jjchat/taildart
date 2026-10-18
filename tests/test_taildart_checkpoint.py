"""Exercise the renamed package, config, and bundled checkpoint together.

Run from the repository root with ``python -m unittest discover -s tests``.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from gluonts.dataset.common import ListDataset

from scripts.helpers.forecast_eval import load_taildart_from_ckpt
from taildart.model.distributions import JohnsonSUOutput
from taildart.model.taildart_module import TailDartModel
from taildart.training import load_model_config


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "final_checkpoints/taildart_dense_base_johnsonsu.ckpt"
CONFIG = ROOT / "configs/taildart_dense_johnsonsu.json"


class TailDartCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        cls.config = load_model_config(CONFIG, model_size="base")
        cls.estimator, cls.module, cls.predictor = load_taildart_from_ckpt(
            CHECKPOINT,
            prediction_length=1,
            context_length=8,
            device="cpu",
            batch_size=1,
            num_parallel_samples=4,
            model_config=cls.config,
        )
        cls.module.eval()

    def test_checkpoint_loads_with_canonical_package_and_config(self):
        self.assertFalse((ROOT / "lag_gemma").exists())
        output = self.checkpoint["metadata"]["model_kwargs"]["distr_output"]
        self.assertIsInstance(output, JohnsonSUOutput)
        self.assertEqual(type(output).__module__, "taildart.model.distributions")
        saved_kwargs = self.checkpoint["metadata"]["model_kwargs"]
        for key in self.config:
            if key.startswith("taildart_") or key in ("n_layer", "ffn_type", "scaling"):
                self.assertEqual(saved_kwargs[key], self.config[key], key)
        # create_module above strictly loads the weights; compare values as well.
        loaded = self.module.model.state_dict()
        saved = self.checkpoint["model_state_dict"]
        self.assertEqual(set(loaded), set(saved))
        for key, tensor in saved.items():
            self.assertTrue(torch.equal(loaded[key], tensor), key)

    def test_cpu_prediction_is_finite_and_reproducible(self):
        dataset = ListDataset(
            [{"start": "2024-01-01", "target": np.linspace(1, 3, 64, dtype=np.float32)}],
            freq="D",
        )
        torch.manual_seed(17)
        first = list(self.predictor.predict(dataset))[0]
        torch.manual_seed(17)
        second = list(self.predictor.predict(dataset))[0]
        self.assertEqual(first.samples.shape, (4, 1))
        self.assertTrue(np.isfinite(first.samples).all())
        np.testing.assert_array_equal(first.samples, second.samples)

    def test_checkpoint_roundtrip_preserves_forward_output(self):
        model = self.module.model
        past_length = model.context_length + max(model.lags_seq)
        past = torch.linspace(1, 3, past_length).unsqueeze(0)
        inputs = {
            "past_target": past,
            "past_observed_values": torch.ones_like(past),
            "past_time_feat": torch.zeros(1, past_length, 6),
            "future_time_feat": torch.zeros(1, 1, 6),
        }
        model.reset_cache()
        with torch.no_grad():
            before = model(**inputs)
        payload = {"model_kwargs": self.module.model_kwargs, "state_dict": model.state_dict()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "taildart.ckpt"
            torch.save(payload, path)
            restored = torch.load(path, map_location="cpu", weights_only=False)
        reloaded = TailDartModel(**restored["model_kwargs"]).eval()
        reloaded.load_state_dict(restored["state_dict"], strict=True)
        with torch.no_grad():
            after = reloaded(**inputs)
        self.assertEqual(len(before[0]), 4)
        for left, right in zip((*before[0], *before[1:]), (*after[0], *after[1:])):
            self.assertTrue(torch.isfinite(left).all())
            torch.testing.assert_close(left, right, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
