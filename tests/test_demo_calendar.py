"""Check calendar alignment through the demo's rolling forecast pipeline."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from gluonts.model.forecast import SampleForecast

from scripts import helpers
from scripts.helpers import forecast_eval


ROOT = Path(__file__).resolve().parents[1]


class DemoCalendarTests(unittest.TestCase):
    def test_context_and_forecast_dates_match_source_observations(self):
        entries = []

        class RecordingPredictor:
            def predict(self, dataset, num_samples):
                for entry in dataset:
                    entries.append(entry)
                    yield SampleForecast(
                        samples=np.zeros((num_samples, 1), dtype=np.float32),
                        start_date=entry["start"] + len(entry["target"]),
                        item_id=entry["item_id"],
                    )

        # Execute the real demo, replacing model inference and the CSV write.
        # Alias its direct-script imports to the helpers used by this test.
        with (
            patch.dict(sys.modules, {
                "helpers": helpers,
                "helpers.forecast_eval": forecast_eval,
            }),
            patch.object(
                forecast_eval, "load_taildart_from_ckpt",
                return_value=(None, None, RecordingPredictor()),
            ),
            patch.object(pd.DataFrame, "to_csv"),
            redirect_stdout(io.StringIO()),
            warnings.catch_warnings(),
        ):
            demo = runpy.run_path(str(ROOT / "scripts/taildart_zero_shot_singledataset.py"))

        source = demo["synthetic_ts_df"]
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(source["date"]).rename(None),
            pd.date_range("2020-01-01", periods=len(source), freq="D"),
        )
        self.assertEqual(len(entries), len(demo["test_windows_df"]))
        self.assertGreater(len(entries), 0)
        for entry, forecast, origin in zip(
            entries, demo["forecasts"],
            range(demo["TRAIN_SIZE"], len(source)),
        ):
            with self.subTest(origin=origin):
                dates = pd.period_range(
                    start=entry["start"], periods=len(entry["target"]),
                    freq=entry["start"].freq,
                ).to_timestamp()
                expected = source.iloc[origin - demo["CONTEXT_LENGTH"]:origin]
                pd.testing.assert_index_equal(
                    dates, pd.DatetimeIndex(expected["date"]).rename(None),
                )
                np.testing.assert_array_equal(
                    entry["target"], expected["values"].to_numpy(dtype=np.float32),
                )
                self.assertEqual(
                    forecast.start_date.to_timestamp(), source["date"].iloc[origin],
                )


if __name__ == "__main__":
    unittest.main()
