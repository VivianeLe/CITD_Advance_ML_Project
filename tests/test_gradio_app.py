"""Regression checks for the inference and CSV paths used by Gradio."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import torch

from app import TrafficApp


ROOT = Path(__file__).resolve().parents[1]


class TrafficAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.app = TrafficApp(ROOT)
        cls.addClassCleanup(cls.app.downloads.cleanup)

    def test_demo_matches_notebook(self):
        result = self.app.forecast(self.app.demo, "2018-02-17 08:00")
        # Clean CPU inference in build_model - Nam Anh.ipynb, same test window.
        reference = [[267.76575, 295.13358, 312.47266],
                     [99.22418, 108.89818, 121.70652],
                     [190.36749, 204.35625, 213.57994]]
        np.testing.assert_allclose(result.clean[[0, 200, 306]], reference, atol=0.001, rtol=0)
        self.assertEqual(result.clean.shape, (307, 3))
        self.assertEqual(len(self.app.report), 12)

    def test_template_without_future_and_reordered_columns(self):
        template = self.app.load_csv(self.app.csv_template())
        self.assertEqual(len(template), 24)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "traffic.csv"
            template.iloc[:, ::-1].to_csv(path)
            frame = self.app.load_csv(path)
            pd.testing.assert_frame_equal(frame, template)
            frame.drop(columns=["8"]).to_csv(path)
            with self.assertRaisesRegex(ValueError, "Thiếu ID.*8"):
                self.app.load_csv(path)
        result = self.app.forecast(template, template.index[-1])
        self.assertTrue(np.isnan(result.actual).all())
        self.assertTrue(np.isfinite(result.clean).all())

    def test_simulation_and_export(self):
        origin = "2018-02-17 08:00"
        settings = dict(mode="random", percent=20, minutes=30)
        first = self.app.forecast(self.app.demo, origin, **settings)
        again = self.app.forecast(self.app.demo, origin, **settings)
        np.testing.assert_array_equal(first.simulated, again.simulated)
        self.assertEqual(len(first.affected_ids), 61)
        self.assertFalse(np.array_equal(first.clean, first.simulated))
        region = self.app.forecast(self.app.demo, origin, mode="region", percent=10, minutes=30, center="0")
        self.assertEqual(len(region.affected_ids), 31)
        self.assertIn("0", region.affected_ids)
        exported = pd.read_csv(self.app.export_forecast(region))
        self.assertEqual(len(exported), 307 * 3)
        self.assertEqual(exported.sensor_id.nunique(), 307)
        self.assertEqual(set(exported.horizon_minutes), {15, 30, 60})

    def test_uploaded_csv_preserves_measurements_and_rejects_faults(self):
        history = self.app.demo.iloc[:24].copy()
        history.iloc[-1, 0] = np.nan
        history.iloc[-1, 1] = 0
        with TemporaryDirectory() as directory:
            path = Path(directory) / "traffic.csv"
            history.to_csv(path)
            uploaded = self.app.load_csv(path)
        origin = history.index[-1]
        result = self.app.forecast(uploaded, origin)
        np.testing.assert_array_equal(result.history.to_numpy(), history.to_numpy())
        np.testing.assert_array_equal(result.clean, self.app.predictor.predict(history))
        self.assertIsNone(result.simulated)
        exported = pd.read_csv(self.app.export_forecast(result))
        self.assertEqual(len(exported), 307 * 3)
        self.assertEqual(set(exported.columns), {
            "sensor_id", "forecast_origin", "horizon_minutes", "target_timestamp",
            "forecast_clean", "observed",
        })
        for mode in ("random", "region"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "Chỉ dữ liệu mẫu PeMS04"):
                self.app.forecast(uploaded, origin, mode=mode)


if __name__ == "__main__":
    unittest.main()
