"""Regression checks for forecast controls and separate benchmark horizons."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import gradio as gr
import numpy as np
import pandas as pd
import torch

from app import TrafficApp
from traffic_ui import build_ui


ROOT = Path(__file__).resolve().parents[1]


class TrafficUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.service = TrafficApp(ROOT)
        cls.addClassCleanup(cls.service.downloads.cleanup)
        cls.ui = build_ui(cls.service)
        cls.callbacks = {function.api_name: function for function in cls.ui.fns.values()}

    def component(self, kind, *, label=None, value=None):
        return next(
            component for component in self.ui.blocks.values()
            if isinstance(component, kind)
            and (label is None or getattr(component, "label", None) == label)
            and (value is None or getattr(component, "value", None) == value)
        )

    def test_forecast_has_no_time_control_or_api_argument(self):
        labels = [getattr(component, "label", None) for component in self.ui.blocks.values()]
        self.assertNotIn("Thời điểm dự báo", labels)
        self.assertNotIn("select_time", self.callbacks)
        self.assertEqual(len(self.callbacks["run_forecast"].inputs), 3)
        self.assertEqual(len(self.callbacks["run_simulation"].inputs), 6)

    def test_forecast_and_simulation_show_two_horizons(self):
        for api_name, arguments in (
            ("run_forecast", ("demo", None, "0")),
            ("run_simulation", ("demo", None, "0", "random", 20, 15)),
        ):
            with self.subTest(api_name=api_name):
                callback = self.callbacks[api_name]
                updates = callback.fn(*arguments)
                state, cards, plot, comparison, download = callback.outputs
                result = updates[state].result
                self.assertEqual(result.origin, self.service.demo.index[-1])
                self.assertEqual(result.horizons_minutes, (15, 30))
                self.assertIn("Sau 15 phút", updates[cards]["value"])
                self.assertIn("Sau 30 phút", updates[cards]["value"])
                self.assertNotIn("Sau 60 phút", updates[cards]["value"])
                forecast_line = next(
                    line for line in updates[plot]["value"].axes[0].lines
                    if line.get_label() == "Dự báo"
                )
                self.assertEqual(len(forecast_line.get_xdata()), 2)
                exported = pd.read_csv(updates[download]["value"])
                self.assertEqual(len(exported), 307 * 2)
                self.assertEqual(set(exported.horizon_minutes), {15, 30})
                if api_name == "run_simulation":
                    table = updates[comparison]["value"]
                    self.assertEqual(table["Mốc"].tolist(), ["+15 phút", "+30 phút"])

    def test_source_switching_clears_results_and_validates_csv(self):
        callback = self.callbacks["select_data"]
        frame_state = callback.outputs[0]
        run = self.component(gr.Button, value="Dự báo")
        simulate = self.component(gr.Button, value="Chạy mô phỏng")
        status = self.component(gr.Textbox, label="Kiểm tra CSV")
        forecast_state = self.callbacks["run_forecast"].outputs[0]

        empty = callback.fn("csv", None)
        self.assertIsNone(empty[frame_state])
        self.assertIsNone(empty[forecast_state])
        self.assertFalse(empty[run]["interactive"])
        self.assertFalse(empty[simulate]["interactive"])

        valid = callback.fn("csv", self.service.csv_template())
        self.assertEqual(len(valid[frame_state]), 24)
        self.assertTrue(valid[run]["interactive"])
        self.assertFalse(valid[simulate]["interactive"])

        forecast = self.callbacks["run_forecast"].fn("csv", valid[frame_state], "0")
        self.assertEqual(forecast[forecast_state].result.origin, valid[frame_state].index[-1])

        with TemporaryDirectory() as directory:
            invalid = callback.fn("csv", str(Path(directory) / "missing.csv"))
        self.assertIsNone(invalid[frame_state])
        self.assertIsNone(invalid[forecast_state])
        self.assertFalse(invalid[run]["interactive"])
        self.assertIn("CSV không hợp lệ", invalid[status]["value"])

        demo = callback.fn("demo", self.service.csv_template())
        self.assertIsNone(demo[frame_state])
        self.assertIsNone(demo[forecast_state])
        self.assertTrue(demo[run]["interactive"])
        self.assertTrue(demo[simulate]["interactive"])

    def test_benchmark_retains_saved_three_horizon_report(self):
        horizon_filter = self.component(gr.Dropdown, label="Mốc dự báo")
        self.assertEqual([value for _, value in horizon_filter.choices], ["avg", "15", "30", "60"])
        callback = self.callbacks["filter_report"]
        shown, path = callback.fn("all", "60")
        np.testing.assert_array_equal(shown["MAE"], self.service.report["MAE_60"])
        exported = pd.read_csv(path)
        pd.testing.assert_frame_equal(exported, self.service.report)


if __name__ == "__main__":
    unittest.main()
