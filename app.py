"""Local Gradio application for the trained PeMS04 traffic model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

import numpy as np
import pandas as pd

from stid_inference import STIDPredictor
from traffic_data import load_csv, load_demo, load_report, observations_at, window_at


@dataclass
class Forecast:
    origin: pd.Timestamp
    history: pd.DataFrame
    clean: np.ndarray
    simulated: np.ndarray | None
    actual: np.ndarray
    fault: dict | None
    affected_ids: tuple[str, ...]
    horizons_minutes: tuple[int, ...]


class TrafficApp:
    def __init__(self, root):
        self.predictor = STIDPredictor(root)
        self.horizons_minutes = (15, 30)
        self._horizon_indices = [self.predictor.horizons_minutes.index(h)
                                 for h in self.horizons_minutes]
        self.sensor_ids = self.predictor.sensor_ids
        self.demo = load_demo(root)
        self.report = load_report(root)
        if tuple(self.demo.columns) != self.sensor_ids:
            raise ValueError("Thứ tự cảm biến PeMS04 không khớp với scaler đã lưu.")
        self.downloads = TemporaryDirectory(prefix="pems04-gradio-")
        self._template = Path(self.downloads.name) / "pems04-mau.csv"
        self.demo.iloc[:self.predictor.n_past].to_csv(self._template, encoding="utf-8-sig")

    def csv_template(self):
        return str(self._template)

    def load_csv(self, path):
        return load_csv(path, self.sensor_ids)

    def forecast(self, frame, origin=None, *, mode="clean", percent=5, minutes=15, center=None, seed=2026):
        # Only the built-in dataset is used for synthetic outages, even if a CSV matches it.
        if mode != "clean" and frame is not self.demo:
            raise ValueError("Chỉ dữ liệu mẫu PeMS04 mới hỗ trợ giả lập cảm biến hỏng.")
        if origin is None:
            if frame.empty:
                raise ValueError("Không có dữ liệu để dự báo.")
            origin = frame.index[-1]
        history = window_at(frame, origin).copy()
        origin = history.index[-1]
        clean = self.predictor.predict(history)[:, self._horizon_indices]
        simulated = None
        fault = None
        affected = ()
        if mode != "clean":
            fault = dict(mode=mode, percent=percent, minutes=minutes, center=center, seed=seed)
            simulated = self.predictor.predict(history, fault=fault)[:, self._horizon_indices]
            affected = self.predictor.affected_sensor_ids(fault)
        actual = observations_at(frame, origin, self.horizons_minutes).to_numpy()
        return Forecast(origin, history, clean, simulated, actual, fault, affected,
                        self.horizons_minutes)

    def _write_csv(self, frame, filename):
        # Each download keeps its own file, including across browser sessions.
        directory = Path(mkdtemp(dir=self.downloads.name))
        path = directory / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return str(path)

    def export_forecast(self, result):
        horizons = result.horizons_minutes
        horizon_column = np.tile(horizons, len(self.sensor_ids))
        table = pd.DataFrame({
            "sensor_id": np.repeat(self.sensor_ids, len(horizons)),
            "forecast_origin": result.origin,
            "horizon_minutes": horizon_column,
            "target_timestamp": result.origin + pd.to_timedelta(horizon_column, unit="min"),
            "forecast_clean": result.clean.ravel(),
            "observed": result.actual.ravel(),
        })
        if result.simulated is not None:
            table["forecast_simulated"] = result.simulated.ravel()
            table["difference"] = table["forecast_simulated"] - table["forecast_clean"]
            table["affected_sensor"] = table["sensor_id"].isin(result.affected_ids)
            table["scenario"] = result.fault["mode"]
            table["miss_pct"] = result.fault["percent"]
            table["miss_minutes"] = result.fault["minutes"]
            if result.fault["mode"] == "region":
                table["center"] = result.fault["center"]
            else:
                table["seed"] = result.fault["seed"]
        return self._write_csv(table, "du-bao-pems04.csv")

    def export_report(self, frame):
        return self._write_csv(frame, "danh-gia-stid.csv")


def main():
    import torch
    from traffic_ui import build_ui

    torch.set_num_threads(min(4, torch.get_num_threads()))
    app = TrafficApp(Path(__file__).resolve().parent)
    ui = build_ui(app)
    ui.queue().launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
