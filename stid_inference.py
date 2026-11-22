"""CPU inference for the PeMS04 STID checkpoint shipped with this project."""
from __future__ import annotations

import math
import pickle
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import shortest_path
from torch import nn
from torch.nn import functional as F


class _Context(nn.Module):
    """Keep the parameter names and input layout of the trained STID model."""

    def __init__(self, nodes, channels, options):
        super().__init__()
        embedding = options["node_emb"]
        self.past = nn.Linear(options["n_past"] * channels, options["hidden"])
        self.node = nn.Embedding(nodes, embedding)
        self.tod = nn.Embedding(options["tod_slots"], embedding)
        self.weekend = nn.Embedding(options["weekend_slots"], embedding)

    def forward(self, x):
        batch, steps, nodes, channels = x.shape
        history = x.permute(0, 2, 1, 3).reshape(batch, nodes, steps * channels)
        past = F.gelu(self.past(history))
        angle = torch.remainder(torch.atan2(x[:, -1, 0, 2], x[:, -1, 0, 3]), 2 * math.pi)
        slots = self.tod.num_embeddings
        slot = torch.round(angle * (slots / (2 * math.pi))).long().remainder(slots)
        weekend = x[:, -1, 0, 4].round().long().clamp(0, 1)
        return torch.cat((
            past, self.node.weight.unsqueeze(0).expand(batch, -1, -1),
            self.tod(slot).unsqueeze(1).expand(-1, nodes, -1),
            self.weekend(weekend).unsqueeze(1).expand(-1, nodes, -1),
        ), dim=-1)


class _ResidualMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.first = nn.Linear(width, 2 * width)
        self.second = nn.Linear(2 * width, width)

    def forward(self, x):
        # Dropout is inactive during inference.
        return x + self.second(F.gelu(self.first(self.norm(x))))


class _STID(nn.Module):
    def __init__(self, bundle):
        super().__init__()
        options = bundle["model_options"]
        width = options["hidden"] + 3 * options["node_emb"]
        self.register_buffer("A", bundle["state_dict"]["A"])
        self.context = _Context(bundle["n_nodes"], bundle["n_in"], options)
        self.blocks = nn.Sequential(*[_ResidualMLP(width) for _ in range(options["depth"])])
        self.head = nn.Linear(width, len(bundle["horizons"]))
        self.residual = options["residual"]

    def forward(self, x):
        prediction = self.head(self.blocks(self.context(x)))
        if self.residual:
            prediction = prediction + x[:, -1, :, 0:1]
        return prediction


class STIDPredictor:
    """Predict all sensors from one history window, without future observations."""

    def __init__(self, root):
        self.root = Path(root)
        outputs = self.root / "outputs"
        # Only the repository checkpoint is opened here. Uploads are data CSVs.
        with (outputs / "stid_standard_tuned.pkl").open("rb") as handle:
            bundle = pickle.load(handle)
        if (bundle["model_name"] != "STID" or bundle["n_in"] != 6
                or not bundle["model_options"]["use_time"] or not bundle["cfg"]["USE_ADJ_FEAT"]):
            raise ValueError("Checkpoint phải là STID dùng thời gian và đặc trưng láng giềng.")
        nodes = bundle["n_nodes"]
        self.n_past = bundle["n_past"]
        self.step_minutes = bundle["cfg"]["STEP_MIN"]
        self.horizons_minutes = tuple(step * self.step_minutes for step in bundle["horizons"])
        with np.load(outputs / "stid_standard_tuned_scaler.npz", allow_pickle=False) as scaler:
            if (scaler["schema_version"].item() != 1 or scaler["name"].item() != "standard"
                    or bundle["scaler"]["name"] != "standard"):
                raise ValueError("Model và file scaler phải dùng StandardScaler phiên bản 1.")
            self.sensor_ids = tuple(scaler["sensor_ids"].astype(str).tolist())
            mean, std = scaler["mu"], scaler["sd"]
        if (self.sensor_ids != tuple(map(str, range(nodes))) or mean.shape != (nodes,)
                or std.shape != (nodes,) or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or not (std > 0).all()
                or not np.array_equal(mean, bundle["scaler"]["mu"])
                or not np.array_equal(std, bundle["scaler"]["sd"])):
            raise ValueError("Tham số hoặc thứ tự cảm biến của scaler không khớp model.")
        self._mean = torch.as_tensor(mean, dtype=torch.float32)
        self._std = torch.as_tensor(std, dtype=torch.float32)
        self._fill = torch.as_tensor(bundle["fill"], dtype=torch.float32)
        self._neighbors = torch.as_tensor(bundle["a_nb"], dtype=torch.float32)
        if self._fill.shape != (nodes,) or self._neighbors.shape != (nodes, nodes):
            raise ValueError("Kích thước imputer hoặc ma trận láng giềng không khớp model.")
        self._model = _STID(bundle)
        self._model.load_state_dict(bundle["state_dict"])
        self._model.eval()

    @cached_property
    def _distances(self):
        edges = pd.read_csv(self.root / "data" / "pems04" / "distance.csv",
                            dtype={"from": str, "to": str})
        positions = {sensor: i for i, sensor in enumerate(self.sensor_ids)}
        distances = np.full((len(positions), len(positions)), np.inf, dtype=np.float32)
        for source, target, cost in edges[["from", "to", "cost"]].itertuples(index=False, name=None):
            distances[positions[source], positions[target]] = cost
        distances = np.minimum(distances, distances.T)
        return shortest_path(np.where(np.isfinite(distances), distances, 0), method="D", directed=False)

    def _fault_indices(self, fault):
        mode = fault["mode"]
        minutes = fault["minutes"]
        percent = fault["percent"]
        nodes = len(self.sensor_ids)
        count = round(nodes * percent / 100)
        if mode == "random" and percent in (5, 20, 30) and minutes in (15, 30, 45):
            # The first repetition of the notebook's seeded random scenario.
            generator = torch.Generator().manual_seed(int(fault.get("seed", 2026)) * 1000)
            return torch.rand(nodes, generator=generator).topk(count).indices.numpy()
        if mode == "region" and percent == 10 and minutes in (15, 30):
            center = str(fault["center"])
            if center not in self.sensor_ids:
                raise ValueError(f"Không có cảm biến tâm {center}.")
            distances = self._distances[self.sensor_ids.index(center)]
            reachable = np.isfinite(distances).sum()
            if reachable < count:
                raise ValueError(
                    f"Vùng quanh cảm biến {center} chỉ nối được {reachable} cảm biến, "
                    f"cần {count} để mô phỏng 10%. Hãy chọn tâm khác, ví dụ cảm biến 0."
                )
            return np.argsort(distances)[:count]
        raise ValueError("Tỷ lệ hoặc thời lượng không thuộc kịch bản mất cảm biến đã chọn.")

    def affected_sensor_ids(self, fault):
        """Return the sensor IDs masked by this scenario."""
        if not fault or fault["mode"] == "clean":
            return ()
        return tuple(self.sensor_ids[i] for i in self._fault_indices(fault))

    @torch.inference_mode()
    def predict(self, frame, fault=None):
        """Return float32 flow counts in (sensor, horizon) order."""
        if tuple(frame.columns) != self.sensor_ids:
            raise ValueError("ID hoặc thứ tự cột cảm biến không khớp scaler train.")
        index = frame.index
        if not isinstance(index, pd.DatetimeIndex) or len(index) != self.n_past or index.hasnans:
            raise ValueError(f"Cần đúng {self.n_past} mốc thời gian để dự báo.")
        expected = pd.date_range(index[0], periods=self.n_past, freq=f"{self.step_minutes}min")
        if not index.equals(expected) or index[0] != index[0].floor(f"{self.step_minutes}min"):
            raise ValueError(f"Các mốc phải liên tiếp, đúng lưới {self.step_minutes} phút.")
        raw = torch.from_numpy(frame.to_numpy(dtype=np.float32, copy=True))
        observed = torch.isfinite(raw) & (raw >= 0)
        if fault and fault["mode"] != "clean":
            indices = self._fault_indices(fault)
            steps = int(fault["minutes"]) // self.step_minutes
            observed[-steps:, indices] = False
        scaled = (torch.where(observed, raw, self._fill) - self._mean) / self._std
        mask = observed.float()
        # Compute the neighbor feature *after* masking the simulated outage.
        numerator = torch.einsum("nm,tm->tn", self._neighbors, scaled * mask)
        denominator = torch.einsum("nm,tm->tn", self._neighbors, mask)
        neighbor = torch.where(denominator > 1e-6, numerator / denominator.clamp_min(1e-6), scaled)
        angle = 2 * np.pi * (index.hour.to_numpy() * 60 + index.minute.to_numpy()) / 1440.0
        time = np.stack([np.sin(angle), np.cos(angle), index.dayofweek.to_numpy() >= 5], axis=1)
        time = torch.as_tensor(time, dtype=torch.float32)[:, None, :].expand(-1, len(self.sensor_ids), -1)
        x = torch.cat((scaled[..., None], mask[..., None], time, neighbor[..., None]), dim=-1)
        prediction = self._model(x.unsqueeze(0))[0]
        flow = (prediction * self._std[:, None] + self._mean[:, None]).clamp_min(0)
        if not torch.isfinite(flow).all():
            raise ValueError("Model trả về giá trị dự báo không hữu hạn.")
        return flow.numpy()
