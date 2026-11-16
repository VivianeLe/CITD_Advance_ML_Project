"""Verify a tuned STID export on CPU without loading data or training.

Usage::
    python outputs/tuning_stid/verify_export.py outputs/tuning_stid/staged-...
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import pickle
import tempfile


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def find_cell(cells, marker):
    matches = ["".join(cell.get("source", [])) for cell in cells
               if cell["cell_type"] == "code"
               and marker in "".join(cell.get("source", []))]
    check(len(matches) == 1, f"Expected one snapshot cell containing {marker!r}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="STID experiment directory")
    args = parser.parse_args()
    directory = args.directory.resolve()

    # Set this before importing torch; every tensor/model below stays on CPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_num_threads(2)
    device = torch.device("cpu")
    export_path = directory / "best_model.pkl"
    snapshot = directory / "source-notebook.ipynb"
    best = json.loads((directory / "best_config.json").read_text(encoding="utf-8"))
    rows = json.loads((directory / "trials.json").read_text(encoding="utf-8"))
    winner = next(row for row in rows if row["trial_id"] == best["winner_trial_id"])
    check(winner["status"] == "ok", "The selected trial did not complete successfully")
    check(winner["config"]["model"] == "STID", "The selected trial is not STID")
    # Resolve by basename so the export can be verified on another OS/path.
    checkpoint_path = directory / Path(winner["checkpoint"]).name
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    with export_path.open("rb") as handle:
        bundle = pickle.load(handle)
    check(bundle["model_name"] == "STID", "The exported model is not STID")
    check(checkpoint["trial_id"] == winner["trial_id"], "Checkpoint winner mismatch")
    check(bundle["model_options"] == checkpoint["model_options"], "Exported options differ from checkpoint")
    check(bundle["train_config"] == checkpoint["train_config"], "Exported training configuration differs")

    def check_state(actual):
        expected = checkpoint["state_dict"]
        check(set(actual) == set(expected), "State dictionary keys differ from checkpoint")
        for name, tensor in actual.items():
            check(tensor.device.type == "cpu", f"{name} is not on CPU")
            check(torch.equal(tensor, expected[name]), f"Checkpoint tensor differs: {name}")

    check_state(bundle["state_dict"])
    cells = json.loads(snapshot.read_text(encoding="utf-8-sig"))["cells"]
    cfg = dict(bundle["cfg"])
    ns = {
        "__name__": "__stid_export_verification__", "__file__": str(snapshot),
        "np": np, "pd": pd, "torch": torch, "nn": nn, "F": F,
        "math": math, "pickle": pickle, "Path": Path, "CFG": cfg,
        "DEVICE": device, "N_NODES": int(bundle["n_nodes"]),
        "N_IN": int(bundle["n_in"]), "H": len(bundle["horizons"]),
        "HORIZONS": list(bundle["horizons"]),
        "ADJ_RAW": np.asarray(bundle["a_nb"]),
        "A_HAT_T": bundle["state_dict"]["A"].to(device),
    }
    scaler_tree = ast.parse(find_cell(cells, "class StandardScaler"))
    scaler_tree.body = [node for node in scaler_tree.body
                        if isinstance(node, ast.ClassDef) and node.name == "StandardScaler"]
    check(len(scaler_tree.body) == 1, "StandardScaler definition is missing")
    exec(compile(scaler_tree, f"{snapshot}:scaler", "exec"), ns)
    ns["SCALERS"] = {"standard": ns["StandardScaler"]}
    model_source = find_cell(cells, "class STIDNet(")
    exec(compile(model_source, f"{snapshot}:models", "exec"), ns)
    check(ns["N_IN"] == bundle["n_in"], "Snapshot input channels differ from export")
    helper_tree = ast.parse(find_cell(cells, "def load_model_pkl("))
    helper_tree.body = [node for node in helper_tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name in ("save_model_pkl", "load_model_pkl")]
    check(len(helper_tree.body) == 2, "Snapshot pickle helpers are missing")
    exec(compile(helper_tree, f"{snapshot}:pickle_helpers", "exec"), ns)

    model = ns["load_model_pkl"](export_path, device=device)
    check_state(model.state_dict())
    check(model.model_options == bundle["model_options"], "Loaded options differ from export")
    check(np.array_equal(model.scaler.mu, bundle["scaler"]["mu"]), "Scaler means differ")
    check(np.array_equal(model.scaler.sd, bundle["scaler"]["sd"]), "Scaler standard deviations differ")
    check(np.array_equal(model.fill, bundle["fill"]), "Imputer values differ")
    check(np.array_equal(model.A_nb.numpy(), bundle["a_nb"]), "Neighbour adjacency differs")

    # Synthetic input only: this does not touch the training/validation/test data.
    batch, steps, nodes = 2, int(bundle["n_past"]), int(bundle["n_nodes"])
    base_channels = int(bundle["n_in"]) - int(bool(cfg["USE_ADJ_FEAT"]))
    generator = torch.Generator(device="cpu").manual_seed(2026)
    inputs = torch.randn(batch, steps, nodes, base_channels, generator=generator)
    inputs[..., 1] = 1.0
    inputs[:, :, ::7, 1] = 0.0
    if cfg["USE_TOD"]:
        angle = (torch.arange(steps, dtype=torch.float32) + 280) * (2 * math.pi / 288)
        inputs[..., 2] = angle.sin()[None, :, None]
        inputs[..., 3] = angle.cos()[None, :, None]
        inputs[..., 4] = torch.tensor([0.0, 1.0])[:, None, None]
    inputs = ns["add_adj_feature"](inputs, model.A_nb)
    check(inputs.shape == (batch, steps, nodes, bundle["n_in"]), "Fixture input shape differs")

    def predict(loaded):
        check(not loaded.training, "Loaded model is not in eval mode")
        check(all(value.device.type == "cpu" for value in loaded.parameters()), "Model is not on CPU")
        with torch.no_grad():
            prediction = loaded(inputs)
        check(tuple(prediction.shape) == (batch, nodes, len(bundle["horizons"])), "Prediction shape differs")
        check(bool(torch.isfinite(prediction).all()), "Prediction contains non-finite values")
        return prediction

    original = predict(model)
    cfg.update(NODE_EMB=int(bundle["model_options"]["node_emb"]) + 7,
               MODEL_DEPTH=int(bundle["model_options"]["depth"]) + 2,
               DROPOUT=0.67, RESIDUAL=not bool(bundle["model_options"]["residual"]))
    reloaded = ns["load_model_pkl"](export_path, device=device)
    check(reloaded.model_options == bundle["model_options"], "Ambient CFG overwrote saved model options")
    check_state(reloaded.state_dict())
    changed_cfg_prediction = predict(reloaded)
    torch.testing.assert_close(changed_cfg_prediction, original, rtol=0, atol=0)

    with tempfile.TemporaryDirectory(prefix="stid-export-verification-") as temporary:
        roundtrip_path = Path(temporary) / "roundtrip.pkl"
        ns["save_model_pkl"](reloaded, roundtrip_path, bundle.get("train_result"), bundle.get("test_clean"))
        roundtrip = ns["load_model_pkl"](roundtrip_path, device=device)
        check(roundtrip.model_options == bundle["model_options"], "Roundtrip options differ")
        check_state(roundtrip.state_dict())
        torch.testing.assert_close(predict(roundtrip), original, rtol=0, atol=0)

    report = {
        "status": "passed", "device": "cpu", "training_performed": False,
        "data_loaded": False, "winner_trial_id": winner["trial_id"],
        "source_snapshot": str(snapshot), "export": str(export_path),
        "checkpoint": str(checkpoint_path), "prediction_shape": list(original.shape),
        "checks": ["checkpoint_state_matches", "saved_scaler_imputer_adjacency_match",
                   "finite_prediction_with_expected_shape", "ambient_cfg_does_not_override_options",
                   "reload_predictions_identical", "save_load_roundtrip_identical"],
    }
    report_path = directory / "export_verification.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
