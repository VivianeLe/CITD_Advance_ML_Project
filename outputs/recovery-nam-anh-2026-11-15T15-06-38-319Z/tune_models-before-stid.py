"""Run reproducible validation-only tuning using the real notebook pipeline.

Examples (use the same Python environment as the notebook)::
    python -u tune_models.py --smoke
    python -u tune_models.py --run

Screening uses stride 3 as a proxy; the two finalists restart from scratch with
stride 1 and up to 100 epochs. Only the validation-selected winner sees test.
Completed trials/checkpoints resume automatically when code, data and plan match.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_pipeline(notebook):
    cells = json.loads(notebook.read_text(encoding="utf-8-sig"))["cells"]
    code = ["".join(cell.get("source", [])) for cell in cells if cell["cell_type"] == "code"]
    markers = ("CFG = dict(", "def load_data(", "def clean(", "def make_split(",
               "def build_adjacency(", "class BaseScaler", "class STGCN(",
               "class MetricAcc", "def train_model(")
    selected = []
    for marker in markers:
        matches = [source for source in code if re.search(r"(?m)^" + re.escape(marker), source)]
        if len(matches) != 1:
            raise ValueError(f"Expected one notebook cell containing {marker!r}; got {len(matches)}")
        selected.append(matches[0])
    namespace = {"__name__": "__notebook_tuning__", "__file__": str(notebook)}

    def execute(source, definitions_only=False):
        tree = ast.parse(source)
        if definitions_only:
            tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(tree, str(notebook), "exec"), namespace)

    execute(selected[0])
    if namespace["CFG"]["SYNTHETIC"] or namespace["CFG"]["FAST"]:
        raise ValueError("Use real data with CFG FAST=False and SYNTHETIC=False for tuning")
    if namespace["DEVICE"].type != "cuda":
        raise RuntimeError("CUDA GPU required; run with the notebook's GPU Python environment")
    for source in selected[1:4]:
        execute(source, definitions_only=True)
    raw, distance = namespace["load_data"](namespace["CFG"])
    _, numeric = namespace["validate"](raw)
    frame = namespace["clean"](numeric, namespace["CFG"])
    values = frame.to_numpy()
    splits = namespace["make_split"](values, frame.index, namespace["CFG"]["SPLIT"])
    namespace.update(raw_df=raw, DIST=distance, df=frame, VALUES=values,
                     T_ALL=len(frame), N_NODES=frame.shape[1], SPLITS=splits,
                     train_data=splits["train"], val_data=splits["val"], test_data=splits["test"])
    for source in selected[4:]:
        execute(source)
    data_hash = hashlib.sha256(values.tobytes())
    data_hash.update(str(list(frame.columns)).encode())
    data_hash.update(frame.index.asi8.tobytes())
    if distance is not None:
        data_hash.update(distance.tobytes())
    return namespace, {"source_sha256": digest(selected), "data_sha256": data_hash.hexdigest(),
                       "cfg": namespace["CFG"].copy(), "shape": list(values.shape)}


class Tuner:
    def __init__(self, namespace, directory, retry_failed=False):
        self.ns, self.directory, self.retry_failed = namespace, directory, retry_failed
        self.path = directory / "trials.json"
        self.rows = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    def persist(self):
        write_json(self.path, finite_json(self.rows))
        flat = [{**row["config"], **{key: value for key, value in row.items()
                                    if key not in ("config", "history", "traceback")}} for row in self.rows]
        if flat:
            columns = sorted(set().union(*(row.keys() for row in flat)))
            with (self.directory / "trials.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(flat)

    def trial(self, stage, config, epochs, stride, smoke=False):
        identity = {"stage": stage, **config, "epochs": epochs, "stride": stride}
        trial_id = f"{stage}-{digest(identity)[:12]}"
        previous = next((row for row in self.rows if row["trial_id"] == trial_id), None)
        checkpoint = self.directory / f"{trial_id}.pt"
        if previous and previous["status"] == "ok" and checkpoint.exists():
            print(f"[resume] {trial_id} val_MAPE={previous['val_MAPE']:.3f}%", flush=True)
            return previous
        if previous and previous["status"] == "failed" and not self.retry_failed:
            return previous
        if previous:
            self.rows.remove(previous)
        row = {"trial_id": trial_id, "stage": stage, "config": identity, "status": "running"}
        self.rows.append(row)
        self.persist()
        ns, torch = self.ns, self.ns["torch"]
        started = time.time()
        original_stride = ns["CFG"]["TRAIN_STRIDE"]
        try:
            ns["CFG"]["TRAIN_STRIDE"] = stride
            train, validation = ns["train_data"], ns["val_data"]
            if smoke:
                split = ns["SplitData"]
                train = split(train.values[:512], train.index[:512], "smoke_train")
                validation = split(validation.values[:256], validation.index[:256], "smoke_val")
            print(f"\n[{trial_id}] {identity}", flush=True)
            model, result, _ = ns["train_model"](
                train, validation, ns["test_data"], config["scaler"], config["model"], config["n_past"],
                miss_rates=ns["CFG"]["MISS_RATES"] if config["augmentation"] == "fault" else None,
                epochs=epochs, seed=config["seed"], tag=f"+tune-{stage}", verbose=True,
                hidden=config["hidden"], lr=config["lr"], batch_size=config["batch_size"],
                eval_bs=config["eval_bs"], monitor="MAPE_avg", evaluate_test=False)
            for metric in ("MAE", "MAPE", "RMSE"):
                row[f"val_{metric}"] = float(result[f"best_val_{metric}"])
            if not all(math.isfinite(row[f"val_{metric}"]) for metric in ("MAE", "MAPE", "RMSE")):
                raise ValueError("Non-finite validation metrics")
            row.update(status="ok", best_epoch=result["best_epoch"], epochs_run=result["epochs_run"],
                       best_train_loss=result.get("best_train_loss"), n_params=result["n_params"],
                       history=result["history"], checkpoint=checkpoint.name)
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "scaler_state": model.scaler.__dict__, "fill": model.fill,
                        "config": identity, "trial_id": trial_id}, temporary)
            temporary.replace(checkpoint)
            del model
            print(f"[{trial_id}] val_MAE={row['val_MAE']:.3f} val_MAPE={row['val_MAPE']:.3f}%", flush=True)
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            print(f"[{trial_id}] FAILED: {row['error']}", flush=True)
        finally:
            ns["CFG"]["TRAIN_STRIDE"] = original_stride
            row["elapsed_s"] = round(time.time() - started, 2)
            self.persist()
            gc.collect()
            torch.cuda.empty_cache()
        return row

    @staticmethod
    def best(rows, count=2):
        good = sorted((row for row in rows if row["status"] == "ok"), key=lambda row: (row["val_MAPE"], row["val_MAE"]))
        unique, seen = [], set()
        for row in good:
            key = digest({key: value for key, value in row["config"].items() if key not in ("stage", "stride", "epochs")})
            if key not in seen:
                seen.add(key)
                unique.append(row)
        if not unique:
            raise RuntimeError("No successful trials; see trials.json for errors")
        return unique[:count]

    def test_winner(self, winner):
        path = self.directory / "final_result.json"
        if path.exists():
            prior = json.loads(path.read_text(encoding="utf-8"))
            if prior["winner_trial_id"] == winner["trial_id"]:
                print(json.dumps(prior, ensure_ascii=False, indent=2), flush=True)
                return
        ns, config = self.ns, winner["config"]
        # Only load checkpoints created by this runner in its fingerprinted run directory.
        state = ns["torch"].load(self.directory / winner["checkpoint"], map_location="cpu", weights_only=False)
        scaler = ns["SCALERS"][config["scaler"]]()
        scaler.__dict__.update(state["scaler_state"])
        model = ns["build_model"](config["model"], ns["N_NODES"], config["n_past"], config["hidden"])
        model.load_state_dict(state["state_dict"])
        model.scaler, model.fill, model.n_past, model.eval_bs = scaler, state["fill"], config["n_past"], config["eval_bs"]
        window = ns["Windows"](ns["test_data"], scaler, model.fill, config["n_past"], 1)
        metrics = ns["evaluate"](model, window, scaler, bs=config["eval_bs"])
        result = {"winner_trial_id": winner["trial_id"], "config": config,
                  "selection": "validation MAPE only; final-stage trials; no test used in tuning",
                  "validation": {key: winner[key] for key in ("val_MAE", "val_MAPE", "val_RMSE")},
                  "test": metrics, "training_loss_at_best_epoch": winner["best_train_loss"],
                  "targets": {"loss_max": 0.05, "MAPE_percent_max": 12.0},
                  "target_met": {"loss": winner["best_train_loss"] is not None and winner["best_train_loss"] <= 0.05,
                                 "test_MAPE": metrics["MAPE_avg"] <= 12.0},
                  "note": "MAE uses original data units. Loss and MAPE targets are distinct, not guaranteed."}
        write_json(path, finite_json(result))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="One real-data GPU epoch; no test evaluation")
    mode.add_argument("--run", action="store_true", help="18 screens + up to 8 refinements + 2 full finalists")
    parser.add_argument("--notebook", type=Path, default=ROOT / "build_model.ipynb")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "tuning")
    parser.add_argument("--screen-epochs", type=int, default=10)
    parser.add_argument("--full-epochs", type=int, default=100)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.screen_epochs < 1 or args.full_epochs < 1:
        parser.error("epoch budgets must be positive")
    os.chdir(ROOT)
    namespace, provenance = load_pipeline(args.notebook.resolve())
    plan = {"version": 1, "mode": "smoke" if args.smoke else "staged", "monitor": "MAPE_avg",
            "screen_epochs": args.screen_epochs, "full_epochs": args.full_epochs, "screen_stride": 3,
            "models": ["GRU", "GNN", "STGCN"], "scalers": ["standard", "minmax", "log"],
            "augmentation": ["clean", "fault"], "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest = finite_json({"provenance": provenance, "plan": plan})
    fingerprint = digest(manifest)
    directory = args.out.resolve() / f"{plan['mode']}-{fingerprint[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest["fingerprint"] = fingerprint
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("Resume manifest mismatch; choose a new output directory")
    write_json(manifest_path, manifest)
    print(f"Results: {directory}", flush=True)
    tuner = Tuner(namespace, directory, args.retry_failed)
    base = dict(model="STGCN", scaler="standard", augmentation="clean", n_past=12,
                hidden=64, lr=1e-3, batch_size=32, eval_bs=32, seed=namespace["CFG"]["SEED"])
    if args.smoke:
        row = tuner.trial("smoke", base, 1, 8, smoke=True)
        if row["status"] != "ok":
            raise RuntimeError(row["error"])
        return
    screened = []
    for model, scaler, augmentation in itertools.product(plan["models"], plan["scalers"], plan["augmentation"]):
        config = {**base, "model": model, "scaler": scaler, "augmentation": augmentation}
        screened.append(tuner.trial("screen", config, args.screen_epochs, 3))
    refined = []
    for candidate in tuner.best(screened):
        config = {key: value for key, value in candidate["config"].items() if key in base}
        for lr, window, hidden in ((3e-4, 12, 64), (1e-3, 24, 64), (3e-4, 24, 64), (1e-3, 12, 128)):
            refined.append(tuner.trial("refine", {**config, "lr": lr, "n_past": window, "hidden": hidden}, args.screen_epochs, 3))
    finalists = []
    for candidate in tuner.best(screened + refined):
        config = {key: value for key, value in candidate["config"].items() if key in base}
        finalists.append(tuner.trial("full", config, args.full_epochs, 1))
    tuner.test_winner(tuner.best(finalists, 1)[0])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed trials are saved and resume automatically.", flush=True)
        sys.exit(130)
