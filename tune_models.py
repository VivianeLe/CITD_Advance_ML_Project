"""Tune STID + StandardScaler by validation MAPE using the notebook pipeline.

    python -u tune_models.py --smoke
    python -u tune_models.py --run --max-trials 4
    python -u tune_models.py --run --stage full --evaluate-final-test

STGCN is never trained here. Screening/refinement use stride 1 by default;
the two finalists restart with the full epoch budget. Test is untouched unless explicitly
requested after final selection. Re-run the same command to resume safely.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import time
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parent
CONFIG_KEYS = ("model", "scaler", "n_past", "hidden", "lr", "batch_size", "eval_bs",
               "node_emb", "depth", "dropout", "miss_rates", "patience", "lr_patience", "seed")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
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


def load_pipeline(notebook, allow_cpu=False):
    cells = json.loads(notebook.read_text(encoding="utf-8-sig"))["cells"]
    code = ["".join(cell.get("source", [])) for cell in cells if cell["cell_type"] == "code"]
    markers = ("CFG = dict(", "def load_data(", "def clean(", "def make_split(",
               "def build_adjacency(", "class StandardScaler", "class STGCN(",
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
    cfg = namespace["CFG"]
    if cfg.get("SYNTHETIC", False) or cfg["FAST"]:
        raise ValueError("Use real data with CFG FAST=False for tuning")
    if namespace["DEVICE"].type != "cuda" and not allow_cpu:
        raise RuntimeError("CUDA GPU required; --allow-cpu is available for small smoke runs")
    if cfg["MODEL"] != "STID" or cfg["SCALER"] != "standard":
        raise ValueError("This runner requires notebook MODEL='STID', SCALER='standard'")
    for source in selected[1:4]:
        execute(source, definitions_only=True)
    raw, distance = namespace["load_data"](cfg)
    frame = namespace["clean"](raw, cfg)
    values = frame.to_numpy()
    splits = namespace["make_split"](values, frame.index, cfg["SPLIT"])
    namespace.update(raw_df=raw, DIST=distance, df=frame, VALUES=values,
                     T_ALL=len(frame), N_NODES=frame.shape[1], SPLITS=splits,
                     train_data=splits["train"], val_data=splits["val"], test_data=splits["test"])
    for source in selected[4:]:
        execute(source)
    if "STID" not in namespace["MODELS"]:
        raise ValueError("The selected notebook does not define STID")
    data_hash = hashlib.sha256(values.tobytes())
    data_hash.update(str(list(frame.columns)).encode())
    data_hash.update(frame.index.asi8.tobytes())
    data_hash.update(distance.tobytes())
    return namespace, {"source_sha256": digest(selected), "data_sha256": data_hash.hexdigest(),
                       "cfg": cfg.copy(), "shape": list(values.shape),
                       "device": str(namespace["DEVICE"]), "torch_version": namespace["torch"].__version__}


def cfg_overrides(config, epochs=None, stride=1):
    overrides = {"MODEL": "STID", "SCALER": "standard", "N_PAST": config["n_past"],
                 "HIDDEN": config["hidden"], "LR": config["lr"], "BATCH": config["batch_size"],
                 "EVAL_BATCH": config["eval_bs"], "NODE_EMB": config["node_emb"],
                 "MODEL_DEPTH": config["depth"], "DROPOUT": config["dropout"],
                 "MISS_RATES": tuple(config["miss_rates"]), "TRAIN_STRIDE": stride,
                 "PATIENCE": config["patience"], "LR_PATIENCE": config["lr_patience"]}
    if epochs is not None:
        overrides["EPOCHS"] = epochs
    return overrides


class BudgetExhausted(Exception):
    pass


class Tuner:
    def __init__(self, namespace, directory, retry_failed=False, max_trials=None, max_minutes=None):
        self.ns, self.directory, self.retry_failed = namespace, directory, retry_failed
        self.path = directory / "trials.json"
        self.rows = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        self.max_trials, self.completed_now = max_trials, 0
        self.deadline = time.monotonic() + max_minutes * 60 if max_minutes else None

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
        if config["model"] != "STID" or config["scaler"] != "standard":
            raise ValueError("Only STID + StandardScaler may be tuned")
        identity = {"stage": stage, **config, "epochs": epochs, "stride": stride}
        trial_id = f"{stage}-{digest(identity)[:12]}"
        previous = next((row for row in self.rows if row["trial_id"] == trial_id), None)
        checkpoint = self.directory / f"{trial_id}.pt"
        if previous and previous["status"] == "ok" and checkpoint.exists():
            print(f"[resume] {trial_id} val_MAPE={previous['val_MAPE']:.3f}%", flush=True)
            return previous
        if previous and previous["status"] == "failed" and not self.retry_failed:
            return previous
        if ((self.max_trials is not None and self.completed_now >= self.max_trials)
                or (self.deadline is not None and time.monotonic() >= self.deadline)):
            raise BudgetExhausted
        if previous:
            self.rows.remove(previous)
        row = {"trial_id": trial_id, "stage": stage, "config": identity, "status": "running"}
        self.rows.append(row)
        self.persist()
        ns, torch = self.ns, self.ns["torch"]
        started = time.time()
        original_cfg = ns["CFG"].copy()
        model = None
        try:
            ns["CFG"].update(cfg_overrides(config, epochs, stride))
            train, validation = ns["train_data"], ns["val_data"]
            if smoke:
                split = ns["SplitData"]
                train = split(train.values[:128], train.index[:128], "smoke_train")
                validation = split(validation.values[:96], validation.index[:96], "smoke_val")
            print(f"\n[{trial_id}] {identity}", flush=True)
            model, result, _ = ns["train_model"](
                train, validation, ns["test_data"], "standard", "STID", config["n_past"],
                miss_rates=tuple(config["miss_rates"]),
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
                        "a_nb": model.A_nb.detach().cpu().numpy(),
                        "model_options": dict(model.model_options), "train_config": dict(model.train_config),
                        "cfg": dict(ns["CFG"]), "result": result,
                        "config": identity, "trial_id": trial_id}, temporary)
            temporary.replace(checkpoint)
            print(f"[{trial_id}] val_MAE={row['val_MAE']:.3f} val_MAPE={row['val_MAPE']:.3f}%", flush=True)
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            print(f"[{trial_id}] FAILED: {row['error']}", flush=True)
        finally:
            ns["CFG"].clear()
            ns["CFG"].update(original_cfg)
            row["elapsed_s"] = round(time.time() - started, 2)
            self.completed_now += 1
            self.persist()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return row

    @staticmethod
    def best(rows, count=2):
        good = sorted((row for row in rows if row["status"] == "ok"), key=lambda row: (row["val_MAPE"], row["val_MAE"]))
        unique, seen = [], set()
        for row in good:
            key = digest({key: row["config"][key] for key in CONFIG_KEYS})
            if key not in seen:
                seen.add(key)
                unique.append(row)
        if not unique:
            raise RuntimeError("No successful trials; see trials.json for errors")
        return unique[:count]

    def export_winner(self, evaluate_test=False):
        good = [row for row in self.rows if row["status"] == "ok" and row["stage"] != "smoke"]
        if not good:
            return None
        finals = [row for row in good if row["stage"] == "full"]
        winner = self.best(finals or good, 1)[0]
        ns, config = self.ns, winner["config"]
        state = ns["torch"].load(self.directory / winner["checkpoint"], map_location="cpu", weights_only=False)
        bundle = dict(state_dict=state["state_dict"], model_name="STID", train_config=state["train_config"],
                      cfg=state["cfg"], model_options=state["model_options"], n_nodes=ns["N_NODES"],
                      n_in=ns["N_IN"], n_past=config["n_past"], horizons=list(ns["HORIZONS"]),
                      scaler={"name": "standard", "mu": state["scaler_state"]["mu"], "sd": state["scaler_state"]["sd"]},
                      fill=state["fill"], a_nb=state["a_nb"], train_result=state["result"], test_clean={},
                      saved_at=ns["pd"].Timestamp.now().isoformat(), torch_version=ns["torch"].__version__)
        temporary = self.directory / "best_model.pkl.tmp"
        with temporary.open("wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(self.directory / "best_model.pkl")
        result = {"winner_trial_id": winner["trial_id"], "stage": winner["stage"],
                  "provisional": not bool(finals), "config": config,
                  "cfg_overrides": finite_json(cfg_overrides(config, config["epochs"], 1)),
                  "validation": {key: winner[key] for key in ("val_MAE", "val_MAPE", "val_RMSE")},
                  "selection": "validation MAPE only; full-stage candidates preferred; test never used in selection",
                  "checkpoint": str(self.directory / winner["checkpoint"]),
                  "model_pkl": str(self.directory / "best_model.pkl")}
        write_json(self.directory / "best_config.json", result)
        if evaluate_test:
            if not finals:
                raise RuntimeError("Run a full-stage finalist before evaluating test")
            prior_path = self.directory / "final_result.json"
            if prior_path.exists():
                prior = json.loads(prior_path.read_text(encoding="utf-8"))
                if prior["winner_trial_id"] == winner["trial_id"]:
                    bundle["test_clean"] = prior["test"]
                    with temporary.open("wb") as handle:
                        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    temporary.replace(self.directory / "best_model.pkl")
                    print(json.dumps(prior, ensure_ascii=False, indent=2), flush=True)
                    return winner
            torch = ns["torch"]
            model = ns["MODELS"]["STID"](ns["N_NODES"], ns["N_IN"], ns["H"], config["hidden"],
                                          config["n_past"], state["state_dict"]["A"].to(ns["DEVICE"]),
                                          model_options=state["model_options"]).to(ns["DEVICE"])
            model.load_state_dict(state["state_dict"])
            scaler = ns["SCALERS"]["standard"]()
            scaler.__dict__.update(state["scaler_state"])
            model.A_nb = torch.as_tensor(state["a_nb"], device=ns["DEVICE"])
            model.scaler, model.fill, model.n_past, model.eval_bs = scaler, state["fill"], config["n_past"], config["eval_bs"]
            window = ns["Windows"](ns["test_data"], scaler, model.fill, config["n_past"], 1)
            metrics = ns["evaluate"](model, window, scaler, bs=config["eval_bs"])
            write_json(prior_path, finite_json({**result, "test": metrics}))
            bundle["test_clean"] = metrics
            with temporary.open("wb") as handle:
                pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(self.directory / "best_model.pkl")
        print(f"[best] {winner['trial_id']} validation MAPE={winner['val_MAPE']:.4f}% -> {self.directory / 'best_config.json'}", flush=True)
        return winner


def screen_configs(base):
    # Explicit designs avoid a huge Cartesian search and preserve the baseline.
    # The first six are the initial GPU budget; retain the same fault augmentation.
    updates = ({}, {"hidden": 128, "n_past": 24}, {"hidden": 64, "n_past": 24},
               {"hidden": 128, "n_past": 12},
               {"hidden": 128, "n_past": 24, "node_emb": 32},
               {"hidden": 128, "n_past": 24, "dropout": 0.1, "lr": 6e-4},
               {"lr": 3e-4}, {"lr": 6e-4}, {"node_emb": 32},
               {"depth": 2}, {"depth": 4, "dropout": 0.1}, {"dropout": 0.0})
    return [{**base, **update} for update in updates]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="One tiny real-data epoch, no test evaluation")
    mode.add_argument("--run", action="store_true", help="12 screens, up to 8 refinements, 2 full finalists")
    parser.add_argument("--notebook", type=Path, default=ROOT / "build_model - Nam Anh.ipynb")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "tuning_stid")
    parser.add_argument("--stage", choices=("all", "screen", "refine", "full"), default="all")
    parser.add_argument("--screen-epochs", type=int, default=40)
    parser.add_argument("--screen-stride", type=int, default=1, help="Use >1 only for a cheaper training proxy")
    parser.add_argument("--full-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-trials", type=int, help="Maximum new trials this invocation; cached trials are free")
    parser.add_argument("--max-minutes", type=float, help="Stop before starting another trial after this wall-time budget")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Permit CPU execution (intended for smoke checks)")
    parser.add_argument("--evaluate-final-test", action="store_true", help="Evaluate only the selected full-stage winner on test")
    args = parser.parse_args()
    if min(args.screen_epochs, args.screen_stride, args.full_epochs, args.patience, args.lr_patience, args.batch_size, args.eval_batch_size) < 1 or (args.max_trials is not None and args.max_trials < 1) or (args.max_minutes is not None and (not math.isfinite(args.max_minutes) or args.max_minutes <= 0)):
        parser.error("epoch/trial/time budgets must be positive")
    if args.smoke and args.evaluate_final_test:
        parser.error("--smoke cannot evaluate test")
    os.chdir(ROOT)
    ns, provenance = load_pipeline(args.notebook.resolve(), args.allow_cpu)
    cfg = ns["CFG"]
    base = dict(model="STID", scaler="standard", n_past=cfg["N_PAST"], hidden=cfg["HIDDEN"],
                lr=cfg["LR"], batch_size=args.batch_size, eval_bs=args.eval_batch_size,
                node_emb=cfg["NODE_EMB"], depth=cfg["MODEL_DEPTH"], dropout=cfg["DROPOUT"],
                miss_rates=tuple(cfg["MISS_RATES"]), patience=args.patience, lr_patience=args.lr_patience, seed=cfg["SEED"])
    screens = screen_configs(base)
    plan = {"version": 2, "mode": "smoke" if args.smoke else "staged", "model": "STID", "scaler": "standard",
            "monitor": "MAPE_avg", "screen_epochs": args.screen_epochs, "full_epochs": args.full_epochs,
            "screen_stride": args.screen_stride, "screens": screens,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest = finite_json({"provenance": provenance, "plan": plan})
    fingerprint = digest(manifest)
    directory = args.out.resolve() / f"{plan['mode']}-{fingerprint[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest["fingerprint"] = fingerprint
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("Resume manifest mismatch; choose a new output directory")
    write_json(manifest_path, manifest)
    print(f"STID-only results: {directory}", flush=True)
    tuner = Tuner(ns, directory, args.retry_failed, args.max_trials, args.max_minutes)
    if args.smoke:
        row = tuner.trial("smoke", {**base, "hidden": 16, "node_emb": 4, "depth": 1}, 1, 8, smoke=True)
        if row["status"] != "ok":
            raise RuntimeError(row["error"])
        return
    try:
        screened = [row for row in tuner.rows if row["stage"] == "screen"]
        if args.stage in ("all", "screen"):
            screened = [tuner.trial("screen", config, args.screen_epochs, args.screen_stride) for config in screens]
        refined = [row for row in tuner.rows if row["stage"] == "refine"]
        if args.stage in ("all", "refine"):
            refined = []
            for candidate in tuner.best(screened):
                config = {key: candidate["config"][key] for key in CONFIG_KEYS}
                for update in ({"lr": config["lr"] * 0.5}, {"hidden": 128, "node_emb": 32},
                               {"n_past": 24, "dropout": 0.1}, {"batch_size": 32}):
                    refined.append(tuner.trial("refine", {**config, **update}, args.screen_epochs, args.screen_stride))
        if args.stage in ("all", "full"):
            for candidate in tuner.best(screened + refined):
                config = {key: candidate["config"][key] for key in CONFIG_KEYS}
                tuner.trial("full", config, args.full_epochs, 1)
    except BudgetExhausted:
        print("Invocation budget reached; completed STID trials are saved. Re-run to continue.", flush=True)
    finally:
        winner = tuner.export_winner(evaluate_test=args.evaluate_final_test)
        if winner is None and tuner.completed_now:
            raise RuntimeError("No successful STID trials; inspect trials.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed STID trials are saved and resume automatically.", flush=True)
        sys.exit(130)
