# EXPERIMENT_SEARCH_V2
# Chạy trên toàn bộ train với stride=1. Chọn bằng validation MAPE, chỉ test model cuối.
EXPERIMENTS_V2 = [
    dict(name="stid-node-time", model="STID", scaler="log", n_past=12, hidden=128, lr=1e-3, use_exog=False),
    dict(name="stid-exog-24", model="STID", scaler="log", n_past=24, hidden=128, lr=1e-3, use_exog=True),
    dict(name="adaptive-exog", model="GNN_ADAPT", scaler="log", n_past=12, hidden=96, lr=1e-3, use_exog=True),
    dict(name="gnn-exog-24", model="GNN", scaler="log", n_past=24, hidden=128, lr=1e-3, use_exog=True),
    dict(name="stgcn-clean", model="STGCN", scaler="log", n_past=12, hidden=64, lr=1e-3, use_exog=False),
    dict(name="stgcn-exog", model="STGCN", scaler="log", n_past=12, hidden=64, lr=1e-3, use_exog=True),
]


def _v2_write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(_v2_json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def run_experiments_v2(plan=None, epoch_budget=100, smoke=False):
    """Thử tham số và kiến trúc, lưu checkpoint và kết quả thật ngay trong notebook."""
    if DEVICE.type != "cuda":
        raise RuntimeError("Thử nghiệm này cần kernel CUDA: Intel Deep Learning (GPU)")
    if CFG["FAST"] or CFG["SYNTHETIC"]:
        raise ValueError("Chỉ dùng dữ liệu thật, FAST=False, SYNTHETIC=False")
    plan = EXPERIMENTS_V2 if plan is None else plan
    original = CFG.copy()
    # Đây là tham số tối ưu hóa; công thức loss/metric không đổi.
    shared = dict(EPOCHS=epoch_budget, PATIENCE=15, TRAIN_STRIDE=1, NODE_EMB=16,
                  MODEL_DEPTH=3, DROPOUT=0.05, LR_PATIENCE=5, LR_FACTOR=0.5, MIN_LR=1e-5)
    _sources = ["".join(c.get("source", [])) for c in _v2_json.loads((PROJECT_DIR / "build_model.ipynb").read_text(encoding="utf-8"))["cells"]]
    _relevant = [s for s in _sources if any(k in s for k in
                 ("class MetricAcc:", "def train_model(", "class STGCN(", "class BaseScaler", "def build_adjacency(",
                  "# EXPERIMENT_INPUTS_V2", "# EXPERIMENT_MODELS_V2", "# EXPERIMENT_SEARCH_V2"))]
    manifest = dict(plan=plan, shared=shared, original_config=original, smoke=smoke,
                    source_hash=_v2_hashlib.sha256("\n".join(_relevant).encode()).hexdigest(),
                    data_hash=_v2_hashlib.sha256(Path(_v2_npz).read_bytes()).hexdigest())
    fingerprint = _v2_hashlib.sha256(_v2_json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    directory = OUTPUT_DIR / "notebook_v2" / (("smoke-" if smoke else "search-") + fingerprint)
    directory.mkdir(parents=True, exist_ok=True)
    _v2_write_json(directory / "manifest.json", manifest)
    results_path = directory / "trials.json"
    rows = _v2_json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []

    def persist():
        _v2_write_json(results_path, rows)
        flat = [{**r["config"], **{k: v for k, v in r.items() if k not in ("config", "history")}} for r in rows]
        if flat:
            pd.DataFrame(flat).to_csv(directory / "trials.csv", index=False)

    print("GPU:", torch.cuda.get_device_name(0), "| Kết quả:", directory, flush=True)
    try:
        CFG.update(shared)
        for trial in plan:
            config = {**trial, "epochs": epoch_budget, "seed": 42, "batch_size": 32, "eval_bs": 32, **shared}
            trial_id = _v2_hashlib.sha256(_v2_json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
            checkpoint = directory / f"{trial_id}.pt"
            previous = next((r for r in rows if r["trial_id"] == trial_id), None)
            if previous and previous["status"] == "ok" and checkpoint.exists():
                print(f"Đã có {trial['name']}: val_MAPE {previous['val_MAPE']:.3f}%", flush=True)
                continue
            if previous:
                rows.remove(previous)
            row = dict(trial_id=trial_id, config=config, status="running")
            rows.append(row); persist()
            configure_inputs_v2(trial["use_exog"])
            print(f"\nTHỬ {trial['name']} | {trial} | N_IN={N_IN}", flush=True)
            model = result = None
            started = time.time()
            try:
                tr, va = train_data, val_data
                if smoke:
                    tr = SplitData(tr.values[:512], tr.index[:512], "smoke_train")
                    va = SplitData(va.values[:256], va.index[:256], "smoke_val")
                model, result, test_result = train_model(
                    tr, va, test_data, trial["scaler"], trial["model"], trial["n_past"],
                    hidden=trial["hidden"], lr=trial["lr"], batch_size=32, eval_bs=32,
                    epochs=epoch_budget, seed=42, miss_rates=None, use_adj=True,
                    monitor="MAPE_avg", evaluate_test=False, tag=f"+v2-{trial['name']}", verbose=True)
                assert test_result == {}, "Tuning không được đánh giá test"
                row.update(status="ok", val_MAE=result["best_val_MAE"], val_MAPE=result["best_val_MAPE"],
                           val_RMSE=result["best_val_RMSE"], best_train_loss=result["best_train_loss"],
                           min_train_loss=min(result["history"]["train_loss"]), best_epoch=result["best_epoch"],
                           epochs_run=result["epochs_run"], n_params=result["n_params"], history=result["history"],
                           checkpoint=checkpoint.name, n_in=N_IN)
                # Chỉ lưu trọng số và tiền xử lý; không pickle lớp model của notebook.
                state = dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                             scaler_state=model.scaler.__dict__, fill=model.fill, config=config,
                             input_flags={k: CFG.get(k) for k in ("USE_EXOG", "USE_TOD", "USE_ADJ_FEAT", "RESIDUAL")})
                torch.save(state, checkpoint)
                print(f"XONG {trial['name']}: loss {row['best_train_loss']:.4f}, val_MAE {row['val_MAE']:.3f}, val_MAPE {row['val_MAPE']:.3f}%", flush=True)
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                print(f"LỖI {trial['name']}: {row['error']}", flush=True)
            finally:
                row["elapsed_s"] = round(time.time() - started, 2)
                persist()
                del model, result
                _v2_gc.collect(); torch.cuda.empty_cache()
        good = sorted((r for r in rows if r["status"] == "ok"), key=lambda r: (r["val_MAPE"], r["val_MAE"]))
        if not good:
            raise RuntimeError("Không có lượt thành công; xem trials.json")
        display(pd.DataFrame([{**r["config"], **{k: r[k] for k in ("best_train_loss", "val_MAE", "val_MAPE", "val_RMSE", "best_epoch")}} for r in good]))
        if smoke:
            print("Kiểm tra nhỏ hoàn tất; không dùng kết quả này để kết luận chất lượng.")
            return directory
        winner = good[0]
        final_path = directory / "final_result.json"
        if final_path.exists():
            final = _v2_json.loads(final_path.read_text(encoding="utf-8"))
        else:
            state = torch.load(directory / winner["checkpoint"], map_location="cpu", weights_only=False)
            config = state["config"]
            CFG.update(shared); CFG.update(state["input_flags"])
            configure_inputs_v2(config["use_exog"])
            model = build_model(config["model"], N_NODES, config["n_past"], config["hidden"], use_adj=True)
            model.load_state_dict(state["state_dict"])
            scaler = SCALERS[config["scaler"]](); scaler.__dict__.update(state["scaler_state"])
            model.scaler, model.fill, model.n_past, model.eval_bs = scaler, state["fill"], config["n_past"], 32
            te = Windows(test_data, scaler, model.fill, config["n_past"], 1)
            metrics = evaluate(model, te, scaler, bs=32)
            final = dict(winner=winner["config"], validation={k: winner[k] for k in ("val_MAE", "val_MAPE", "val_RMSE")},
                         train_loss=winner["best_train_loss"], test=metrics,
                         targets=dict(loss=0.05, MAPE_percent=12.0),
                         target_met=dict(loss=winner["best_train_loss"] <= 0.05, test_MAPE=metrics["MAPE_avg"] <= 12.0),
                         protocol="Giữ loss/metric; chọn bằng validation MAPE; chỉ test sau khi chốt model.")
            _v2_write_json(final_path, final)
            del model, te, state
        print("KẾT QUẢ CUỐI:", _v2_json.dumps(final, ensure_ascii=False, indent=2), flush=True)
        return directory
    finally:
        CFG.clear(); CFG.update(original)
        configure_inputs_v2(original.get("USE_EXOG", False))
        _v2_gc.collect(); torch.cuda.empty_cache()
