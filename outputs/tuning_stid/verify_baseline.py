"""Evaluate the existing STID checkpoint on validation without retraining."""
import ast
import json
from pathlib import Path
import time

import nbformat

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "build_model - Nam Anh.ipynb"
nb = nbformat.read(NOTEBOOK, as_version=4)
nbformat.validate(nb)
ns = {"__name__": "__baseline_validation__", "__file__": str(NOTEBOOK)}
for index in (2, 4, 6, 10, 12, 13, 15, 17, 19, 21):
    exec(compile(nb.cells[index].source, f"{NOTEBOOK}:cell{index}", "exec"), ns)
    if index == 2 and ns["DEVICE"].type != "cuda":
        raise RuntimeError("This verification expects the configured CUDA kernel")

loader = ast.parse(nb.cells[25].source)
loader.body = [node for node in loader.body if isinstance(node, ast.FunctionDef)]
exec(compile(loader, f"{NOTEBOOK}:pickle_helpers", "exec"), ns)
started = time.monotonic()
model = ns["load_model_pkl"](ROOT / "outputs" / "stid_standard.pkl")
assert model.name_ == "STID"
window = ns["Windows"](ns["val_data"], model.scaler, model.fill, model.n_past, 1)
model.eval_bs = 32
metrics = ns["evaluate"](model, window, model.scaler, bs=32)
result = dict(checkpoint="outputs/stid_standard.pkl", split="validation", model="STID",
              validation=metrics, train_config=model.train_config,
              model_options=model.model_options,
              elapsed_s=round(time.monotonic() - started, 2))
path = ROOT / "outputs" / "tuning_stid" / "baseline_validation.json"
path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
