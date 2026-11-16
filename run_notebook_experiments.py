"""Execute selected notebook experiments in an isolated CUDA kernel.

    python -u run_notebook_experiments.py --smoke
    python -u run_notebook_experiments.py --run
    python -u run_notebook_experiments.py --snapshot

Real cell outputs are saved into the source notebook during execution. Source
and unrelated cells are preserved; concurrent edits to selected source abort
the run instead of replacing the edited code with an older snapshot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
SETUP_MARKERS = (
    "CFG = dict(", "def load_data(", "def validate(", "class SplitData:",
    "def build_adjacency(", "class BaseScaler:", "class STGCN(",
    "class MetricAcc:", "def train_model(", "def get_region(",
    "# EXPERIMENT_INPUTS_V2", "# EXPERIMENT_MODELS_V2", "# EXPERIMENT_SEARCH_V2",
)
DECODER = json.JSONDecoder()
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def raw_cells(text):
    """Locate raw cell spans without rewriting notebook-level JSON."""
    match = re.search(r'"cells"\s*:\s*\[', text)
    if match is None:
        raise ValueError("Notebook has no cells array")
    position = match.end()
    while True:
        while text[position].isspace() or text[position] == ",":
            position += 1
        if text[position] == "]":
            return
        start = position
        cell, position = DECODER.raw_decode(text, position)
        yield start, position, cell


def update_output_fields(text, outputs, count):
    """Replace only outputs/count JSON values, preserving source/metadata bytes."""
    changes, position = [], 1
    wanted = {"outputs": outputs, "execution_count": count}
    while True:
        while text[position].isspace() or text[position] == ",":
            position += 1
        if text[position] == "}":
            break
        key, position = DECODER.raw_decode(text, position)
        while text[position].isspace():
            position += 1
        if text[position] != ":":
            raise ValueError("Invalid notebook cell JSON")
        position += 1
        while text[position].isspace():
            position += 1
        start = position
        _, position = DECODER.raw_decode(text, position)
        if key in wanted:
            changes.append((start, position, json.dumps(wanted.pop(key), ensure_ascii=False)))
    if wanted:
        additions = ", ".join(json.dumps(key) + ": " + json.dumps(value, ensure_ascii=False)
                              for key, value in wanted.items())
        changes.append((position, position, ", " + additions))
    for start, end, value in sorted(changes, reverse=True):
        text = text[:start] + value + text[end:]
    return text


class LiveNotebook:
    def __init__(self, path, notebook, selected):
        self.path, self.notebook, self.selected = path, notebook, selected
        self.executed = set()
        self.last_save = 0.0

    def save(self, force=False):
        if not force and time.monotonic() - self.last_save < 15:
            return
        for _ in range(3):
            original = self.path.read_bytes()
            text = original.decode("utf-8-sig")
            cells = list(raw_cells(text))
            patches = []
            for index, marker, expected_source in self.selected:
                matches = [(start, end, cell) for start, end, cell in cells
                           if cell.get("cell_type") == "code"
                           and re.search(r"(?m)^" + re.escape(marker), "".join(cell.get("source", [])))]
                if len(matches) != 1 or "".join(matches[0][2].get("source", [])) != expected_source:
                    raise RuntimeError(f"Notebook source changed while running: {marker}")
                if index in self.executed:
                    start, end, _ = matches[0]
                    executed = self.notebook.cells[index]
                    patches.append((start, end, update_output_fields(
                        text[start:end], executed.get("outputs", []), executed.get("execution_count"))))
            for start, end, replacement in sorted(patches, reverse=True):
                text = text[:start] + replacement + text[end:]
            encoded = (b"\xef\xbb\xbf" if original.startswith(b"\xef\xbb\xbf") else b"") + text.encode("utf-8")
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            try:
                temporary.write_bytes(encoded)
                if self.path.read_bytes() != original:
                    continue  # Merge an editor save into a fresh snapshot.
                os.replace(temporary, self.path)
                self.last_save = time.monotonic()
                return
            finally:
                if temporary.exists():
                    temporary.unlink()
        raise RuntimeError("Notebook keeps changing; stopped to preserve concurrent edits")


class LoggingClient(NotebookClient):
    def __init__(self, *args, live, write, **kwargs):
        super().__init__(*args, **kwargs)
        self.live, self.write = live, write

    def process_message(self, message, cell, cell_index):
        try:
            return super().process_message(message, cell, cell_index)
        finally:
            kind, content = message.get("msg_type"), message.get("content", {})
            if kind == "stream":
                self.write(("[stderr] " if content.get("name") == "stderr" else "") + content.get("text", ""))
            elif kind in ("display_data", "execute_result", "update_display_data"):
                data = content.get("data", {})
                self.write("[display] " + data.get("text/plain", ", ".join(data))[:2000] + "\n")
            elif kind == "error":
                self.write(ANSI.sub("", "\n".join(content.get("traceback", []))) + "\n")
            self.live.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Run the dedicated smoke cell only")
    mode.add_argument("--run", action="store_true", help="Run the real experiment search cell")
    mode.add_argument("--snapshot", action="store_true", help="Evaluate a completed checkpoint without running the search")
    parser.add_argument("--notebook", type=Path, default=ROOT / "build_model.ipynb")
    parser.add_argument("--kernel", default="intel-dl")
    args = parser.parse_args()
    path = args.notebook.resolve()
    directory = path.parent / "outputs" / "notebook_v2"
    if args.snapshot:
        directory = directory / "snapshot"
    directory.mkdir(parents=True, exist_ok=True)
    original = path.read_bytes()
    notebook = nbformat.reads(original.decode("utf-8-sig"), as_version=4)
    execution_marker = ("# EXPERIMENT_SNAPSHOT_V2" if args.snapshot else
                        "# EXPERIMENT_SMOKE_V2" if args.smoke else "# EXPERIMENT_RUN_V2")
    markers = (*SETUP_MARKERS, execution_marker)
    selected = []
    for marker in markers:
        matches = [(index, cell.source) for index, cell in enumerate(notebook.cells)
                   if cell.cell_type == "code" and re.search(r"(?m)^" + re.escape(marker), cell.source)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one code cell for {marker!r}; found {len(matches)}")
        selected.append((matches[0][0], marker, matches[0][1]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup = directory / f"backup-{path.stem}-{stamp}-{os.getpid()}.ipynb"
    with backup.open("xb") as handle:
        handle.write(original)
    status = dict(pid=os.getpid(), mode="snapshot" if args.snapshot else "smoke" if args.smoke else "run", state="starting",
                  started_utc=stamp, notebook=str(path), backup=str(backup), current_cell=None)
    status_path = directory / "status.json"
    atomic_json(status_path, status)
    live = LiveNotebook(path, notebook, selected)
    with (directory / "run.log").open("a", encoding="utf-8", buffering=1) as log:
        def write(value):
            log.write(value); log.flush()
            try:
                sys.stdout.write(value); sys.stdout.flush()
            except UnicodeEncodeError:
                pass

        write(f"\n[{stamp}] pid={os.getpid()} mode={status['mode']} backup={backup.name}\n")
        client = LoggingClient(notebook, live=live, write=write, timeout=None,
                               kernel_name=args.kernel, allow_errors=False,
                               resources={"metadata": {"path": str(path.parent)}})
        try:
            with client.setup_kernel():
                for index, marker, _ in selected:
                    status.update(state="running", current_cell=index, current_marker=marker)
                    atomic_json(status_path, status)
                    write(f"\n[cell {index}] {marker}\n")
                    live.executed.add(index)
                    try:
                        client.execute_cell(notebook.cells[index], index)
                    finally:
                        live.save(force=True)
            status.update(state="complete", finished_utc=datetime.now(timezone.utc).isoformat())
            write("\nNotebook execution completed.\n")
        except BaseException as exc:
            status.update(state="failed", error=f"{type(exc).__name__}: {exc}",
                          finished_utc=datetime.now(timezone.utc).isoformat())
            write(traceback.format_exc() + "\n")
            raise
        finally:
            atomic_json(status_path, status)


if __name__ == "__main__":
    main()
