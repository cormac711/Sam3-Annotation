"""Thin wrapper around ../3_export.py's `run()` for the app backend.

No logic duplicated: 3_export.py's export logic (YOLO/crops dataset
building) is unchanged and shared by both the CLI and this wrapper. This
module only adapts argparse.Namespace construction and progress reporting
to match the shape server.py expects (same DetectProgress-style callback
pattern used by detect_core, for a consistent SSE contract on both jobs).
"""

import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import importlib.util

from detect_core import Cancelled

if getattr(sys, "frozen", False):
    # PyInstaller onedir: bundled as a plain data file next to the frozen
    # backend (see sam3_backend.spec) -- __file__-relative lookup below
    # doesn't work once this module is embedded in the PYZ archive rather
    # than sitting on disk next to a real ../3_export.py.
    _script_path = Path(sys._MEIPASS) / "3_export.py"
else:
    _script_path = Path(__file__).resolve().parent.parent / "3_export.py"

_spec = importlib.util.spec_from_file_location("export_script", _script_path)
_export_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_export_script)


@dataclass
class ExportParams:
    annotations: str
    images_dir: Optional[str] = None
    out: str = "dataset"
    format: str = "both"
    val_ratio: float = 0.1
    seed: int = 42
    pad: float = 0.05
    min_crop: int = 32
    link: bool = False


@dataclass
class ExportProgress:
    done: int
    total: int


def run_job(params: ExportParams, progress_cb: Callable[[ExportProgress], None],
            should_cancel: Callable[[], bool] = lambda: False):
    args = Namespace(
        annotations=params.annotations,
        images_dir=params.images_dir,
        out=params.out,
        format=params.format,
        val_ratio=params.val_ratio,
        seed=params.seed,
        pad=params.pad,
        min_crop=params.min_crop,
        link=params.link,
    )

    def cb(done, total):
        if should_cancel():
            raise Cancelled()
        progress_cb(ExportProgress(done=done, total=total))

    return _export_script.run(args, progress_cb=cb)
