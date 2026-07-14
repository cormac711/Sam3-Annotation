"""Local-only FastAPI backend for the Sam3 Annotation app.

Spawned by Electron as a child process bound to 127.0.0.1:<port> (port is
handed in via --port, chosen by Electron before spawn). Never listens on any
non-loopback interface -- this is a sidecar for one desktop app, not a
network service.

Two job types (detect, export) share the same lifecycle: POST /start kicks
off a background thread, GET /stream (SSE) streams progress until it ends
with a "done"/"error"/"cancelled" event, POST /cancel requests a graceful
stop. `JobRunner` implements that lifecycle once for both.
"""

import argparse
import asyncio
import json
import queue
import sys
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_core
import export_core

app = FastAPI()
# The renderer loads via file:// or a custom scheme depending on packaging;
# CORS is irrelevant for a loopback sidecar with no cookies/auth, opened wide
# here purely so localhost dev (npm start / vite, if ever added) isn't blocked.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

runner = detect_core.Sam3Runner()


def _to_jsonable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    return obj


class JobRunner:
    """One background job at a time, progress pushed to an SSE stream."""

    def __init__(self, name):
        self.name = name
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._events: "queue.Queue" = queue.Queue()
        self._cancel = threading.Event()
        self._running = False

    @property
    def running(self):
        return self._running

    def start(self, job_fn):
        """`job_fn(progress_cb, should_cancel) -> result` -- caller supplies a
        closure over whatever args its job needs, so this stays agnostic to
        each job type's signature."""
        with self._lock:
            if self._running:
                raise RuntimeError(f"a {self.name} job is already running")
            self._running = True
            self._cancel.clear()
            self._events = queue.Queue()

            def body():
                try:
                    result = job_fn(self._on_progress, self._cancel.is_set)
                    self._events.put({"type": "done", "result": _to_jsonable(result)})
                except detect_core.Cancelled:
                    self._events.put({"type": "cancelled"})
                except Exception as e:
                    self._events.put({"type": "error", "message": str(e)})
                finally:
                    self._running = False

            self._thread = threading.Thread(target=body, daemon=True)
            self._thread.start()

    def _on_progress(self, progress):
        self._events.put({"type": "progress", **_to_jsonable(progress)})

    def cancel(self):
        self._cancel.set()

    def stream(self):
        while True:
            item = self._events.get()
            yield f"data: {json.dumps(item)}\n\n"
            if item["type"] in ("done", "error", "cancelled"):
                return


detect_job = JobRunner("detect")
export_job = JobRunner("export")


# --------------------------------------------------------------------------
# health / system
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/system/gpu")
def system_gpu():
    return detect_core.gpu_status()


@app.get("/system/hf-auth")
def system_hf_auth():
    try:
        from huggingface_hub import whoami
        info = whoami()
        return {"authenticated": True, "name": info.get("name")}
    except Exception:
        return {"authenticated": False}


class HfLoginBody(BaseModel):
    token: str


@app.post("/system/hf-login")
def system_hf_login(body: HfLoginBody):
    from huggingface_hub import login
    try:
        login(token=body.token, add_to_git_credential=False)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


# --------------------------------------------------------------------------
# detect
# --------------------------------------------------------------------------
class DetectStartBody(BaseModel):
    input_dir: str
    prompt: str = "person"
    threshold: float = 0.5
    mask_threshold: float = 0.5
    model_id: str = "facebook/sam3"
    device: str = "auto"
    precision: str = "fp16"
    batch_size: int = 4
    vram_limit_mb: Optional[int] = None
    recursive: bool = False
    save_masks: bool = True
    preview: int = 0
    no_resume: bool = False
    compile_model: bool = False
    output: Optional[str] = None


@app.post("/detect/start")
def detect_start(body: DetectStartBody):
    params = detect_core.DetectParams(**body.model_dump())
    try:
        detect_job.start(lambda cb, should_cancel:
                          detect_core.run_job(runner, params, cb, should_cancel))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@app.get("/detect/stream")
def detect_stream():
    return StreamingResponse(detect_job.stream(), media_type="text/event-stream")


@app.post("/detect/cancel")
def detect_cancel():
    detect_job.cancel()
    return {"ok": True}


@app.post("/detect/unload")
def detect_unload():
    runner.unload()
    return {"ok": True}


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
class ExportStartBody(BaseModel):
    annotations: str
    images_dir: Optional[str] = None
    out: str = "dataset"
    format: str = "both"
    val_ratio: float = 0.1
    seed: int = 42
    pad: float = 0.05
    min_crop: int = 32
    link: bool = False


@app.post("/export/start")
def export_start(body: ExportStartBody):
    params = export_core.ExportParams(**body.model_dump())
    try:
        export_job.start(lambda cb, should_cancel:
                          export_core.run_job(params, cb, should_cancel))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@app.get("/export/stream")
def export_stream():
    return StreamingResponse(export_job.stream(), media_type="text/event-stream")


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------
def main():
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8756)
    args = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
