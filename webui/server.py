#!/usr/bin/env python3
"""
AutoML Web UI — FastAPI server that watches a runs directory and serves
a live dashboard.

The server is intentionally simple. It exposes:
  - GET /                       serves the dashboard HTML
  - GET /api/health             liveness probe
  - GET /api/runs               list runs (latest first)
  - GET /api/runs/{run_id}      full state.json for a run
  - GET /api/runs/{run_id}/events
                                Server-Sent Events stream of state + log changes
  - GET /api/runs/{run_id}/attempts/{attempt_id}/log
                                last N lines of train_log.jsonl
  - GET /api/runs/{run_id}/file?path=...
                                serve a file inside the run dir (path-traversal safe)
  - POST /api/runs/{run_id}/gates/{gate_id}
                                body: {"action": "approve"|"reject"|"edit", "note": "..."}
                                updates state.gates[gate_id].status

State is read from the filesystem on every request — no DB, no caching.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"


class GateAction(BaseModel):
    action: str  # "approve" | "reject" | "edit"
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_run_dir(runs_dir: Path, run_id: str) -> Path:
    """Resolve a run_id to its directory, refusing traversal."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise HTTPException(400, "invalid run_id")
    p = (runs_dir / run_id).resolve()
    if not str(p).startswith(str(runs_dir.resolve())):
        raise HTTPException(400, "path traversal")
    if not p.is_dir():
        raise HTTPException(404, f"run not found: {run_id}")
    return p


def safe_run_file(run_dir: Path, rel_path: str) -> Path:
    """Resolve a relative path inside the run dir, refusing traversal."""
    if rel_path.startswith("/") or ".." in rel_path.split("/"):
        raise HTTPException(400, "invalid path")
    target = (run_dir / rel_path).resolve()
    if not str(target).startswith(str(run_dir.resolve())):
        raise HTTPException(400, "path traversal")
    if not target.is_file():
        raise HTTPException(404, f"file not found: {rel_path}")
    return target


def load_state(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        return {}
    try:
        return json.loads(state_path.read_text())
    except json.JSONDecodeError:
        # state.json could be mid-write; return empty rather than 500
        return {}


def save_state_atomic(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    tmp = run_dir / "state.json.tmp"
    final = run_dir / "state.json"
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(final)


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.is_dir():
        return []
    out = []
    for child in sorted(runs_dir.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        state = load_state(child)
        if not state:
            continue
        out.append({
            "run_id": state.get("run_id", child.name),
            "prompt": (state.get("prompt") or "")[:200],
            "status": state.get("status", "unknown"),
            "phase": state.get("phase", "unknown"),
            "created_at": state.get("created_at"),
            "updated_at": state.get("updated_at"),
            "n_attempts": len(state.get("attempts") or []),
            "best_attempt_id": state.get("best_attempt_id"),
            "has_report": bool(state.get("report_path")),
        })
    return out


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def make_app(runs_dir: Path) -> FastAPI:
    app = FastAPI(title="AutoML Dashboard")
    app.state.runs_dir = runs_dir

    # Mount static dir for /static/* asset URLs
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            raise HTTPException(500, "index.html missing")
        return FileResponse(index_path)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "runs_dir": str(runs_dir), "ts": now_iso()}

    @app.get("/api/runs")
    async def runs_index() -> list[dict[str, Any]]:
        return list_runs(runs_dir)

    @app.get("/api/runs/{run_id}")
    async def run_state(run_id: str) -> dict[str, Any]:
        run_dir = safe_run_dir(runs_dir, run_id)
        state = load_state(run_dir)
        if not state:
            raise HTTPException(404, "state.json not found or invalid")
        return state

    @app.get("/api/runs/{run_id}/attempts/{attempt_id}/log")
    async def attempt_log(run_id: str, attempt_id: str, tail: int = 500) -> JSONResponse:
        run_dir = safe_run_dir(runs_dir, run_id)
        # canonical location:
        log_path = run_dir / "attempts" / attempt_id / "train_log.jsonl"
        if not log_path.is_file():
            return JSONResponse({"lines": [], "n_total": 0})
        lines = log_path.read_text().splitlines()
        n_total = len(lines)
        sliced = lines[-tail:] if tail > 0 else lines
        parsed = []
        for line in sliced:
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                parsed.append({"raw": line})
        return JSONResponse({"lines": parsed, "n_total": n_total})

    @app.get("/api/runs/{run_id}/file")
    async def run_file(run_id: str, path: str) -> Any:
        run_dir = safe_run_dir(runs_dir, run_id)
        target = safe_run_file(run_dir, path)
        # render text-ish files inline, binary as download
        text_exts = {".md", ".txt", ".json", ".jsonl", ".py", ".csv", ".tsv", ".log"}
        if target.suffix.lower() in text_exts:
            return PlainTextResponse(target.read_text(errors="replace"))
        return FileResponse(target)

    @app.post("/api/runs/{run_id}/gates/{gate_id}")
    async def gate_action(run_id: str, gate_id: str, body: GateAction = Body(...)) -> dict[str, Any]:
        run_dir = safe_run_dir(runs_dir, run_id)
        state = load_state(run_dir)
        if not state:
            raise HTTPException(404, "state.json not found")
        gates = state.setdefault("gates", [])
        gate = next((g for g in gates if g.get("id") == gate_id), None)
        if gate is None:
            raise HTTPException(404, f"gate not found: {gate_id}")
        if gate.get("status") not in ("pending", None):
            raise HTTPException(409, f"gate already decided: {gate.get('status')}")
        if body.action == "approve":
            gate["status"] = "approved"
        elif body.action == "reject":
            gate["status"] = "rejected"
        elif body.action == "edit":
            gate["status"] = "edit_requested"
        else:
            raise HTTPException(400, f"unknown action: {body.action}")
        gate["decided_at"] = now_iso()
        if body.note:
            gate["user_note"] = body.note
        save_state_atomic(run_dir, state)
        return gate

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request) -> StreamingResponse:
        """Server-Sent Events stream: emits a `state` event every time state.json
        changes, and `attempt_log` events when any attempt log grows."""
        run_dir = safe_run_dir(runs_dir, run_id)

        async def gen() -> AsyncIterator[bytes]:
            state_mtime = 0.0
            log_offsets: dict[str, int] = {}
            # initial snapshot
            state = load_state(run_dir)
            yield _sse("state", state)
            while True:
                if await request.is_disconnected():
                    break
                changed = False
                state_path = run_dir / "state.json"
                if state_path.is_file():
                    m = state_path.stat().st_mtime
                    if m > state_mtime:
                        state_mtime = m
                        state = load_state(run_dir)
                        yield _sse("state", state)
                        changed = True
                # attempt logs
                attempts_dir = run_dir / "attempts"
                if attempts_dir.is_dir():
                    for attempt_dir in attempts_dir.iterdir():
                        log_path = attempt_dir / "train_log.jsonl"
                        if not log_path.is_file():
                            continue
                        key = attempt_dir.name
                        last_off = log_offsets.get(key, 0)
                        size = log_path.stat().st_size
                        if size > last_off:
                            with log_path.open("r") as f:
                                f.seek(last_off)
                                new_text = f.read()
                            log_offsets[key] = size
                            for raw in new_text.splitlines():
                                if not raw.strip():
                                    continue
                                try:
                                    parsed = json.loads(raw)
                                except json.JSONDecodeError:
                                    parsed = {"raw": raw}
                                yield _sse("attempt_log", {"attempt_id": key, "line": parsed})
                            changed = True
                # heartbeat
                if not changed:
                    yield b": keepalive\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _sse(event: str, data: Any) -> bytes:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("./automl_runs"),
                        help="Directory containing run subdirectories")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    runs_dir = args.runs_dir.expanduser().resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    import uvicorn
    app = make_app(runs_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
