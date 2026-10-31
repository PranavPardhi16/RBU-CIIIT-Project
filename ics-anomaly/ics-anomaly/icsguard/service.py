"""FastAPI service: REST ingestion, live SSE demo replay, dashboard."""
from __future__ import annotations
import asyncio, json, os, threading
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from . import __version__
from .engine import Guard

STATIC = os.path.join(os.path.dirname(__file__), "static")


class Ingest(BaseModel):
    rows: list[list[float | None]] | None = None
    records: list[dict[str, float | None]] | None = None


class Budget(BaseModel):
    budget: float


def _pack(o, y=None):
    d = dict(t=o["t"], v=o["valid"], p=round(o["p"], 6), a=int(o["alarm"]), ca=int(o["cusum_alarm"]), st=round(o["cusum_stat"], 2), r=o["regime"], env=int(o["envelope"]), top=o["top"])
    if y is not None: d["y"] = int(y)
    return d


def create_app(model_path: str, stream_path: str | None = None) -> FastAPI:
    app = FastAPI(title="ICSGuard", version=__version__, description="Budget-calibrated, regime-aware attack detection for ICS telemetry")
    tmpl = Guard.load(model_path); st = dict(live=tmpl.clone(), last=None, run=None, alerts=[]); lock = threading.Lock()
    demo = None
    if stream_path and os.path.exists(stream_path):
        z = np.load(stream_path, allow_pickle=False); demo = dict(X=z["X"], y=z["y"], events=json.loads(str(z["events"])))
    html = open(os.path.join(STATIC, "dashboard.html"), encoding="utf-8").read()

    def info():
        f = tmpl.f
        return dict(version=__version__, tags=f.cols, command_cols=tmpl.command_cols, regime_window_s=int(tmpl.W), budget_alarm_runs_per_day=tmpl.budget,
                    alpha=tmpl.alpha, cusum_h=None if not np.isfinite(tmpl.h) else float(tmpl.h), n_invariants=int(len(f.inv_j)),
                    cusum_tags=[f.cols[int(f.inv_j[i]) % len(f.cols)] for i in tmpl.cs_idx], demo_available=demo is not None)

    @app.get("/", response_class=HTMLResponse)
    def index(): return html

    @app.get("/health")
    def health(): return {"status": "ok", "version": __version__}

    @app.get("/api/model")
    def model(): return info()

    @app.post("/api/budget")
    def set_budget(b: Budget):
        if not (0.1 <= b.budget <= 500): raise HTTPException(422, "budget must be in [0.1, 500] alarm runs/day")
        with lock: tmpl.set_budget(b.budget); st["live"].set_budget(b.budget)
        return info()

    @app.post("/api/reset")
    def reset():
        with lock: st.update(live=tmpl.clone(), last=None, run=None, alerts=[])
        return {"ok": True}

    @app.post("/api/ingest")
    def ingest(body: Ingest):
        cols = tmpl.f.cols
        if body.records is not None: rows = [[r.get(c) for c in cols] for r in body.records]
        elif body.rows is not None: rows = body.rows
        else: raise HTTPException(422, "provide 'rows' or 'records'")
        if len(rows) > 5000: raise HTTPException(413, "max 5000 rows per request")
        out = []
        with lock:
            for r in rows:
                if len(r) != len(cols): raise HTTPException(422, f"each row needs {len(cols)} values in order {cols}")
                x = np.array([np.nan if v is None else v for v in r], float)
                if np.isnan(x).any(): x = np.where(np.isnan(x), st["last"] if st["last"] is not None else 0.0, x)
                st["last"] = x; o = st["live"].process_one(x); out.append(_pack(o))
                if o["alarm"] and st["run"] is None: st["run"] = dict(start=o["t"], tags=o["top"], envelope=o["envelope"], kind="cusum" if o["cusum_alarm"] and not o["conf_alarm"] else "conformal")
                elif o["alarm"]: st["run"]["tags"] = o["top"]; st["run"]["envelope"] = st["run"]["envelope"] or o["envelope"]
                elif st["run"] is not None: st["run"]["end"] = o["t"]; st["alerts"] = (st["alerts"] + [st["run"]])[-200:]; st["run"] = None
        return {"results": out}

    @app.get("/api/alerts")
    def alerts():
        with lock: return {"alerts": st["alerts"], "open": st["run"]}

    def _demo():
        if demo is None: raise HTTPException(404, "no demo stream loaded (start with --stream)")
        return demo

    @app.get("/api/demo/info")
    def demo_info():
        d = _demo(); return dict(rows=int(len(d["y"])), events=d["events"])

    @app.get("/api/demo/replay")
    def demo_replay(n: int = 2000, budget: float | None = None):
        d = _demo(); g = tmpl.clone()
        if budget: g.set_budget(budget)
        n = min(n, len(d["y"]), 20000); return {"results": [_pack(g.process_one(d["X"][i]), d["y"][i]) for i in range(n)]}

    @app.get("/api/demo/stream")
    async def demo_stream(speed: int = 200, budget: float | None = None, start: int = 0):
        d = _demo(); g = tmpl.clone()
        if budget: g.set_budget(budget)
        meta = dict(alpha=g.alpha, h=None if not np.isfinite(g.h) else float(g.h), W=int(g.W))

        async def gen():
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
            i = 0; per = max(1, min(speed, 2000) // 10)
            while i < len(d["y"]):
                batch = [_pack(g.process_one(d["X"][j]), d["y"][j]) for j in range(i, min(i + per, len(d["y"])))]; i += per
                yield f"data: {json.dumps(batch)}\n\n"; await asyncio.sleep(0.1)
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
