"""API tests (FastAPI TestClient). Run: python tests/test_api.py  (or pytest). Needs models/demo*.npz (icsguard train --demo)."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi.testclient import TestClient
from icsguard.service import create_app

ROOT = os.path.join(os.path.dirname(__file__), "..", "models")
client = TestClient(create_app(os.path.join(ROOT, "demo.npz"), os.path.join(ROOT, "demo_stream.npz")))


def test_health_and_ui():
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/"); assert r.status_code == 200 and "ICSGuard" in r.text


def test_model_info():
    m = client.get("/api/model").json(); assert "SP" in m["command_cols"] and len(m["tags"]) == 9 and m["demo_available"]


def test_ingest_rows_and_records_and_validation():
    m = client.get("/api/model").json(); n = len(m["tags"]); client.post("/api/reset")
    rows = [[50.0] * n for _ in range(40)]; r = client.post("/api/ingest", json={"rows": rows}); assert r.status_code == 200 and len(r.json()["results"]) == 40
    rec = [{t: 50.0 for t in m["tags"]}]; assert client.post("/api/ingest", json={"records": rec}).status_code == 200
    assert client.post("/api/ingest", json={"rows": [[1.0, 2.0]]}).status_code == 422
    assert client.post("/api/ingest", json={}).status_code == 422


def test_budget_validation_and_effect():
    a5 = client.post("/api/budget", json={"budget": 5}).json()["alpha"]; a2 = client.post("/api/budget", json={"budget": 2}).json()["alpha"]
    assert a2 <= a5; assert client.post("/api/budget", json={"budget": 0}).status_code == 422; client.post("/api/budget", json={"budget": 5})


def test_demo_replay_detects_attacks():
    info = client.get("/api/demo/info").json(); res = client.get("/api/demo/replay?n=20000").json()["results"]
    assert len(res) == 20000 and {"t", "p", "a", "y", "top"} <= set(res[0])
    hit = 0; tot = 0
    for e in info["events"]:
        if e["end"] <= 20000: tot += 1; hit += any(r["a"] for r in res[e["start"]:e["end"]])
    assert tot >= 1 and hit >= 1, (hit, tot)


def test_sse_stream_format():
    with client.stream("GET", "/api/demo/stream?speed=100") as r:
        assert r.status_code == 200 and "text/event-stream" in r.headers["content-type"]
        lines = []
        for ln in r.iter_lines():
            if ln: lines.append(ln)
            if len(lines) >= 4: break
    assert lines[0].startswith("event: meta") and any(l.startswith("data:") for l in lines)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
