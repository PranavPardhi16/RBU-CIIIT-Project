"""Run: python tests/test_core.py   (or pytest)"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.metrics import events, evaluate, smooth_alarms, point_adjusted_f1, recall_at_budget
from src.synth import simulate
from src.invariants import InvariantMiner
from src.attribution import attribution_hits


def test_events():
    assert events([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]


def test_event_metrics():
    y = np.zeros(1000, int); y[100:150] = 1; y[500:520] = 1
    a = np.zeros(1000, int); a[110:120] = 1; a[800:805] = 1          # detect 1/2, one false alarm
    r = evaluate(y, a, period_s=1.0)
    assert r["events_detected"] == 1 and abs(r["event_recall"] - .5) < 1e-9 and abs(r["event_precision"] - .5) < 1e-9
    assert r["median_delay_s"] == 10 and r["pa_f1_inflated"] > r["event_f1"] * 0 and r["fa_per_day"] > 0


def test_pa_f1_inflates():
    y = np.zeros(1000, int); y[100:300] = 1; a = np.zeros(1000, int); a[100] = 1
    assert point_adjusted_f1(y, a) > 0.9      # one flagged point => whole event credited (the flaw)
    assert evaluate(y, a)["event_recall"] == 1.0


def test_debounce():
    a = np.zeros(20, int); a[3] = 1; a[10:16] = 1
    assert smooth_alarms(a, 5).sum() == 2 and smooth_alarms(a, 5)[3] == 0


def test_no_leak_and_labels():
    df, m = simulate(n=30000, seed=1, n_events=4)
    assert df["label"].iloc[: m["split"]].sum() == 0 and df["label"].sum() > 0
    assert len(m["events"]) == 4 and all(e["sensors"] for e in m["events"])


def test_invariants_recover_known_relation():
    rng = np.random.default_rng(0); a = rng.normal(size=5000); b = 2 * a + rng.normal(0, .05, 5000); c = rng.normal(size=5000)
    X = np.c_[a, b, c]; mdl = InvariantMiner(use_diff=False, min_r2=.9).fit(X[:4000], X[4000:])
    targets = {v["j"] for v in mdl.inv_}; assert targets == {0, 1}                 # a~b found, noise tag c excluded
    Xt = X.copy(); Xt[:, 1] += 3                                                   # spoof b
    assert mdl.score(Xt).mean() > 5 * mdl.score(X).mean()


def test_attribution():
    P = np.zeros((100, 3)); P[:, 1] = 5
    h, _ = attribution_hits(P, [dict(start=10, end=50, sensors=["B"], type="t")], ["A", "B", "C"])
    assert h[1] == 1.0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
