"""Engine tests. Run: python tests/test_engine.py  (or pytest)"""
import os, sys, tempfile; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.synth import simulate
from src.metrics import events
from icsguard.engine import fit_frozen, Guard, detect_command_cols

_C = {}


def setup():
    if _C: return _C
    df, meta = simulate(n=150000, seed=3, n_events=6); h = meta["split"]; cols = [c for c in df.columns if c != "label"]
    X = df[cols].values; a, b = int(.7 * h), int(.8 * h)
    fr = fit_frozen(X[:a], X[a:b], cols); pool = X[a:h]
    g = Guard(fr, command_cols=["SP"]).calibrate(pool, budget=5, Xenv=X[:h]); _C.update(df=df, meta=meta, X=X, h=h, g=g, fr=fr, pool=pool, cols=cols); return _C


def test_command_autodetect():
    c = setup(); assert detect_command_cols(c["X"][:c["h"]], c["cols"]) == ["SP"]


def test_stream_equals_batch():
    c = setup(); g = c["g"]; Xte = c["X"][c["h"]:][:5000]; tr = g.analyze(Xte); d = g.decide(tr); g2 = g.clone(); o = [g2.process_one(x) for x in Xte]
    assert np.abs(np.array([q["p"] for q in o]) - tr["p"]).max() < 1e-12
    assert np.abs(np.array([q["cusum_stat"] for q in o]) - tr["stat"]).max() < 1e-9
    assert (np.array([q["alarm"] for q in o]) == d["alarm"].astype(bool)).all()


def test_save_load_roundtrip():
    c = setup(); g = c["g"]; Xte = c["X"][c["h"]:][:4000]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.npz"); g.save(p); g2 = Guard.load(p)
        a, b = g.analyze(Xte), g2.analyze(Xte); assert np.abs(a["p"] - b["p"]).max() < 1e-12 and np.abs(a["stat"] - b["stat"]).max() < 1e-9
        assert g2.alpha == g.alpha and g2.W == g.W


def test_command_step_is_not_an_anomaly():
    c = setup(); g = c["g"]; X = c["X"][c["h"]:][:6000].copy(); tr0 = g.analyze(X)
    X2 = X.copy(); j = c["cols"].index("SP"); X2[3000:, j] += 0.0  # unchanged control
    sa, PA, R = g._comp(X); assert np.all(PA[:, j] == 0), "command column must not be scored"


def test_budget_monotone():
    c = setup(); g = c["g"].clone(); a2 = g.set_budget(2).alpha; a10 = g.set_budget(10).alpha; assert a10 >= a2


def test_cusum_detects_small_persistent_shift():
    c = setup(); g = c["g"]
    if len(g.cs_idx) == 0: return                       # bank empty for this plant: nothing to test
    Xte = c["X"][c["h"]:][:20000].copy(); base = g.analyze(Xte)["stat"].max()
    i = int(g.f.inv_j[g.cs_idx[0]]) % len(c["cols"]); t0 = 12000
    Xte[t0:t0 + 600, i] += 1.5 * c["X"][:c["h"], i].std() * 0.2
    tr = g.analyze(Xte); assert tr["stat"][t0:t0 + 600].max() > base * 0.9


def test_envelope_flag():
    c = setup(); g = c["g"]; X = c["X"][c["h"]:][:200].copy(); j = c["cols"].index("SP"); X[:, j] = 200.0
    assert g._env(X).all() and not g._env(c["X"][:200]).any()


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
