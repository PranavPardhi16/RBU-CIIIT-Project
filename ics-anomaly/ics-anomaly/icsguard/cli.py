"""ICSGuard command line.
  icsguard train --demo --out models/demo            # simulate a plant, train, calibrate, save demo model + replay stream
  icsguard train --csv plant.csv --label-col label --command-cols SP --out models/plant --budget 5
  icsguard replay --model models/plant.npz --csv new.csv [--budget 5]
  icsguard serve --model models/demo.npz --stream models/demo_stream.npz --port 8000
  icsguard info --model models/demo.npz
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


def _split(X, a=0.7, b=0.8):
    n = len(X); return X[:int(a * n)], X[int(a * n):int(b * n)], X[int(b * n):]


def train_from_array(X, cols, command_cols=None, budget=5.0):
    from .engine import fit_frozen, Guard, detect_command_cols
    Xtr, Xes, Xcal = _split(X); cmd = command_cols if command_cols is not None else detect_command_cols(X, cols)
    fr = fit_frozen(Xtr, Xes, cols); pool = np.vstack([Xes, Xcal])
    return Guard(fr, command_cols=cmd).calibrate(pool, budget=budget, Xenv=np.vstack([Xtr, pool]))


def cmd_train(a):
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.demo:
        from src.synth import simulate
        df, meta = simulate(n=150000, seed=a.seed, n_events=12); h = meta["split"]; cols = [c for c in df.columns if c != "label"]
        g = train_from_array(df[cols].values[:h], cols, ["SP"], a.budget); g.save(a.out + ".npz")
        ev = [dict(type=e["type"], start=e["start"] - h, end=e["end"] - h, sensors=e["sensors"]) for e in meta["events"]]
        np.savez_compressed(a.out + "_stream.npz", X=df[cols].values[h:], y=df["label"].values[h:], cols=np.array(cols), events=json.dumps(ev))
        print(f"demo model -> {a.out}.npz ; replay stream -> {a.out}_stream.npz ({len(df) - h} rows, {len(ev)} labelled attacks)")
    else:
        import pandas as pd
        df = pd.read_csv(a.csv)
        if a.label_col and a.label_col in df.columns: df = df[df[a.label_col] == 0]
        drop = [c for c in [a.label_col, a.time_col] if c and c in df.columns]
        num = df.drop(columns=drop).select_dtypes("number"); cols = [c for c in num.columns if num[c].std() > 0]
        g = train_from_array(num[cols].values.astype(float), cols, a.command_cols.split(",") if a.command_cols else None, a.budget); g.save(a.out + ".npz")
    g = __import__("icsguard.engine", fromlist=["Guard"]).Guard.load(a.out + ".npz")
    print(f"tags={len(g.f.cols)} command_cols={g.command_cols} regime_window_W={g.W}s invariants={len(g.f.inv_j)} cusum_tags={len(g.cs_idx)} budget={g.budget}/day alpha={g.alpha:.5f} h={g.h}")


def cmd_replay(a):
    import pandas as pd
    from .engine import Guard
    g = Guard.load(a.model)
    if a.budget: g.set_budget(a.budget)
    df = pd.read_csv(a.csv); X = df[g.f.cols].values.astype(float); tr = g.analyze(X); d = g.decide(tr)
    from src.metrics import events
    runs = events(d["alarm"]); print(f"{len(X)} rows, {len(runs)} alarm runs ({len(runs) / (len(X) / 86400):.1f}/day), {int(tr['env'].any())} envelope-flag={bool(tr['env'].any())}")
    for s, e in runs[:50]:
        tags = [g.f.cols[j] for j in np.argsort(-tr["Pn"][s:e].mean(0))[:3]]; print(f"  t={s}-{e} ({e - s}s) top tags: {tags} cusum={bool(d['cusum'][s:e].any())}")
    if "label" in df.columns:
        from src.metrics import evaluate
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in evaluate(df["label"].values, d["alarm"]).items()})


def cmd_serve(a):
    import uvicorn
    from .service import create_app
    uvicorn.run(create_app(a.model, a.stream), host=a.host, port=a.port, log_level="info")


def cmd_info(a):
    from .engine import Guard
    g = Guard.load(a.model); print(json.dumps(dict(tags=g.f.cols, command_cols=g.command_cols, W=g.W, invariants=len(g.f.inv_j), cusum_tags=[g.f.cols[int(g.f.inv_j[i]) % len(g.f.cols)] for i in g.cs_idx], budget=g.budget, alpha=g.alpha, h=g.h), indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="icsguard"); sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--demo", action="store_true"); t.add_argument("--csv"); t.add_argument("--label-col"); t.add_argument("--time-col")
    t.add_argument("--command-cols"); t.add_argument("--budget", type=float, default=5.0); t.add_argument("--seed", type=int, default=7); t.add_argument("--out", required=True); t.set_defaults(f=cmd_train)
    r = sp.add_parser("replay"); r.add_argument("--model", required=True); r.add_argument("--csv", required=True); r.add_argument("--budget", type=float); r.set_defaults(f=cmd_replay)
    v = sp.add_parser("serve"); v.add_argument("--model", required=True); v.add_argument("--stream"); v.add_argument("--host", default="0.0.0.0"); v.add_argument("--port", type=int, default=8000); v.set_defaults(f=cmd_serve)
    i = sp.add_parser("info"); i.add_argument("--model", required=True); i.set_defaults(f=cmd_info)
    a = ap.parse_args(argv); a.f(a)


if __name__ == "__main__":
    main()
