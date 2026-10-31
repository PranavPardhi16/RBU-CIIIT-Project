"""ConformGuard ablation benchmark: 10 simulated plants x alarm budgets x variants.
  python run_conformguard.py --seeds 10 --out results_cg
Variants isolate each mechanism: global-static (baseline) -> +Mondrian regimes -> +rolling calibration -> +CUSUM.
Reports realised alarm-runs/day vs budget (calibration), recall/F1, in-envelope false alarms, per-attack recall, attribution."""
import argparse, json, os, time, warnings
import numpy as np, pandas as pd
from src.synth import simulate, ATTACK_TYPES
from src.metrics import evaluate, event_detections, events
from src.attribution import attribution_hits
from src.stats import bootstrap_ci, paired_wilcoxon
from icsguard.engine import fit_frozen, Guard
warnings.filterwarnings("ignore")

VARIANTS = {
    "ar_static":       dict(components=("ar",), mondrian=False, rolling=False, cusum=False),
    "fused_static":    dict(mondrian=False, rolling=False, cusum=False),
    "cg_mondrian":     dict(mondrian=True, rolling=False, cusum=False),
    "cg_mond_roll":    dict(mondrian=True, rolling=True, cusum=False),
    "cg_mond_cusum":   dict(mondrian=True, rolling=False, cusum=True),
    "cg_full":         dict(mondrian=True, rolling=True, cusum=True),
    "ar_mond":         dict(components=("ar",), mondrian=True, rolling=False, cusum=False),
    "ar_mond_cusum":   dict(components=("ar",), mondrian=True, rolling=False, cusum=True),
}


def run_seed(seed, n, n_events, budgets, offset=0):
    df, meta = simulate(n=n, seed=seed + offset, n_events=n_events); h = meta["split"]; cols = [c for c in df.columns if c != "label"]
    X = df[cols].values; y = df["label"].values[h:]; Xte = X[h:]
    a, b = int(.7 * h), int(.8 * h); Xtr, Xes, Xcal = X[:a], X[a:b], X[b:h]; pool = np.vstack([Xes, Xcal])
    ev = [dict(e, start=e["start"] - h, end=e["end"] - h) for e in meta["events"]]
    mc = [c - h for c in meta["mode_changes"] if c >= h]; days = len(y) / 86400.0
    fr = fit_frozen(Xtr, Xes, cols); out = {}
    for name, kw in VARIANTS.items():
        g = Guard(fr, command_cols=["SP"], **{"components": ("ar", "inv"), **kw}).calibrate(pool, budget=budgets[0], Xenv=np.vstack([Xtr, pool])); tr = g.analyze(Xte)
        for B in budgets:
            d = g.decide(tr, B); al = d["alarm"]; r = evaluate(y, al)
            fas = [(s, e) for s, e in events(al) if y[s:e].sum() == 0]; fa_in = [(s, e) for s, e in fas if not tr["env"][s:e].any()]
            det = {x["start"]: x for x in event_detections(y, al)}
            r["fa_in_env_per_day"] = len(fa_in) / days; r["fa_total_per_day"] = len(fas) / days
            r["per_type"] = {t: float(np.mean([det[e["start"]]["detected"] for e in ev if e["type"] == t])) for t in ATTACK_TYPES}
            r["fa_near_mc"] = float(np.mean([any(0 <= s - c <= 900 for c in mc) for s, e in fas])) if fas else np.nan
            if True:
                hh, _ = attribution_hits(tr["Pn"], ev, cols); r["attr_hit1"], r["attr_hit3"] = hh[1], hh[3]
            r.update(W=int(g.W), alpha=float(g.alpha), h=float(g.h)); out[f"{name}|{B}"] = r
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--n", type=int, default=150000)
    ap.add_argument("--n_events", type=int, default=12); ap.add_argument("--budgets", default="2,5,10"); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default="results_cg"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); budgets = [float(x) for x in a.budgets.split(",")]
    runs = []
    for s in range(a.seeds):
        t = time.time(); runs.append(run_seed(s, a.n, a.n_events, budgets, a.offset)); print(f"seed {s} {time.time()-t:.0f}s", flush=True)
    keys = list(runs[0].keys()); rows = []
    for k in keys:
        v, B = k.split("|"); g = lambda f: np.array([f(r[k]) for r in runs], float); row = dict(variant=v, budget=float(B))
        for m in ["event_recall", "event_precision", "event_f1", "fa_per_day", "fa_in_env_per_day", "median_delay_s", "pa_f1_inflated"]:
            x = g(lambda r: r[m]); row[m] = np.nanmean(x); lo, hi = bootstrap_ci(x); row[m + "_lo"], row[m + "_hi"] = lo, hi
        row["calib_ratio_in_env_median"] = float(np.median(g(lambda r: r["fa_in_env_per_day"]) / float(B)))
        for t in ATTACK_TYPES: row["recall_" + t] = np.nanmean(g(lambda r: r["per_type"][t]))
        if "attr_hit3" in runs[0][k]: row["attr_hit1"] = g(lambda r: r["attr_hit1"]).mean(); row["attr_hit3"] = g(lambda r: r["attr_hit3"]).mean()
        row["fa_near_mode_change"] = np.nanmean(g(lambda r: r["fa_near_mc"])); rows.append(row)
    S = pd.DataFrame(rows); S.to_csv(f"{a.out}/summary.csv", index=False)
    sig = []
    for B in budgets:
        for x, y_ in [("cg_mondrian", "fused_static"), ("cg_full", "fused_static"), ("cg_full", "cg_mondrian"), ("cg_mond_cusum", "cg_mondrian"), ("cg_mond_roll", "cg_mondrian"), ("cg_full", "ar_static"), ("ar_mond", "ar_static"), ("ar_mond_cusum", "ar_static"), ("ar_mond_cusum", "ar_mond")]:
            for m in ["event_f1", "event_recall", "fa_in_env_per_day"]:
                sig.append(dict(budget=B, a=x, b=y_, metric=m, **paired_wilcoxon([r[f"{x}|{B}"][m] for r in runs], [r[f"{y_}|{B}"][m] for r in runs])))
    pd.DataFrame(sig).to_csv(f"{a.out}/significance.csv", index=False); json.dump(runs, open(f"{a.out}/raw.json", "w"), default=float)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for B in budgets:
        print(f"\n=== budget {B:g} alarm-runs/day ==="); print(S[S.budget == B][["variant", "event_recall", "event_precision", "event_f1", "fa_per_day", "fa_in_env_per_day", "calib_ratio_in_env_median", "recall_ramp_stealth", "recall_replay_mask", "recall_flow_tamper", "fa_near_mode_change"]].round(2).to_string(index=False))
    print("\n=== significance (Wilcoxon, paired over seeds) ==="); print(pd.DataFrame(sig).query("budget==5").round(4).to_string(index=False))
    print("\nattribution (full):", S[(S.variant == "cg_full") & (S.budget == 5)][["attr_hit1", "attr_hit3"]].round(3).to_dict("records"))


if __name__ == "__main__":
    main()
