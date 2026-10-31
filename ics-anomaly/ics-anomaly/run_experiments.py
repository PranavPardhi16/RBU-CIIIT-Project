"""Multi-seed experiment runner: detectors, ablations, event-wise metrics, attribution,
benign-mode-change false-alarm analysis, statistics, figures.

  python run_experiments.py --seeds 10                     # synthetic benchmark (default)
  python run_experiments.py --hai "data/hai/train*.csv" --hai_test "data/hai/test*.csv" --seeds 3
"""
import argparse, json, os, time, warnings
import numpy as np, pandas as pd
from src.synth import simulate, ATTACK_TYPES
from src.data import load_hai, feature_cols
import re
from src.models import IForest, PCAResidual, ARResidual, MLPAE, HybridScore, GatedScore
from src.invariants import InvariantMiner
from src.metrics import (evaluate, threshold_from_normal, threshold_for_alarm_budget, smooth_alarms, tradeoff_curve, recall_at_budget,
                         event_detections, false_alarm_runs)
from src.attribution import PerSensorNormalizer, attribution_hits
from src.stats import bootstrap_ci, paired_wilcoxon
warnings.filterwarnings("ignore")

BASE = ["iforest", "pca", "ar", "mlp_ae", "invariant", "invariant_nodiff", "lstm_ae"]
HYBRIDS = {  # name: (components, lambdas)
    "hybrid_pca_inv": (["pca", "invariant"], [1, 1]), "hybrid_ar_inv": (["ar", "invariant"], [1, 1]),
    "hybrid_mlp_inv": (["mlp_ae", "invariant"], [1, 1]),
    "hybrid_mlp_inv_lam0.5": (["mlp_ae", "invariant"], [1, .5]), "hybrid_mlp_inv_lam2": (["mlp_ae", "invariant"], [1, 2]),
    "hybrid_mlp_ar": (["mlp_ae", "ar"], [1, 1]),  # ablation: two learned scorers, no invariants
    "hybrid_lstm_inv": (["lstm_ae", "invariant"], [1, 1]),
}


def dyn(name):
    """'inv_gate600' -> ('gate', None, 600); 'hybrid_ar_invgate600' -> ('hyb', 'ar', 600)."""
    m = re.fullmatch(r"inv_gate(\d+)", name)
    if m: return ("gate", None, int(m.group(1)))
    m = re.fullmatch(r"hybrid_(mlp|ar|pca)_invgate(\d+)", name)
    if m: return ("hyb", {"mlp": "mlp_ae", "ar": "ar", "pca": "pca"}[m.group(1)], int(m.group(2)))
    return None


class Inv:
    def __init__(self, **kw): self.m = InvariantMiner(**kw)
    def fit(self, X, Xes=None): self.m.fit(X, Xes); return self
    def score(self, X): return self.m.score(X)
    def per_sensor(self, X): return self.m.per_sensor(X)


def build_base(name, seed):
    if name == "iforest": return IForest(seed)
    if name == "pca": return PCAResidual()
    if name == "ar": return ARResidual()
    if name == "mlp_ae": return MLPAE(seed=seed)
    if name == "invariant": return Inv()
    if name == "invariant_nodiff": return Inv(use_diff=False)
    if name == "lstm_ae":
        from src.deep import LSTMAE; return LSTMAE(seed=seed)
    raise ValueError(name)


def get_data(args, seed):
    seed = seed + getattr(args, "seed_offset", 0)
    if args.hai:
        tr, te = load_hai(args.hai), load_hai(args.hai_test)
        return tr[tr["label"] == 0].reset_index(drop=True), te, None
    df, meta = simulate(n=args.n, seed=seed, n_events=args.n_events); h = meta["split"]
    assert df["label"].iloc[:h].sum() == 0, "attack leaked into training half"
    sh = lambda x: x - h
    m = dict(events=[dict(e, start=sh(e["start"]), end=sh(e["end"])) for e in meta["events"]],
             mode_changes=[sh(c) for c in meta["mode_changes"] if c >= h],
             dropouts=[(sh(a), sh(b)) for a, b in meta["dropouts"] if a >= h])
    return df.iloc[:h].reset_index(drop=True), df.iloc[h:].reset_index(drop=True), m


def run_seed(args, cfg, seed, keep=False):
    train, test, meta = get_data(args, seed)
    cols = [c for c in feature_cols(train) if train[c].std() > 0]
    n = len(train); i1, i2 = int(.7 * n), int(.8 * n)       # train | early-stop | calibration  (temporal)
    Xtr, Xes, Xcal = (train[cols].values[a:b] for a, b in [(0, i1), (i1, i2), (i2, n)])
    Xte, yte = test[cols].values, test["label"].values
    want = cfg["models"]; need = set(b for b in BASE if b in want)
    for h in HYBRIDS:
        if h in want: need |= set(HYBRIDS[h][0])
    for w_ in want:
        d_ = dyn(w_)
        if d_: need |= {"invariant"} | ({d_[1]} if d_[1] else set())
    fitted, t_fit = {}, {}
    for b in need:
        t0 = time.time(); fitted[b] = build_base(b, seed).fit(Xtr, Xes); t_fit[b] = time.time() - t0
    models = {b: fitted[b] for b in need if b in want or b in ()}
    for b in need: models[b] = fitted[b]
    for h in HYBRIDS:
        if h in want:
            parts, lams = HYBRIDS[h]; models[h] = HybridScore([fitted[p] for p in parts], lams).calibrate(Xcal)
    gi = cols.index("SP") if "SP" in cols else None
    for w_ in want:
        d_ = dyn(w_)
        if d_ and gi is not None:
            g = GatedScore(fitted["invariant"], gi, d_[2]).calibrate(Xcal)
            models[w_] = g if d_[0] == "gate" else HybridScore([fitted[d_[1]], g], [1, 1]).calibrate(Xcal)
    out, saved = {}, {}
    ev_meta = meta["events"] if meta else []
    day = len(yte) / 86400.0
    W = cfg["mode_change_window_s"]
    for name in [m for m in want if m in models]:
        mdl = models[name]; s_cal, s_te = mdl.score(Xcal), mdl.score(Xte); r = {}
        for fp in cfg["fp_rates"]:
            al = smooth_alarms(s_te > threshold_from_normal(s_cal, fp), cfg["debounce"]); r[f"fp{fp}"] = evaluate(yte, al)
        ab = smooth_alarms(s_te > threshold_for_alarm_budget(s_cal, cfg.get("alarm_budget_per_day", 5), cfg["debounce"]), cfg["debounce"])
        r["alarm_budget"] = evaluate(yte, ab)
        prim = smooth_alarms(s_te > threshold_from_normal(s_cal, cfg["primary_fp"]), cfg["debounce"])
        qs = np.quantile(s_cal, 1 - np.geomspace(0.3, 1e-6, 50)); ext = np.geomspace(max(s_cal.max(), 1e-9), max(s_te.max(), s_cal.max() * 1.01), 20)
        fa, rc = tradeoff_curve(yte, s_te, np.unique(np.r_[qs, ext]), cfg["debounce"])
        r["curve"] = dict(fa=fa.tolist(), rc=rc.tolist()); r["budget"] = {str(b): recall_at_budget(fa, rc, b) for b in cfg["budgets_fa_per_day"]}
        r["fit_s"] = t_fit.get(name, 0.0)
        if meta:
            det = {d["start"]: d for d in event_detections(yte, prim)}
            r["per_type"] = {t: float(np.mean([det[e["start"]]["detected"] for e in ev_meta if e["type"] == t] or [np.nan])) for t in ATTACK_TYPES}
            fas = false_alarm_runs(yte, prim)
            near = sum(any(0 <= a - mc <= W for mc in meta["mode_changes"]) for a, b in fas)
            drop = sum(any(a < d1 + 5 and b > d0 - 5 for d0, d1 in meta["dropouts"]) for a, b in fas)
            r["false_alarms"] = dict(total=len(fas), near_mode_change=near, near_dropout=drop)
            P = mdl.per_sensor(Xte)
            if P is not None:
                norm = PerSensorNormalizer().fit(mdl.per_sensor(Xcal)); h, byt = attribution_hits(norm(P), ev_meta, cols)
                r["attr_hit1"], r["attr_hit3"], r["attr_by_type_hit3"] = h[1], h[3], byt
        out[name] = r
        if keep: saved[name] = dict(score=s_te, thr=threshold_from_normal(s_cal, cfg["primary_fp"]))
    extra = dict(cols=cols, n_test=len(yte), mode_change_time_frac=(min(1.0, len(meta["mode_changes"]) * W / len(yte)) if meta else None))
    if "invariant" in fitted: extra["invariants"] = fitted["invariant"].m.describe(cols)
    if "mlp_ae" in fitted: extra["mlp_history"] = fitted["mlp_ae"].history
    if "lstm_ae" in fitted: extra["lstm_history"] = fitted["lstm_ae"].history
    if keep: extra["scores"] = saved; extra["y"] = yte; extra["meta"] = meta
    return out, extra


def aggregate(runs, cfg, outdir):
    names = list(runs[0].keys()); pf = f"fp{cfg['primary_fp']}"; rows = []
    for m in names:
        g = lambda f: np.array([f(r[m]) for r in runs], float)
        row = dict(model=m)
        for k in ["event_recall", "event_precision", "event_f1", "median_delay_s", "fa_per_day", "pa_f1_inflated"]:
            v = g(lambda r: r[pf][k]); row[k] = np.nanmean(v); row[k + "_std"] = np.nanstd(v)
            lo, hi = bootstrap_ci(v); row[k + "_ci_lo"], row[k + "_ci_hi"] = lo, hi
        for k in ["event_recall", "event_f1", "fa_per_day"]:
            row["ab_" + k] = g(lambda r: r["alarm_budget"][k]).mean()
        for b in cfg["budgets_fa_per_day"]:
            v = g(lambda r: r["budget"][str(b)]); row[f"recall@{b}FA/day"] = v.mean(); row[f"recall@{b}FA/day_std"] = v.std()
        if "attr_hit1" in runs[0][m]:
            row["attr_hit1"] = g(lambda r: r["attr_hit1"]).mean(); row["attr_hit3"] = g(lambda r: r["attr_hit3"]).mean()
        if "false_alarms" in runs[0][m]:
            tot = g(lambda r: r["false_alarms"]["total"]).sum(); near = g(lambda r: r["false_alarms"]["near_mode_change"]).sum()
            row["fa_total"] = tot; row["fa_frac_near_mode_change"] = near / tot if tot else np.nan
        rows.append(row)
    summ = pd.DataFrame(rows); summ.to_csv(f"{outdir}/summary.csv", index=False)
    if "per_type" in runs[0][names[0]]:
        pt = pd.DataFrame({m: {t: np.nanmean([r[m]["per_type"][t] for r in runs]) for t in ATTACK_TYPES} for m in names}).T
        pt.to_csv(f"{outdir}/per_attack_type_recall.csv")
    sig = []
    pairs = [("hybrid_mlp_inv", "mlp_ae"), ("hybrid_mlp_inv", "invariant"), ("hybrid_ar_inv", "ar"), ("hybrid_ar_inv", "invariant"), ("hybrid_mlp_inv", "hybrid_mlp_ar")]
    for a, b in pairs:
        if a in names and b in names and len(runs) >= 5:
            for metric, f in [("event_f1", lambda r, m: r[m][pf]["event_f1"]), (f"recall@{cfg['budgets_fa_per_day'][0]}FA/day", lambda r, m: r[m]["budget"][str(cfg['budgets_fa_per_day'][0])])]:
                res = paired_wilcoxon([f(r, a) for r in runs], [f(r, b) for r in runs]); sig.append(dict(a=a, b=b, metric=metric, **res))
    pd.DataFrame(sig).to_csv(f"{outdir}/significance_wilcoxon.csv", index=False)
    return summ, sig


def make_figures(runs, extras, cfg, outdir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(f"{outdir}/figures", exist_ok=True); names = list(runs[0].keys())
    grid = np.linspace(0, 20, 81); plt.figure(figsize=(7, 4.5))
    for m in names:
        ys = [np.interp(grid, np.array(r[m]["curve"]["fa"]), np.array(r[m]["curve"]["rc"])) for r in runs]
        plt.plot(grid, np.mean(ys, 0), label=m, lw=2.2 if m.startswith("hybrid_mlp_inv") and "lam" not in m else 1.2)
    plt.xlabel("false alarms / day (oracle threshold sweep)"); plt.ylabel("event recall"); plt.title("Event recall vs false-alarm budget (mean over seeds)")
    plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{outdir}/figures/tradeoff_curves.png", dpi=150); plt.close()
    if "per_type" in runs[0][names[0]]:
        sel = [m for m in ["pca", "ar", "mlp_ae", "invariant", "hybrid_mlp_inv"] if m in names]; x = np.arange(len(ATTACK_TYPES)); w = .8 / len(sel)
        plt.figure(figsize=(8, 4))
        for i, m in enumerate(sel):
            plt.bar(x + i * w, [np.nanmean([r[m]["per_type"][t] for r in runs]) for t in ATTACK_TYPES], w, label=m)
        plt.xticks(x + .4 - w / 2, ATTACK_TYPES, rotation=15); plt.ylabel("event recall @ primary threshold"); plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(f"{outdir}/figures/per_attack_type.png", dpi=150); plt.close()
    ex = extras[0]
    if "scores" in ex and ex["meta"]:
        sel = [m for m in ["mlp_ae", "invariant", "hybrid_mlp_inv"] if m in ex["scores"]]
        ev = ex["meta"]["events"]; lo, hi = max(0, ev[0]["start"] - 6000), min(len(ex["y"]), ev[3]["end"] + 3000)
        fig, ax = plt.subplots(len(sel), 1, figsize=(10, 2.2 * len(sel)), sharex=True)
        for a, m in zip(np.atleast_1d(ax), sel):
            s = ex["scores"][m]["score"]; a.plot(np.arange(lo, hi), s[lo:hi], lw=.6); a.axhline(ex["scores"][m]["thr"], color="r", ls="--", lw=.8)
            a.set_ylabel(m, fontsize=8); a.set_yscale("log")
            for e in ev:
                if e["end"] > lo and e["start"] < hi: a.axvspan(e["start"], e["end"], color="orange", alpha=.3)
            for c in ex["meta"]["mode_changes"]:
                if lo < c < hi: a.axvline(c, color="g", ls=":", lw=1)
        plt.suptitle("Scores on test stream: orange = attack, green dotted = benign mode change, red = threshold", fontsize=9)
        plt.tight_layout(); plt.savefig(f"{outdir}/figures/score_timeline.png", dpi=150); plt.close()
    h = extras[0].get("mlp_history")
    if h:
        plt.figure(figsize=(5, 3.5)); plt.plot([e["train_mse"] for e in h], label="train"); plt.plot([e["val_mse"] for e in h], label="early-stop val")
        plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.tight_layout(); plt.savefig(f"{outdir}/figures/mlp_ae_training.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--seeds", type=int); ap.add_argument("--n", type=int); ap.add_argument("--n_events", type=int)
    ap.add_argument("--models"); ap.add_argument("--hai"); ap.add_argument("--hai_test"); ap.add_argument("--out", default="results")
    args = ap.parse_args(); cfg = json.load(open(args.config))
    if args.seeds: cfg["seeds"] = args.seeds
    if args.models: cfg["models"] = args.models.split(",")
    args.n = args.n or cfg["n"]; args.n_events = args.n_events or cfg["n_events"]; os.makedirs(args.out, exist_ok=True)
    runs, extras = [], []
    for s in range(cfg["seeds"]):
        t0 = time.time(); r, ex = run_seed(args, cfg, s, keep=(s == 0)); runs.append(r); extras.append(ex)
        print(f"seed {s} done in {time.time()-t0:.0f}s", flush=True)
    summ, sig = aggregate(runs, cfg, args.out); make_figures(runs, extras, cfg, args.out)
    json.dump(dict(cfg=cfg, runs=runs, invariants_seed0=extras[0].get("invariants"), mlp_history_seed0=extras[0].get("mlp_history"),
                   lstm_history_seed0=extras[0].get("lstm_history")), open(f"{args.out}/raw_runs.json", "w"), default=float)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    show = ["model", "event_recall", "event_precision", "event_f1", "fa_per_day", "median_delay_s"] + [c for c in summ.columns if c.startswith("recall@") and not c.endswith("std")] + ["ab_event_recall", "ab_fa_per_day", "ab_event_f1", "attr_hit3", "fa_frac_near_mode_change", "pa_f1_inflated"]
    print(summ[[c for c in show if c in summ.columns]].round(3).to_string(index=False))
    if sig: print(pd.DataFrame(sig).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
