"""Choose gate window W on VALIDATION simulations (seeds 100-102), disjoint from evaluation seeds 0-9."""
import argparse, json, numpy as np
from run_experiments import run_seed
Ws = [0, 120, 300, 600, 900, 1500]
cfg = json.load(open("configs/full.json")); cfg["seeds"] = 3
cfg["models"] = ["ar", "invariant"] + [f"inv_gate{w}" for w in Ws if w] + [f"hybrid_ar_invgate{w}" for w in Ws if w] + [f"hybrid_mlp_invgate{w}" for w in Ws if w]
args = argparse.Namespace(hai=None, hai_test=None, n=150000, n_events=12, seed_offset=100)
runs = [run_seed(args, cfg, s)[0] for s in range(3)]
pf = f"fp{cfg['primary_fp']}"; res = {}
for m in cfg["models"]:
    res[m] = dict(event_f1=float(np.mean([r[m][pf]["event_f1"] for r in runs])), recall=float(np.mean([r[m][pf]["event_recall"] for r in runs])),
                  fa_day=float(np.mean([r[m][pf]["fa_per_day"] for r in runs])), ab_f1=float(np.mean([r[m]["alarm_budget"]["event_f1"] for r in runs])))
    print(f"{m:24s} {res[m]}")
best = max(Ws[1:], key=lambda w: res[f"hybrid_ar_invgate{w}"]["ab_f1"] + res[f"hybrid_ar_invgate{w}"]["event_f1"])
res["chosen_W"] = best; json.dump(res, open("results/gate_tuning.json", "w"), indent=1); print("chosen W =", best, "(by mean of event_f1 + alarm-budget event_f1 of hybrid_ar_invgate, validation seeds only)")
