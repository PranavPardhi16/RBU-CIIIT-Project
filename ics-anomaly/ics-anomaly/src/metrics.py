"""Event-wise evaluation for ICS anomaly detection (PA-F1 reported only for contrast)."""
from __future__ import annotations
import numpy as np


def events(label):
    lab = np.asarray(label).astype(int); d = np.diff(np.concatenate([[0], lab, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def threshold_from_normal(s, fp_rate=1e-3): return float(np.quantile(s, 1 - fp_rate))


def smooth_alarms(raw, k=5):
    a = np.asarray(raw).astype(int)
    if k <= 1: return a
    return (np.convolve(a, np.ones(k, dtype=int), mode="full")[: len(a)] >= k).astype(int)


def event_detections(label, alarm, tolerance=0):
    out = []
    for (s, e) in events(label):
        lo = max(0, s - tolerance); hit = np.where(alarm[lo:e] == 1)[0]
        out.append(dict(start=int(s), end=int(e), detected=bool(len(hit)), delay=float(max(0, lo + hit[0] - s)) if len(hit) else np.nan))
    return out


def false_alarm_runs(label, alarm, tolerance=0):
    ev = events(label)
    return [(a, b) for (a, b) in events(alarm) if not any(a < e + tolerance and b > s - tolerance for (s, e) in ev)]


def evaluate(label, alarm, period_s=1.0, tolerance=0):
    label = np.asarray(label).astype(int); alarm = np.asarray(alarm).astype(int)
    det = event_detections(label, alarm, tolerance); fa = false_alarm_runs(label, alarm, tolerance); n_al = len(events(alarm))
    nd = sum(d["detected"] for d in det); delays = [d["delay"] for d in det if d["detected"]]
    days = len(label) * period_s / 86400.0
    rec = nd / max(len(det), 1); prec = (n_al - len(fa)) / max(n_al, 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(n_events=len(det), events_detected=nd, event_recall=rec, event_precision=prec, event_f1=f1,
                median_delay_s=float(np.median(delays)) if delays else np.nan,
                fa_per_day=len(fa) / max(days, 1e-9), pa_f1_inflated=point_adjusted_f1(label, alarm))


def point_adjusted_f1(label, alarm):
    adj = alarm.copy()
    for (s, e) in events(label):
        if adj[s:e].any(): adj[s:e] = 1
    tp = int(((adj == 1) & (label == 1)).sum()); fp = int(((adj == 1) & (label == 0)).sum()); fn = int(((adj == 0) & (label == 1)).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return 2 * p * r / (p + r) if p + r else 0.0


def tradeoff_curve(label, score, thresholds, debounce=5, period_s=1.0):
    """Event recall vs false alarms/day over thresholds (ORACLE operating points: swept on test)."""
    fa, rc = [], []
    for t in thresholds:
        r = evaluate(label, smooth_alarms(score > t, debounce), period_s); fa.append(r["fa_per_day"]); rc.append(r["event_recall"])
    o = np.argsort(fa); return np.array(fa)[o], np.array(rc)[o]


def recall_at_budget(fa, rc, budget):
    ok = np.array(fa) <= budget; return float(np.max(np.array(rc)[ok])) if ok.any() else 0.0


def threshold_for_alarm_budget(score_normal, budget_per_day, debounce=5, period_s=1.0):
    """Smallest threshold whose attack-free calibration alarm RUNS/day <= budget (operator-centric:
    controls incidents per day rather than raw sample exceedance rate)."""
    days = len(score_normal) * period_s / 86400.0
    for q in np.r_[np.linspace(0.90, 0.999, 60), np.linspace(0.999, 1.0, 80)]:
        t = float(np.quantile(score_normal, q))
        if len(events(smooth_alarms(score_normal > t, debounce))) / days <= budget_per_day:
            return t
    return float(score_normal.max())
