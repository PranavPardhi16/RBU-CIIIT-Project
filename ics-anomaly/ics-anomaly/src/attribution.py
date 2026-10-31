"""Per-tag attribution scoring: does the detector point the operator at the right tag?"""
from __future__ import annotations
import numpy as np


class PerSensorNormalizer:
    def fit(self, P_cal): self.q = np.quantile(P_cal, 0.99, axis=0) + 1e-9; return self
    def __call__(self, P): return P / self.q


def attribution_hits(P_norm, events, cols, ks=(1, 3)):
    """For each labelled event rank tags by mean normalised per-sensor score over the event.
    hit@k = any ground-truth affected tag in the top-k. Returns {k: rate} and per-type rates."""
    hits = {k: [] for k in ks}; by_type = {}
    for ev in events:
        m = P_norm[ev["start"]:ev["end"]].mean(0); order = [cols[i] for i in np.argsort(-m)]
        for k in ks:
            h = any(t in order[:k] for t in ev["sensors"]); hits[k].append(h)
            if k == 3: by_type.setdefault(ev["type"], []).append(h)
    return {k: float(np.mean(v)) if v else np.nan for k, v in hits.items()}, {t: float(np.mean(v)) for t, v in by_type.items()}
