"""Data loading. HAI loader + synthetic ICS generator for pipeline testing.

HAI: download from https://github.com/icsdataset/hai (files are large; may need git-lfs).
Column names differ between HAI versions (timestamp column 'time' or 'timestamp';
label column 'attack' or 'Attack'). The loader auto-detects these; CHECK the result
against the version's readme before trusting it.
"""
from __future__ import annotations
import glob
import numpy as np
import pandas as pd

TIME_CANDIDATES = ["time", "timestamp", "Timestamp", "Time"]
LABEL_CANDIDATES = ["attack", "Attack", "label", "Label"]


def _find(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def load_hai(path_glob: str) -> pd.DataFrame:
    """Load and concatenate HAI csv files matching a glob, in sorted order."""
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"No files match {path_glob}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    t = _find(df.columns, TIME_CANDIDATES)
    y = _find(df.columns, LABEL_CANDIDATES)
    if y is None:
        raise ValueError(f"No label column found among {LABEL_CANDIDATES}; got {list(df.columns)[:10]}...")
    out = df.copy()
    if t is not None:
        out[t] = pd.to_datetime(out[t])
        out = out.sort_values(t).reset_index(drop=True)
    out = out.rename(columns={y: "label"})
    # HAI files may include other attack-type label columns; drop non-sensor extras
    drop = [c for c in out.columns if c.lower().startswith("attack_") or c == t]
    out = out.drop(columns=drop, errors="ignore")
    out["label"] = (out["label"] > 0).astype(int)
    return out


def make_synthetic_ics(n=20000, n_events=8, seed=0):
    """Toy water-tank-like CPS: level = integral(inflow - outflow), inflow follows a
    setpoint controller (control coupling), plus benign mode changes and attacks
    (sensor spoof, stealthy drift, tampered flow reading). Pipeline smoke-test only."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    setpoint = 50 + 10 * (np.floor(t / 4000) % 2)            # benign mode change
    level = np.zeros(n); inflow = np.zeros(n); outflow = np.zeros(n)
    level[0] = 50
    label = np.zeros(n, dtype=int)
    for i in range(1, n):
        inflow[i] = np.clip(0.5 * (setpoint[i] - level[i - 1]) + 5 + rng.normal(0, .1), 0, 12)
        outflow[i] = 5 + rng.normal(0, .1)
        level[i] = level[i - 1] + 0.1 * (inflow[i] - outflow[i])
    lvl_meas = level + rng.normal(0, .05, n)
    inflow_m = inflow.copy(); outflow_m = outflow.copy()
    pump = (inflow > 5).astype(float) + rng.normal(0, .02, n)
    starts = np.sort(rng.choice(np.arange(6000, n - 600, 500), size=n_events, replace=False))
    for k, s in enumerate(starts):
        e = s + int(rng.integers(150, 400)); label[s:e] = 1
        kind = k % 3
        if kind == 0:   lvl_meas[s:e] += 3.0
        elif kind == 1: lvl_meas[s:e] += np.linspace(0, 2.5, e - s)
        else:           inflow_m[s:e] += 3.0
    return pd.DataFrame({"LIT101": lvl_meas, "FIT101": inflow_m, "FIT201": outflow_m,
                         "P101": pump, "SP": setpoint + rng.normal(0, .02, n), "label": label})


def split_train_test(df, train_frac=0.5):
    """Temporal split; train keeps only attack-free rows (unsupervised setting)."""
    cut = int(len(df) * train_frac)
    train = df.iloc[:cut]; train = train[train["label"] == 0]
    test = df.iloc[cut:]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def feature_cols(df):
    return [c for c in df.columns if c != "label" and pd.api.types.is_numeric_dtype(df[c])]
