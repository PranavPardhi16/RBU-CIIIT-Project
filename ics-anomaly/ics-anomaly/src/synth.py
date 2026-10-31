"""Two-tank process simulator with PI control, benign nuisances and labelled attacks.

Physics (dt = 1 s):  tank1 <- pump q_in (PI controlled to setpoint, first-order actuator lag)
                     tank1 -> tank2 via f12 = c1*sqrt(h1);  tank2 -> demand f2 = c2*sqrt(h2)*u2
`u2` (consumer valve opening) is NOT logged, so benign demand changes break any naive
level-vs-flow relation: that is the "benign operating-mode change" false-alarm trap.

Benign nuisances: setpoint steps, unlogged demand steps (test half extrapolates beyond the
training range), diurnal temperature + random walk (distractor sensor), sensor noise,
short sample-and-hold dropouts (labelled normal).

Attacks (second half only; each carries ground-truth "affected logged sensors" for
attribution scoring). Sensor spoofs act on the value the PLC sees, so the controller
reacts and the physics diverge, as in a real incident:
  spoof_bias         constant bias on LIT101
  ramp_stealth       slow ramp bias on LIT101 (small final magnitude)
  flow_tamper        logged FIT102 offset (physics unchanged; breaks mass balance)
  actuator_override  pump forced to a fixed flow while logged P101 still shows the command
  replay_mask        valve partly closed physically (u2*0.6) while LIT102 & FIT201 replay
                     old data (hard: replayed pair stays mutually consistent)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

SENSORS = ["LIT101", "LIT102", "FIT101", "FIT102", "FIT201", "SP", "P101", "PIT101", "TT101"]
ATTACK_TYPES = ["spoof_bias", "ramp_stealth", "flow_tamper", "actuator_override", "replay_mask"]
ATTACK_SENSORS = {
    "spoof_bias": ["LIT101"], "ramp_stealth": ["LIT101"], "flow_tamper": ["FIT102"],
    "actuator_override": ["P101", "FIT101"], "replay_mask": ["LIT102", "FIT201"],
}
C1, C2, A1, A2 = 0.8, 1.0, 150.0, 150.0


def _piecewise(rng, n, pool_first, pool_second, half, dmin, dmax):
    out = np.zeros(n); changes = []; t = 0
    cur = rng.choice(pool_first)
    while t < n:
        d = int(rng.integers(dmin, dmax)); out[t:t + d] = cur; t += d
        pool = pool_first if t < half else pool_second
        nxt = rng.choice(pool)
        while nxt == cur:
            nxt = rng.choice(pool)
        cur = nxt
        if t < n:
            changes.append(t)
    return out, changes


def simulate(n=150000, seed=0, n_events=12, attack_start_frac=0.5, dropout_rate=1 / 3000, burn=3000):
    rng = np.random.default_rng(seed)
    N = n + burn; half = burn + int(n * attack_start_frac)
    sp, sp_ch = _piecewise(rng, N, np.arange(42, 59, 2.0), np.arange(38, 63, 2.0), half, 3000, 8000)
    u2, u2_ch = _piecewise(rng, N, [0.85, 1.0, 1.15], [0.75, 0.85, 1.0, 1.15, 1.3], half, 4000, 9000)

    bias = np.zeros(N); flow_off = np.zeros(N); u2_mult = np.ones(N)
    over_q = np.full(N, np.nan); replay = []; events = []
    slot = (N - half) // n_events
    types = list(rng.permutation([ATTACK_TYPES[k % 5] for k in range(n_events)]))
    for k, typ in enumerate(types):
        dur = int(rng.integers(300, 900))
        s = half + k * slot + int(rng.integers(500, slot - dur - 500)); e = s + dur
        sign = rng.choice([-1.0, 1.0]); mag = 0.0
        if typ == "spoof_bias":
            mag = rng.uniform(2.5, 6.0); bias[s:e] = sign * mag
        elif typ == "ramp_stealth":
            mag = rng.uniform(0.8, 2.0); bias[s:e] = sign * np.linspace(0, mag, dur)
        elif typ == "flow_tamper":
            mag = rng.uniform(0.6, 1.5); flow_off[s:e] = sign * mag
        elif typ == "actuator_override":
            mag = rng.choice([rng.uniform(2.0, 3.5), rng.uniform(9.0, 12.0)]); over_q[s:e] = mag
        else:
            mag = 0.6; u2_mult[s:e] = 0.6; replay.append((s, e))
        events.append(dict(type=str(typ), start=s, end=e, mag=float(mag), sensors=ATTACK_SENSORS[typ]))

    nz = {k: rng.normal(0, v, N) for k, v in dict(l1=.12, l2=.12, f1=.10, f2=.10, f3=.10, p=.01, pit=.02).items()}
    h1 = 50.0; q = C1 * np.sqrt(h1); h2 = (C1 * np.sqrt(h1) / (C2 * u2[0])) ** 2; I = 0.0
    Kp, Ki = 1.2, 0.002
    log = {k: np.zeros(N) for k in SENSORS}
    for i in range(N):
        h1m = h1 + nz["l1"][i] + bias[i]
        err = sp[i] - h1m
        I = np.clip(I + Ki * err, -8, 8)
        qcmd = np.clip(C1 * np.sqrt(sp[i]) + Kp * err + I, 0, 15)
        q += 0.3 * (qcmd - q)
        qp = over_q[i] if not np.isnan(over_q[i]) else q
        f12 = C1 * np.sqrt(max(h1, 0)); f2 = C2 * np.sqrt(max(h2, 0)) * u2[i] * u2_mult[i]
        log["LIT101"][i] = h1m; log["LIT102"][i] = h2 + nz["l2"][i]
        log["FIT101"][i] = qp + nz["f1"][i]; log["FIT102"][i] = f12 + flow_off[i] + nz["f3"][i]
        log["FIT201"][i] = f2 + nz["f2"][i]; log["SP"][i] = sp[i]
        log["P101"][i] = qcmd / 15 + nz["p"][i]; log["PIT101"][i] = 1.0 + 0.02 * h1 + 0.01 * qp + nz["pit"][i]
        h1 += (qp - f12) / A1; h2 += (f12 - f2) / A2
    t = np.arange(N)
    log["TT101"] = 25 + 3 * np.sin(2 * np.pi * t / 40000) + np.cumsum(rng.normal(0, 0.01, N)) * 0.2

    old = {k: v.copy() for k, v in log.items()}
    for (s, e) in replay:                                   # replay old normal data on two logged tags
        for tag in ("LIT102", "FIT201"):
            log[tag][s:e] = old[tag][s - 5000: e - 5000]
    drops = []
    for _ in range(rng.poisson(N * dropout_rate)):          # benign sample-and-hold dropouts
        s = int(rng.integers(burn + 10, N - 50)); d = int(rng.integers(5, 25))
        tag = rng.choice([c for c in SENSORS if c != "SP"])
        log[tag][s:s + d] = log[tag][s - 1]; drops.append((s, s + d))

    df = pd.DataFrame(log); df["label"] = 0
    for ev in events:
        df.loc[ev["start"]:ev["end"] - 1, "label"] = 1
    df = df.iloc[burn:].reset_index(drop=True)
    sh = lambda x: x - burn
    meta = dict(
        events=[dict(ev, start=sh(ev["start"]), end=sh(ev["end"])) for ev in events],
        mode_changes=[sh(c) for c in sp_ch + u2_ch if c > burn],
        dropouts=[(sh(a), sh(b)) for a, b in drops if a > burn], split=half - burn)
    return df, meta
