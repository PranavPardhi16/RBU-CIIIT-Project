"""ConformGuard engine.

Training (sklearn, offline): fit an AR residual model and sparse linear invariants on attack-free data.
Inference (numpy only): batch `analyze` and streaming `process_one` share the SAME code path
(`Frozen.components`, `Guard._step_*`), and a unit test checks they agree.

Method (three coupled mechanisms, each ablated in run_conformguard.py)
 1. Regime-conditional (Mondrian) conformal p-values. A "regime" is derived from logged command tags
    (setpoint-like columns that change rarely): TRANSIENT for W seconds after a command change, else STEADY.
    W is estimated from attack-free calibration data (settling time of the fused score), not hand-set.
    p-values are computed against the calibration scores of the SAME regime, so one alpha gives the
    same exceedance rate in both regimes instead of flooding alarms after mode changes.
 2. Quarantined rolling calibration. Calibration pools are updated online with scores that were
    delayed by `delay` seconds and whose surrounding +/-delay window contained no suspicious p-value
    (p <= accept_alpha). Tracks benign drift while preventing slow attacks from being absorbed.
 3. Regime-frozen CUSUM bank on signed, normalised invariant residuals. Accumulators hold their value
    during transients, and detect small persistent shifts (stealthy ramps) that one-step residuals miss.
Alarm budget: the operator specifies alarm RUNS per day; alpha and the CUSUM threshold h are mapped from the
budget using attack-free pool replay.
"""
from __future__ import annotations
import bisect, json
from collections import deque
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------------- helpers

def _events(a):
    a = np.asarray(a).astype(int); d = np.diff(np.concatenate([[0], a, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def smooth(raw, k):
    a = np.asarray(raw).astype(int)
    if k <= 1: return a
    return (np.convolve(a, np.ones(k, dtype=int), mode="full")[: len(a)] >= k).astype(int)


def detect_command_cols(X, cols, max_change_frac=0.005):
    """Auto-detect setpoint/command-like tags: non-constant, but change in <0.5% of samples."""
    out = []
    for j, c in enumerate(cols):
        d = np.abs(np.diff(X[:, j])); frac = (d > 1e-9).mean()
        if X[:, j].std() > 0 and frac < max_change_frac: out.append(c)
    return out


# ----------------------------------------------------------------------------- frozen scorers

@dataclass
class Frozen:
    cols: list
    mu: np.ndarray; sd: np.ndarray
    L: int; ar_coef: np.ndarray; ar_int: np.ndarray; ar_rs: np.ndarray
    diff_lag: int; dsd: np.ndarray
    inv_B: np.ndarray; inv_b: np.ndarray; inv_s: np.ndarray; inv_j: np.ndarray; inv_w: np.ndarray; inv_r2: np.ndarray

    @property
    def H(self): return max(self.L, self.diff_lag)

    def components(self, X):
        """Vectorised: returns s_ar (N,), PA (N,d) per-sensor AR z^2, R (N,m) signed normalised invariant residuals."""
        N, d = X.shape; H = self.H; Z = (X - self.mu) / self.sd
        PA = np.zeros((N, d)); R = np.zeros((N, len(self.inv_j)))
        if N <= H: return PA.sum(1), PA, R
        t = np.arange(H, N)
        lags = np.hstack([Z[t - k] for k in range(1, self.L + 1)])
        PA[H:] = ((Z[t] - (lags @ self.ar_coef.T + self.ar_int)) / self.ar_rs) ** 2
        if len(self.inv_j):
            D = np.zeros_like(Z); D[self.diff_lag:] = (Z[self.diff_lag:] - Z[:-self.diff_lag]) / self.dsd
            ZF = np.hstack([Z, D]); R[H:] = (ZF[t][:, self.inv_j.astype(int)].reshape(len(t), -1) - (ZF[t] @ self.inv_B.T + self.inv_b)) / self.inv_s
        return PA.sum(1), PA, R


def fit_frozen(Xtr, Xes, cols, L=5, diff_lag=30, lasso_alpha=1e-3, min_r2=0.85, stride=2, ridge_alpha=1.0):
    from sklearn.linear_model import Ridge, Lasso
    d = Xtr.shape[1]; mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Z, Ze = (Xtr - mu) / sd, (Xes - mu) / sd
    lag = lambda A: np.hstack([A[L - k: len(A) - k] for k in range(1, L + 1)])
    ar = Ridge(alpha=ridge_alpha).fit(lag(Z), Z[L:])
    rs = (Ze[L:] - ar.predict(lag(Ze))).std(0) + 1e-8
    dsd = (Z[diff_lag:] - Z[:-diff_lag]).std(0) + 1e-8
    def design(A):
        D = np.zeros_like(A); D[diff_lag:] = (A[diff_lag:] - A[:-diff_lag]) / dsd; return np.hstack([A, D])
    ZF, ZFe = design(Z), design(Ze); B, b, s, js, ws, r2s = [], [], [], [], [], []
    for j in range(2 * d):
        idx = [k for k in range(2 * d) if k % d != j % d]
        m = Lasso(alpha=lasso_alpha, max_iter=3000).fit(ZF[::stride, idx], ZF[::stride, j])
        res = ZFe[:, j] - m.predict(ZFe[:, idx]); r2 = 1 - res.var() / (ZFe[:, j].var() + 1e-12)
        if r2 >= min_r2:
            row = np.zeros(2 * d); row[idx] = m.coef_; w = np.zeros(d)
            for k, c in zip(idx, m.coef_): w[k % d] = max(w[k % d], abs(c))
            w = w / (w.max() + 1e-12); w[j % d] = 1.0
            B.append(row); b.append(m.intercept_); s.append(res.std() + 1e-8); js.append(j); ws.append(w); r2s.append(r2)
    m_ = len(js)
    return Frozen(list(cols), mu, sd, L, ar.coef_, ar.intercept_, rs, diff_lag, dsd,
                  np.array(B).reshape(m_, 2 * d), np.array(b), np.array(s), np.array(js, dtype=float), np.array(ws).reshape(m_, d), np.array(r2s))


# ----------------------------------------------------------------------------- guard

class Guard:
    def __init__(self, frozen: Frozen, command_cols=None, mondrian=True, rolling=False, cusum=True, components=("ar",),
                 debounce=5, cusum_k=0.375, cusum_cap=300.0, cusum_clip=3.0, rebaseline=120, delay=300, accept_alpha=0.01, cusum_share=0.5, budget=5.0):
        self.f, self.mondrian, self.rolling, self.cusum, self.comps = frozen, mondrian, rolling, cusum, tuple(components)
        self.k, self.ck, self.ccap, self.clip, self.rb, self.D, self.aq, self.share, self.budget = debounce, cusum_k, cusum_cap, cusum_clip, rebaseline, delay, accept_alpha, cusum_share, budget
        self.command_cols = command_cols or []; self.cidx = [frozen.cols.index(c) for c in self.command_cols]
        self.reset()

    def _comp(self, X):
        """Components with commands treated as exogenous inputs: not scored (AR) and not invariant targets."""
        sa, PA, R = self.f.components(X)
        if self.cidx:
            PA = PA.copy(); PA[..., self.cidx] = 0.0; sa = PA.sum(-1)
            bad = [i for i, j in enumerate(self.f.inv_j.astype(int)) if (j % len(self.f.cols)) in self.cidx]
            if bad and R.size: R = R.copy(); R[..., bad] = 0.0
        return sa, PA, R

    # ---- calibration -------------------------------------------------------------
    def calibrate(self, Xpool, budget=None, Xenv=None):
        f = self.f; self.budget = budget or self.budget
        Xe = Xpool if Xenv is None else Xenv
        self.env_lo = np.array([Xe[:, j].min() for j in self.cidx]); self.env_hi = np.array([Xe[:, j].max() for j in self.cidx])
        sa, PA, R = self._comp(Xpool); v = np.arange(len(Xpool)) >= f.H
        self.thr = np.array([0.05 * Xpool[:, j].std() + 1e-9 for j in self.cidx])
        self.norm = {}
        sinv = np.abs(R).max(1) if R.shape[1] else np.zeros(len(Xpool))
        for name, s in (("ar", sa), ("inv", sinv)):
            med = np.median(s[v]); self.norm[name] = (med, np.quantile(s[v], 0.99) - med + 1e-9)
        self.q_ar = np.quantile(PA[v], 0.99, axis=0) + 1e-9
        inv_ps = self._inv_per_sensor(R); self.q_inv = np.quantile(inv_ps[v], 0.99, axis=0) + 1e-9 if R.shape[1] else np.ones(PA.shape[1])
        fused = self._fuse(sa, np.abs(R).max(1) if R.shape[1] else np.zeros(len(sa)))
        ch = self._changes(Xpool); last = self._since(ch)
        self.W = self._estimate_W(fused, last, v) if (self.mondrian and self.cidx) else 0
        reg = self._regime_from_since(last)
        b = reg if self.mondrian else np.zeros(len(reg), int)
        self.pool0 = [np.sort(fused[v & (b == k)]) for k in (0, 1)]
        self.pool_sizes = [len(p) for p in self.pool0]
        # pool p-values (static, in-sample) -> alpha(budget); CUSUM trace -> h(budget)
        pp = np.ones(len(fused))
        for k in (0, 1):
            m = v & (b == k)
            if m.sum(): pp[m] = np.searchsorted(self.pool0[k], fused[m], side="left"); pp[m] = (len(self.pool0[k]) - pp[m]) / (len(self.pool0[k]) + 1)
        self._pool_p, self._pool_days = pp, len(Xpool) / 86400.0
        # CUSUM bank: only near-white invariants (lag-1 autocorr < 0.8); centre/scale on pool (CUSUM assumes ~iid)
        Rv = R[v]
        with np.errstate(all='ignore'): ac = np.array([np.corrcoef(Rv[1:, i], Rv[:-1, i])[0, 1] for i in range(R.shape[1])]) if R.shape[1] else np.array([])
        mu_all = Rv.mean(0) if R.shape[1] else np.zeros(0); sd_all = (Rv.std(0) + 1e-8) if R.shape[1] else np.ones(0)
        stable = np.zeros(R.shape[1], bool)
        if R.shape[1]:
            st_rows = Rv[(reg[v] == 0)]; nb = len(st_rows) // 1000
            if nb >= 3:
                Rcl = np.clip((st_rows[:nb * 1000] - mu_all) / sd_all, -self.clip, self.clip).reshape(nb, 1000, -1).mean(1)
                stable = np.abs(Rcl).max(0) <= 0.6 * self.ck          # benign bias wander must stay below the CUSUM allowance
        self.cs_idx = np.where((ac < 0.8) & stable)[0] if R.shape[1] else np.array([], int)
        self.cs_mu = mu_all[self.cs_idx]; self.cs_sd = sd_all[self.cs_idx]
        self.cs_ac = ac
        self._pool_stat = self._cusum_trace(R, reg, v)
        self.set_budget(self.budget); self.reset(); return self

    def set_budget(self, B):
        self.budget = B; Bc = B * (1 - self.share) if self.cusum else B; Bs = B * self.share
        v = np.arange(len(self._pool_p)) >= self.f.H; best = 1e-7
        for a in np.geomspace(1e-7, 0.3, 90):
            if len(_events(smooth(v & (self._pool_p <= a), self.k))) / self._pool_days <= Bc: best = a
        self.alpha = best; self.h = np.inf
        if self.cusum and self._pool_stat.size:
            for h in np.linspace(1, 200, 200):
                if len(_events(v & (self._pool_stat > h))) / self._pool_days <= Bs: self.h = h; break
            else: self.h = 200.0
        return self

    def _estimate_W(self, fused, last, v):
        n = len(fused); far = v & ((last < 0) | (last > 3000))
        if far.sum() < 100: return 900
        q = np.quantile(fused[far], 0.95); chg = np.where(np.diff(last, prepend=-1) == 0)[0]  # indices where since==0 => change
        chg = np.where(last == 0)[0]; settles = []
        for c in chg:
            seg = fused[c:c + 3000]
            if len(seg) < 200: continue
            ok = np.lib.stride_tricks.sliding_window_view(seg, 60).max(1) <= q
            settles.append(int(np.argmax(ok)) if ok.any() else 3000)
        return int(np.clip(np.percentile(settles, 90), 120, 3000)) if len(settles) >= 3 else 900

    # ---- scoring pieces ----------------------------------------------------------
    def _fuse(self, sa, si):
        parts = []
        if "ar" in self.comps: parts.append((sa - self.norm["ar"][0]) / self.norm["ar"][1])
        if "inv" in self.comps: parts.append((si - self.norm["inv"][0]) / self.norm["inv"][1])
        return np.max(parts, axis=0)

    def _inv_per_sensor(self, R):
        P = np.zeros((len(R), len(self.f.cols)))
        for i in range(R.shape[1]): P = np.maximum(P, np.abs(R[:, [i]]) * self.f.inv_w[i][None, :])
        return P

    def _changes(self, X):
        if not self.cidx: return np.zeros(len(X), bool)
        ch = np.zeros(len(X), bool)
        for j, t in zip(self.cidx, self.thr): ch[1:] |= np.abs(np.diff(X[:, j])) > t
        return ch

    @staticmethod
    def _since(ch):
        i = np.arange(len(ch)); last = np.maximum.accumulate(np.where(ch, i, -10 ** 9)); out = i - last; out[last < 0] = -1
        out[(i - last) > 10 ** 8] = -1; return out

    def _regime_from_since(self, since):
        return ((since >= 0) & (since <= self.W)).astype(int) if (self.mondrian and self.cidx) else np.zeros(len(since), int)

    def _cs_in(self, R): return np.clip((R[..., self.cs_idx] - self.cs_mu) / self.cs_sd, -self.clip, self.clip)   # robust (Huber-style) input

    def _cusum_trace(self, R, reg, valid):
        """Robust two-sided CUSUM bank, frozen in transients; at each transient->steady transition the
        reference is re-baselined to the mean residual of the last `rb` transient samples (operating-point shift)."""
        N = len(R); m = len(self.cs_idx)
        if m == 0: return np.zeros(N)
        Rc = self._cs_in(R); Sp, Sn, out = np.zeros(m), np.zeros(m), np.zeros(N)
        base = np.zeros(m); win = deque(maxlen=self.rb); prev = 0
        for t in range(N):
            if not valid[t]: continue
            if reg[t] == 1:
                win.append(Rc[t])
            else:
                if prev == 1 and len(win): base = np.mean(win, axis=0); win.clear(); Sp[:] = 0; Sn[:] = 0
                x = Rc[t] - base
                Sp = np.minimum(np.maximum(0, Sp + x - self.ck), self.ccap); Sn = np.minimum(np.maximum(0, Sn - x - self.ck), self.ccap)
            prev = reg[t]; out[t] = max(Sp.max(), Sn.max())
        return out

    # ---- batch analysis ----------------------------------------------------------
    def analyze(self, X):
        sa, PA, R = self._comp(X); N = len(X); v = np.arange(N) >= self.f.H
        si = np.abs(R).max(1) if R.shape[1] else np.zeros(N); fused = self._fuse(sa, si)
        since = self._since(self._changes(X)); reg = self._regime_from_since(since); b = reg if self.mondrian else np.zeros(N, int)
        p = np.ones(N)
        if not self.rolling:
            for k in (0, 1):
                m = v & (b == k)
                if m.sum() and self.pool_sizes[k]: i = np.searchsorted(self.pool0[k], fused[m], side="left"); p[m] = (1 + len(self.pool0[k]) - i) / (len(self.pool0[k]) + 1)
        else:
            sl = [list(map(float, self.pool0[0])), list(map(float, self.pool0[1]))]; dq = [deque(sl[0]), deque(sl[1])]; D = self.D
            for t in range(N):
                if not v[t]: continue
                s = sl[b[t]]; i = bisect.bisect_left(s, fused[t]) if s else 0; p[t] = (1 + len(s) - i) / (len(s) + 1)
                u = t - D
                if u >= self.f.H and p[max(0, u - D):t + 1].min() > self.aq:
                    bu = b[u]; val = float(fused[u]); bisect.insort(sl[bu], val); dq[bu].append(val)
                    if len(dq[bu]) > max(self.pool_sizes[bu], 1):
                        old = dq[bu].popleft(); del sl[bu][bisect.bisect_left(sl[bu], old)]
        stat = self._cusum_trace(R, reg, v) if self.cusum else np.zeros(N)
        Pn = np.zeros((N, len(self.f.cols)))
        if "ar" in self.comps: Pn = np.maximum(Pn, PA / self.q_ar)
        if "inv" in self.comps and R.shape[1]: Pn = np.maximum(Pn, self._inv_per_sensor(R) / self.q_inv)
        return dict(fused=fused, p=p, regime=reg, stat=stat, Pn=Pn, valid=v, env=self._env(X))

    def _env(self, X):
        if not self.cidx: return np.zeros(len(X), bool)
        m = 0.01 * (self.env_hi - self.env_lo) + 1e-9
        return np.any((X[:, self.cidx] < self.env_lo - m) | (X[:, self.cidx] > self.env_hi + m), axis=1)

    def decide(self, tr, budget=None):
        if budget is not None and budget != self.budget: self.set_budget(budget)
        conf = smooth(tr["valid"] & (tr["p"] <= self.alpha), self.k)
        cus = (tr["valid"] & (tr["stat"] > self.h)).astype(int) if self.cusum else np.zeros(len(conf), int)
        return dict(alarm=np.maximum(conf, cus), conf=conf, cusum=cus)

    # ---- streaming ---------------------------------------------------------------
    def reset(self):
        self._buf = deque(maxlen=self.f.H + 1); self._prev = None; self._since_n = -1; self._t = -1
        self._run = 0; self._hist = deque(maxlen=2 * self.D + 1)
        m = len(getattr(self, "cs_idx", [])); self._Sp, self._Sn = np.zeros(m), np.zeros(m); self._ema = np.zeros(len(self.f.cols)); self._base = np.zeros(m); self._win = deque(maxlen=self.rb); self._prev_reg = 0
        if hasattr(self, "pool0"):
            self._sl = [list(map(float, self.pool0[0])), list(map(float, self.pool0[1]))]; self._dq = [deque(self._sl[0]), deque(self._sl[1])]
        return self

    def clone(self):
        import copy
        g = copy.copy(self); g.reset(); return g

    def process_one(self, x):
        x = np.asarray(x, float); self._t += 1; t = self._t; f = self.f
        if self._prev is not None and self.cidx and np.any(np.abs(x[self.cidx] - self._prev[self.cidx]) > self.thr): self._since_n = 0
        elif self._since_n >= 0: self._since_n += 1
        self._prev = x; self._buf.append(x)
        reg = int(self._since_n >= 0 and self._since_n <= self.W) if (self.mondrian and self.cidx) else 0; b = reg if self.mondrian else 0
        out = dict(t=t, valid=False, regime=reg, p=1.0, score=0.0, alarm=False, conf_alarm=False, cusum_alarm=False, cusum_stat=0.0, top=[], alpha=self.alpha, h=self.h, envelope=bool(self._env(x[None, :])[0]))
        if len(self._buf) < f.H + 1 or t < f.H:
            self._hist.append((0.0, b, 1.0, False)); return out
        sa, PA, R = self._comp(np.array(self._buf)); sa, PA, R = sa[-1], PA[-1], R[-1]
        si = np.abs(R).max() if R.size else 0.0; fused = float(self._fuse(np.array([sa]), np.array([si]))[0])
        s = self._sl[b]; i = bisect.bisect_left(s, fused) if s else 0; p = (1 + len(s) - i) / (len(s) + 1)
        self._hist.append((fused, b, p, True))
        if self.rolling and len(self._hist) > self.D:
            hv = list(self._hist); fu, bu, pu, vu = hv[-1 - self.D]
            if vu and min(h_[2] for h_ in hv[max(0, len(hv) - 1 - 2 * self.D):]) > self.aq:
                bisect.insort(self._sl[bu], fu); self._dq[bu].append(fu)
                if len(self._dq[bu]) > max(self.pool_sizes[bu], 1): old = self._dq[bu].popleft(); del self._sl[bu][bisect.bisect_left(self._sl[bu], old)]
        if self.cusum and len(self.cs_idx):
            Rc = self._cs_in(R)
            if reg == 1: self._win.append(Rc)
            else:
                if self._prev_reg == 1 and len(self._win): self._base = np.mean(self._win, axis=0); self._win.clear(); self._Sp[:] = 0; self._Sn[:] = 0
                x = Rc - self._base
                self._Sp = np.minimum(np.maximum(0, self._Sp + x - self.ck), self.ccap); self._Sn = np.minimum(np.maximum(0, self._Sn - x - self.ck), self.ccap)
            self._prev_reg = reg
        stat = float(max(self._Sp.max(), self._Sn.max())) if (self.cusum and len(self.cs_idx)) else 0.0
        self._run = self._run + 1 if p <= self.alpha else 0
        conf = self._run >= self.k; cus = bool(self.cusum and stat > self.h)
        Pn = np.zeros(len(f.cols))
        if "ar" in self.comps: Pn = np.maximum(Pn, PA / self.q_ar)
        if "inv" in self.comps and R.size: Pn = np.maximum(Pn, self._inv_per_sensor(R[None, :])[0] / self.q_inv)
        self._ema = 0.95 * self._ema + 0.05 * Pn; order = np.argsort(-self._ema)[:3]
        out.update(valid=True, p=float(p), score=fused, alarm=bool(conf or cus), conf_alarm=bool(conf), cusum_alarm=cus, cusum_stat=stat,
                   top=[dict(tag=f.cols[j], score=round(float(self._ema[j]), 3)) for j in order])
        return out

    # ---- persistence (npz + json, no pickle) -------------------------------------
    def save(self, path):
        f = self.f; meta = dict(cols=f.cols, L=f.L, diff_lag=f.diff_lag, command_cols=self.command_cols, mondrian=self.mondrian, rolling=self.rolling,
                                cusum=self.cusum, comps=list(self.comps), k=self.k, ck=self.ck, ccap=self.ccap, clip=self.clip, rb=self.rb, D=self.D, aq=self.aq, share=self.share, budget=self.budget,
                                W=int(self.W), norm={k: list(map(float, v)) for k, v in self.norm.items()})
        np.savez_compressed(path, meta=json.dumps(meta), mu=f.mu, sd=f.sd, ar_coef=f.ar_coef, ar_int=f.ar_int, ar_rs=f.ar_rs, dsd=f.dsd, inv_B=f.inv_B, inv_b=f.inv_b,
                            inv_s=f.inv_s, inv_j=f.inv_j, inv_w=f.inv_w, inv_r2=f.inv_r2, thr=self.thr, q_ar=self.q_ar, q_inv=self.q_inv, pool0=self.pool0[0], pool1=self.pool0[1],
                            pool_p=self._pool_p, env_lo=self.env_lo, env_hi=self.env_hi, pool_stat=self._pool_stat, pool_days=np.array(self._pool_days), cs_idx=self.cs_idx, cs_mu=self.cs_mu, cs_sd=self.cs_sd, cs_ac=self.cs_ac)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False); m = json.loads(str(z["meta"]))
        fr = Frozen(m["cols"], z["mu"], z["sd"], m["L"], z["ar_coef"], z["ar_int"], z["ar_rs"], m["diff_lag"], z["dsd"], z["inv_B"], z["inv_b"], z["inv_s"], z["inv_j"], z["inv_w"], z["inv_r2"])
        g = cls.__new__(cls); g.f, g.mondrian, g.rolling, g.cusum, g.comps = fr, m["mondrian"], m["rolling"], m["cusum"], tuple(m["comps"])
        g.k, g.ck, g.ccap, g.clip, g.rb, g.D, g.aq, g.share, g.budget = m["k"], m["ck"], m["ccap"], m.get("clip", 3.0), m.get("rb", 120), m["D"], m["aq"], m["share"], m["budget"]
        g.command_cols = m["command_cols"]; g.cidx = [fr.cols.index(c) for c in g.command_cols]; g.thr = z["thr"]; g.W = m["W"]
        g.norm = {k: tuple(v) for k, v in m["norm"].items()}; g.q_ar, g.q_inv = z["q_ar"], z["q_inv"]
        g.pool0 = [z["pool0"], z["pool1"]]; g.pool_sizes = [len(p) for p in g.pool0]; g._pool_p, g._pool_stat, g._pool_days = z["pool_p"], z["pool_stat"], float(z["pool_days"])
        g.env_lo, g.env_hi = z["env_lo"], z["env_hi"]
        g.cs_idx, g.cs_mu, g.cs_sd, g.cs_ac = z["cs_idx"].astype(int), z["cs_mu"], z["cs_sd"], z["cs_ac"]
        g.set_budget(g.budget); g.reset(); return g
