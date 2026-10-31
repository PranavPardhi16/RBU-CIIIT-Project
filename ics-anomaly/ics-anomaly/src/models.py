"""Anomaly scorers. API: fit(Xtr, Xes=None); score(X)->(N,); per_sensor(X)->(N,d) or None.
Xes = attack-free early-stopping / selection block (temporally after Xtr)."""
from __future__ import annotations
import copy
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class IForest:
    def __init__(self, seed=0): self.seed = seed
    def fit(self, X, Xes=None):
        self.sc = StandardScaler().fit(X)
        self.m = IsolationForest(n_estimators=200, random_state=self.seed, n_jobs=-1).fit(self.sc.transform(X)); return self
    def score(self, X): return -self.m.score_samples(self.sc.transform(X))
    def per_sensor(self, X): return None


class PCAResidual:
    def __init__(self, var=0.95): self.var = var
    def fit(self, X, Xes=None):
        self.sc = StandardScaler().fit(X); self.p = PCA(n_components=self.var).fit(self.sc.transform(X)); return self
    def per_sensor(self, X):
        Z = self.sc.transform(X); return (Z - self.p.inverse_transform(self.p.transform(Z))) ** 2
    def score(self, X): return self.per_sensor(X).sum(1)


class ARResidual:
    """One-step-ahead Ridge predictor from L lags of all sensors (strong linear CPS baseline)."""
    def __init__(self, lags=5, alpha=1.0): self.L, self.alpha = lags, alpha
    def _lagmat(self, Z): return np.hstack([Z[self.L - k: len(Z) - k] for k in range(1, self.L + 1)])
    def fit(self, X, Xes=None):
        self.sc = StandardScaler().fit(X); Z = self.sc.transform(X)
        self.m = Ridge(alpha=self.alpha).fit(self._lagmat(Z), Z[self.L:])
        ref = Z if Xes is None else self.sc.transform(Xes)
        self.rs = (ref[self.L:] - self.m.predict(self._lagmat(ref))).std(0) + 1e-8; return self
    def per_sensor(self, X):
        Z = self.sc.transform(X); R = np.zeros_like(Z)
        R[self.L:] = ((Z[self.L:] - self.m.predict(self._lagmat(Z))) / self.rs) ** 2; return R
    def score(self, X): return self.per_sensor(X).sum(1)


class MLPAE:
    """Denoising MLP autoencoder on [standardised level, lagged difference] features.
    Rigor: seeded, input-noise regularisation, temporal early stopping on Xes, best-weights restore."""
    def __init__(self, hidden=(32, 8, 32), epochs=60, patience=6, lr=1e-3, batch=256, noise=0.1, dlag=5, seed=0):
        self.h, self.epochs, self.patience, self.lr, self.batch, self.noise, self.dlag, self.seed = hidden, epochs, patience, lr, batch, noise, dlag, seed
        self.history = []
    def _feat(self, X):
        Z = (X - self.mu) / self.sd; L = self.dlag
        D = np.vstack([np.zeros((L, Z.shape[1])), Z[L:] - Z[:-L]]) / self.dsd
        return np.hstack([Z, D])
    def fit(self, X, Xes=None):
        if Xes is None:
            cut = int(len(X) * 0.85); X, Xes = X[:cut], X[cut:]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8; self.dsd = np.ones(X.shape[1])
        Z = (X - self.mu) / self.sd; self.dsd = (Z[self.dlag:] - Z[:-self.dlag]).std(0) + 1e-8
        F, Fe = self._feat(X), self._feat(Xes)
        rng = np.random.default_rng(self.seed)
        self.net = MLPRegressor(hidden_layer_sizes=self.h, activation="relu", solver="adam", batch_size=self.batch, learning_rate_init=self.lr,
                                alpha=1e-4, random_state=self.seed)
        best, best_net, bad = np.inf, None, 0
        for ep in range(self.epochs):
            p = rng.permutation(len(F))
            self.net.partial_fit(F[p] + rng.normal(0, self.noise, F.shape), F[p])
            tr = float(((self.net.predict(F[::5]) - F[::5]) ** 2).mean()); va = float(((self.net.predict(Fe) - Fe) ** 2).mean())
            self.history.append(dict(epoch=ep + 1, train_mse=tr, val_mse=va))
            if va < best - 1e-5: best, best_net, bad = va, copy.deepcopy(self.net), 0
            else:
                bad += 1
                if bad >= self.patience: break
        self.net = best_net; self.best_val = best
        self.err_ref = ((self.net.predict(F[::3]) - F[::3]) ** 2).mean(0) + 1e-8; return self
    def per_sensor(self, X):
        F = self._feat(X); E = (self.net.predict(F) - F) ** 2 / self.err_ref
        d = X.shape[1]; return E[:, :d] + E[:, d:]
    def score(self, X): return self.per_sensor(X).sum(1)


class HybridScore:
    """Fuse components: score = max_i lam_i * (s_i - med_i)/(q99_i - med_i), statistics from the
    attack-free calibration block (so a normal-data 99th percentile maps to ~1).
    Central hypothesis: invariants cut false alarms on benign mode changes; learned
    reconstruction covers what linear invariants miss."""
    def __init__(self, parts, lams=None): self.parts, self.lams = parts, lams or [1.0] * len(parts)
    def calibrate(self, Xcal):
        self.st, self.ps_q = [], []
        for p in self.parts:
            s = p.score(Xcal); med = np.median(s); self.st.append((med, np.quantile(s, 0.99) - med + 1e-9))
            P = p.per_sensor(Xcal); self.ps_q.append(None if P is None else np.quantile(P, 0.99, axis=0) + 1e-9)
        return self
    def score(self, X):
        return np.max([lam * (p.score(X) - m) / q for p, lam, (m, q) in zip(self.parts, self.lams, self.st)], axis=0)
    def per_sensor(self, X):
        return np.max([lam * p.per_sensor(X) / q for p, lam, q in zip(self.parts, self.lams, self.ps_q)], axis=0)


class GatedScore:
    """Transient-aware gate. After a logged setpoint/command change (detected on column `gate_idx`)
    the inner scorer's evidence is neutralised for W seconds: control lag temporarily violates
    level-vs-setpoint relations, which is NOT an attack. Trade-off: attacks inside the window can
    only be caught by the other component(s). W is chosen on separate validation simulations."""
    def __init__(self, inner, gate_idx, W): self.inner, self.g, self.W = inner, gate_idx, W
    def calibrate(self, Xcal):
        self.fill = float(np.median(self.inner.score(Xcal))); self.thr = 0.05 * Xcal[:, self.g].std() + 1e-9; return self
    def _mask(self, X):
        ch = np.abs(np.diff(X[:, self.g], prepend=X[0, self.g])) > self.thr
        i = np.arange(len(X)); last = np.maximum.accumulate(np.where(ch, i, -10**9)); return (i - last) <= self.W
    def fit(self, X, Xes=None): return self
    def score(self, X):
        s = self.inner.score(X).copy(); s[self._mask(X)] = self.fill; return s
    def per_sensor(self, X):
        P = self.inner.per_sensor(X).copy(); P[self._mask(X)] = 0.0; return P
