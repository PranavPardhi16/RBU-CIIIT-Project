"""Invariant mining: sparse linear relations among logged tags, learned on attack-free data.

Each candidate  x_j ~ sum_k w_k x_k  (Lasso, other sensors only) is fit on the TRAIN block and
kept only if its R^2 on a HELD-OUT attack-free block (temporal, later in time) exceeds min_r2.
The residual std used to normalise scores also comes from the held-out block, so benign
drift between train and held-out is priced in. Optional lagged-difference features
(x_t - x_{t-L}) allow conservation-style relations (d level ~ net flow).
Per-sensor attribution: residual of invariant i is credited to its target and, with weight
|w|/max|w|, to its regressors.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Lasso


class InvariantMiner:
    def __init__(self, alpha=1e-3, min_r2=0.85, use_diff=True, diff_lag=30, stride=2):
        self.alpha, self.min_r2, self.use_diff, self.diff_lag, self.stride = alpha, min_r2, use_diff, diff_lag, stride

    def _design(self, X):
        Xs = (X - self.mu_) / self.sd_
        if not self.use_diff:
            return Xs
        L = self.diff_lag
        D = np.vstack([np.zeros((L, X.shape[1])), Xs[L:] - Xs[:-L]]) / self.dsd_
        return np.hstack([Xs, D])

    def fit(self, X, Xval=None):
        if Xval is None:
            cut = int(len(X) * 0.8); X, Xval = X[:cut], X[cut:]
        self.d_ = X.shape[1]
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-8
        self.dsd_ = np.ones(self.d_)
        if self.use_diff:
            Xs = (X - self.mu_) / self.sd_
            self.dsd_ = (Xs[self.diff_lag:] - Xs[:-self.diff_lag]).std(0) + 1e-8
        Z, Zv = self._design(X), self._design(Xval)
        d = self.d_; self.inv_ = []
        for j in range(Z.shape[1]):
            idx = [k for k in range(Z.shape[1]) if k % d != j % d]
            m = Lasso(alpha=self.alpha, max_iter=3000).fit(Z[::self.stride, idx], Z[::self.stride, j])
            pred_v = m.predict(Zv[:, idx]); res_v = Zv[:, j] - pred_v
            r2v = 1 - res_v.var() / (Zv[:, j].var() + 1e-12)
            if r2v >= self.min_r2:
                w = np.zeros(d)
                for k, c in zip(idx, m.coef_):
                    w[k % d] = max(w[k % d], abs(c))
                w = w / (w.max() + 1e-12); w[j % d] = 1.0
                self.inv_.append(dict(j=j, m=m, idx=idx, s=res_v.std() + 1e-8, r2=float(r2v), w=w))
        return self

    def residuals(self, X):
        Z = self._design(X)
        R = np.zeros((len(X), len(self.inv_)))
        for i, v in enumerate(self.inv_):
            R[:, i] = np.abs(Z[:, v["j"]] - v["m"].predict(Z[:, v["idx"]])) / v["s"]
        return R

    def score(self, X):
        R = self.residuals(X)
        return R.max(1) if R.shape[1] else np.zeros(len(X))

    def per_sensor(self, X):
        R = self.residuals(X); P = np.zeros((len(X), self.d_))
        for i, v in enumerate(self.inv_):
            P = np.maximum(P, R[:, [i]] * v["w"][None, :])
        return P

    def describe(self, cols):
        return [dict(target=("d_" if v["j"] >= self.d_ else "") + cols[v["j"] % self.d_], r2_heldout=round(v["r2"], 3),
                     inputs=[cols[k] for k in np.where(v["w"] > 0.05)[0] if k != v["j"] % self.d_]) for v in self.inv_]
