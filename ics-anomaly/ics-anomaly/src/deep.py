"""PyTorch LSTM autoencoder with a rigorous training loop.
STATUS: written but NOT executed in the authoring sandbox (PyTorch unavailable there).
Smoke-test first:  python run_experiments.py --models lstm_ae --seeds 1 --n 40000
Training protocol: seeded; temporal early stopping on Xes; AdamW + cosine LR; grad clipping;
input-noise denoising; best-checkpoint restore; per-epoch history exposed in .history."""
from __future__ import annotations
import copy
import numpy as np
from sklearn.preprocessing import StandardScaler


def _windows(Z, w, stride=1):
    idx = np.arange(w)[None, :] + np.arange(0, len(Z) - w + 1, stride)[:, None]
    return Z[idx]


class LSTMAE:
    def __init__(self, window=30, hidden=64, layers=1, epochs=40, patience=6, lr=2e-3, wd=1e-4,
                 batch=256, noise=0.05, train_stride=2, clip=1.0, seed=0, device=None):
        self.__dict__.update(locals()); del self.__dict__["self"]; self.history = []

    def _net(self, d):
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(s, d, h, L):
                super().__init__()
                s.enc = nn.LSTM(d, h, L, batch_first=True); s.dec = nn.LSTM(h, h, L, batch_first=True); s.out = nn.Linear(h, d)
            def forward(s, x):
                _, (hn, _) = s.enc(x); z = hn[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
                y, _ = s.dec(z); return s.out(y)
        return Net(d, self.hidden, self.layers)

    def fit(self, X, Xes=None):
        import torch
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        if Xes is None:
            cut = int(len(X) * 0.85); X, Xes = X[:cut], X[cut:]
        self.dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sc = StandardScaler().fit(X)
        W = torch.tensor(_windows(self.sc.transform(X), self.window, self.train_stride), dtype=torch.float32)
        Wv = torch.tensor(_windows(self.sc.transform(Xes), self.window, 2), dtype=torch.float32).to(self.dev)
        self.net = self._net(X.shape[1]).to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        best, best_state, bad = np.inf, None, 0
        for ep in range(self.epochs):
            self.net.train(); perm = torch.randperm(len(W)); tot = 0.0
            for i in range(0, len(W), self.batch):
                b = W[perm[i:i + self.batch]].to(self.dev)
                loss = ((self.net(b + self.noise * torch.randn_like(b)) - b) ** 2).mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip); opt.step()
                tot += loss.item() * len(b)
            sched.step(); self.net.eval()
            with torch.no_grad():
                va = float(np.mean([((self.net(Wv[j:j + 1024]) - Wv[j:j + 1024]) ** 2).mean().item() for j in range(0, len(Wv), 1024)]))
            self.history.append(dict(epoch=ep + 1, train_mse=tot / len(W), val_mse=va, lr=sched.get_last_lr()[0]))
            if va < best - 1e-6: best, best_state, bad = va, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience: break
        self.net.load_state_dict(best_state); self.best_val = best; return self

    def per_sensor(self, X):
        import torch
        Z = self.sc.transform(X); W = torch.tensor(_windows(Z, self.window), dtype=torch.float32); out = []
        self.net.eval()
        with torch.no_grad():
            for i in range(0, len(W), 1024):
                b = W[i:i + 1024].to(self.dev); out.append(((self.net(b) - b) ** 2)[:, -1, :].cpu().numpy())
        E = np.vstack(out); return np.vstack([np.zeros((self.window - 1, E.shape[1])), E])

    def score(self, X): return self.per_sensor(X).sum(1)
