import numpy as np
from scipy import stats as st


def bootstrap_ci(x, n_boot=5000, seed=0, level=0.95):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2: return (np.nan, np.nan)
    rng = np.random.default_rng(seed); m = rng.choice(x, (n_boot, len(x))).mean(1)
    return tuple(np.quantile(m, [(1 - level) / 2, 1 - (1 - level) / 2]))


def paired_wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b): return dict(p=1.0, mean_diff=0.0)
    return dict(p=float(st.wilcoxon(a, b).pvalue), mean_diff=float(np.mean(a - b)))
