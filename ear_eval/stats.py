"""Significance testing: paired Wilcoxon signed-rank + bootstrap CI on per-question scores."""
import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


def paired_wilcoxon(a, b):
    """Two-sided paired Wilcoxon signed-rank test on per-question scores a (EAR) vs b (baseline)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = int(len(a))
    d = a - b
    out = {"n": n, "mean_diff": float(np.mean(d)) if n else 0.0,
           "median_diff": float(np.median(d)) if n else 0.0, "stat": None, "p": 1.0}
    if n == 0 or np.allclose(d, 0) or wilcoxon is None:
        return out
    try:
        stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        out["stat"], out["p"] = float(stat), float(p)
    except ValueError:
        pass
    return out


def bootstrap_diff(a, b, n_boot=10000, seed=0):
    """Bootstrap 95% CI for the mean paired difference (a - b)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    if len(d) == 0:
        return {"mean_diff": 0.0, "ci95": [0.0, 0.0], "p_two_sided": 1.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    frac_gt0 = float((means > 0).mean())
    p_two = 2 * min(frac_gt0, 1 - frac_gt0)
    return {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
            "p_two_sided": float(min(1.0, p_two))}


def summarize_metric(values):
    v = np.asarray(values, float)
    if len(v) == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def cohens_d_paired(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return float(d.mean() / sd) if sd else 0.0
