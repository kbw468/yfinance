"""Monotone calibration with a minimum-support constraint: sort by x, form consecutive bins of >= min_n cases,
fit isotonic regression on the bin means (weighted by n), predict with clipping to the fitted range.
No probability can rest on fewer than min_n historical cases."""
import numpy as np
from sklearn.isotonic import IsotonicRegression


class BinnedIsotonic:
    def __init__(self, min_n=500):
        self.min_n = min_n

    def fit(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        ok = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[ok], y[ok]
        order = np.argsort(x); x, y = x[order], y[order]
        nb = max(1, len(x) // self.min_n)
        edges = np.linspace(0, len(x), nb + 1).astype(int)
        bx = np.array([x[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
        by = np.array([y[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
        bw = np.diff(edges).astype(float)
        self.iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(bx, by, sample_weight=bw)
        self.n_bins = nb
        return self

    def predict(self, x):
        return self.iso.predict(np.asarray(x, float))
