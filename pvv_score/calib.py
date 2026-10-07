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

    # ---- serialisation: the fitted curve is a piecewise-linear interpolation between its thresholds, clipped at the ends
    def to_dict(self) -> dict:
        return {"x": [float(v) for v in self.iso.X_thresholds_], "y": [float(v) for v in self.iso.y_thresholds_],
                "n_bins": int(self.n_bins), "min_n": int(self.min_n)}

    @classmethod
    def from_dict(cls, d: dict) -> "FrozenCurve":
        return FrozenCurve(np.asarray(d["x"], float), np.asarray(d["y"], float), d.get("n_bins"), d.get("min_n"))


class FrozenCurve:
    """A BinnedIsotonic curve restored from its thresholds; predicts exactly as the sklearn object did."""
    def __init__(self, x, y, n_bins=None, min_n=None):
        self.x, self.y, self.n_bins, self.min_n = x, y, n_bins, min_n

    def predict(self, x):
        x = np.asarray(x, float)
        if len(self.x) == 1:
            return np.full(x.shape, self.y[0])
        return np.interp(x, self.x, self.y)      # np.interp clips to the end values, as out_of_bounds="clip" does

    def to_dict(self) -> dict:
        return {"x": self.x.tolist(), "y": self.y.tolist(), "n_bins": self.n_bins, "min_n": self.min_n}
