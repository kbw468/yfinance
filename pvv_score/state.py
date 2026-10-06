"""State (identity-neutral) factor transforms.

z_f(t) = (f(t) - mean_{t-756..t-1} f) / std_{t-756..t-1} f   (min 252 observations, rolling 3-year norm)

A name that is chronically near its highs, chronically low-vol or chronically quiet scores ~0 on every z-factor.
Only behaviour that is unusual FOR THAT NAME carries information. The norm is rolling (not expanding) so it adapts
to the regime the mandate cares about.
"""
import numpy as np
import pandas as pd

NORM_WINDOW = 756
MIN_OBS = 252


def to_state(F: dict[str, pd.DataFrame], skip: tuple = ()) -> dict[str, pd.DataFrame]:
    out = {}
    for name, f in F.items():
        if name in skip:
            continue
        f = f.astype("float64")
        mu = f.rolling(NORM_WINDOW, min_periods=MIN_OBS).mean().shift(1)
        sd = f.rolling(NORM_WINDOW, min_periods=MIN_OBS).std().shift(1)
        z = (f - mu) / sd.where(sd > 1e-9)
        out[f"z_{name}"] = z.clip(-6, 6).astype("float32")
    return out
