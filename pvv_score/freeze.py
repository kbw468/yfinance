"""Freeze the fitted model into results/model/ so nightly scoring refits nothing.

  weights.json          current test fold's beta-bucket weights, state and level layers (from the walk-forward by-fold files)
  signatures.csv        confirmed signatures with their confirmation probability and lift (ordered by discovery probability)
  calib_composite.json  composite probability curves per beta bucket   (written by buylist)
  calib_signature.json  signature probability curves per composite quintile (written by signature_rule)
  factor_registry.csv   factor -> family, for the page chips
  model.json            provenance: fold year, sample window, counts, file hashes, time frozen
Run after buylist: python -m pvv_score.freeze"""
import hashlib
import json
import shutil
import datetime as dt
import pandas as pd
from .config import CACHE_DIR, RESULTS_DIR, SAMPLE_START, EXCLUDE_PERIODS
from .signatures import DISCOVER_END, CONFIRM_START

MODEL = RESULTS_DIR / "model"


def _weights(layer: str) -> tuple[int, dict]:
    w = pd.read_csv(RESULTS_DIR / f"composite_{layer}_smooth_weights_by_fold.csv")
    y = int(w.year.max())
    cur = w[(w.year == y) & w.bucket.isin(["low", "mid", "high"])]
    return y, {b: dict(zip(g.factor, g.weight.astype(float))) for b, g in cur.groupby("bucket")}


def main():
    MODEL.mkdir(exist_ok=True)
    ys, ws = _weights("state"); yl, wl = _weights("level")
    assert ys == yl, (ys, yl)
    json.dump({"fold_year": ys, "state": ws, "level": wl}, open(MODEL / "weights.json", "w"), indent=1)
    sig = pd.read_csv(RESULTS_DIR / "signatures_all.csv")
    conf = sig[sig.confirmed].sort_values("p_disc", ascending=False).reset_index(drop=True)
    conf[["signature", "k", "n_disc", "p_disc", "lift_disc", "n_conf", "p_conf", "lift_conf"]].to_csv(MODEL / "signatures.csv", index=False)
    shutil.copy(CACHE_DIR / "factor_registry.csv", MODEL / "factor_registry.csv")
    for f in ["calib_composite.json", "calib_signature.json"]:
        assert (MODEL / f).exists(), f"{f} missing: run signature_rule and buylist first"
    asof = pd.read_csv(RESULTS_DIR / "universe_scores_smooth.csv")["asof"].iloc[0]
    files = ["weights.json", "signatures.csv", "calib_composite.json", "calib_signature.json", "factor_registry.csv"]
    h = hashlib.sha256()
    for f in files:
        h.update((MODEL / f).read_bytes())
    meta = {"frozen_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"), "fitted_through": str(asof), "fold_year": ys,
            "sample_start": SAMPLE_START, "excluded": EXCLUDE_PERIODS, "discovery_end": DISCOVER_END, "confirm_start": CONFIRM_START,
            "n_signatures": int(len(conf)), "weights_k": {l: {b: len(w) for b, w in d.items()} for l, d in [("state", ws), ("level", wl)]},
            "files": files, "sha256": h.hexdigest()[:16]}
    json.dump(meta, open(MODEL / "model.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
