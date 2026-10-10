"""The V-recovered leader screen on the latest session, plus the setup's realised record.
Setup (all same-day universe percentiles, tie-safe 20% cuts):
  1. deep drawdown inside the last 126 sessions: bottom 20% of the universe (mdd_126)
  2. smooth climb over the last 63 sessions: top 20% for least depth below its running high (mean_dd_63)
  3. leadership, any one of: 126-session return vs its sector top 20%, RS line vs its 52-week high top 20%, 252-session up-capture top 20%
Discovered 2015-2020, confirmed 2021-2023 (largest confirmation lift among rules with >= 800 confirmation cases), held out 2024+.
Energy, Basic Materials and Utilities are excluded: there the setup failed in every window (16% / 10% / 2% superior in
2015-2020 / 2021-2023 / 2024+, vs 33% / 31% / 33% elsewhere), already visible in the discovery window.
Writes results/ra21/screen_latest.csv and screen_record.csv."""
import json
import numpy as np
import pandas as pd
from ..data_io import load_universe
from .common import load, MARKET, OUT
from .rules import conditions

LEGS = ["mdd_126", "mean_dd_63", "xs_sec_126", "rs_off_high_252", "up_capture_252"]
EXCLUDED_SECTORS = ["Energy", "Basic Materials", "Utilities"]
SHOW = ["off_high_252", "days_since_high_252", "roc_5", "roc_21", "roc_63", "roc_126", "beta_l1_252", "up_volume_share_63", "eff_63", "new_high63_share_63",
        "volume_roc_21_252", "vol_roc_21_252"]


def flag(T: pd.DataFrame, cuts) -> pd.DataFrame:
    C = conditions(T, LEGS, cuts)
    T["leg_sector"] = C["xs_sec_126:TOP"]; T["leg_rs"] = C["rs_off_high_252:TOP"]; T["leg_upcap"] = C["up_capture_252:TOP"]
    T["vrl_any_sector"] = C["mdd_126:BOT"] & C["mean_dd_63:TOP"] & (T.leg_sector | T.leg_rs | T.leg_upcap)
    T["vrl"] = T.vrl_any_sector & ~T.sector.isin(EXCLUDED_SECTORS)
    return T


def main():
    cuts = json.load(open(OUT / "rules_meta.json"))["breadth_cuts"]
    T = flag(load(LEGS + SHOW + MARKET + ["sector", "sup_21", "sup_42", "xs_21", "stopped_21", "dd_21"]), cuts)
    uni = load_universe(); asof = T.date.max()
    X = T[T.vrl]
    rec = X.groupby(X.date.dt.year).agg(firings=("sup_21", "size"), labelled=("sup_21", "count"), names=("ticker", "nunique"), superior_21=("sup_21", "mean"),
                                        superior_42=("sup_42", "mean"), median_excess_21=("xs_21", "median"), stopped=("stopped_21", "mean"))
    base = T.groupby(T.date.dt.year).sup_21.mean(); rec["base"] = base; rec["lift"] = rec.superior_21 / rec.base
    rec.round(4).to_csv(OUT / "screen_record.csv")
    now = T[(T.date == asof) & T.vrl].copy()
    now["legs"] = [", ".join(n for n, v in (("6m vs sector", a), ("RS line at high", b), ("up-capture", c)) if v) for a, b, c in zip(now.leg_sector, now.leg_rs, now.leg_upcap)]
    pr = {c: T[T.date == asof][c].rank(pct=True) for c in LEGS}
    g = T[T.date == asof].assign(p_dd=pr["mdd_126"], p_climb=pr["mean_dd_63"], p_lead=np.maximum.reduce([pr["xs_sec_126"], pr["rs_off_high_252"], pr["up_capture_252"]]))
    near = g[~g.vrl_any_sector & ~g.sector.isin(EXCLUDED_SECTORS) & (g.p_dd <= 0.30) & (g.p_climb >= 0.70) & (g.p_lead >= 0.70)].copy(); near["legs"] = "near miss (each leg within 10 points)"
    excl = T[(T.date == asof) & T.vrl_any_sector & T.sector.isin(EXCLUDED_SECTORS)].copy(); excl["legs"] = "fires, excluded sector"
    out = pd.concat([now, near, excl])
    out["company"] = out.ticker.map(uni.Company); out["sector"] = out.ticker.map(uni.Sector)
    cols = ["ticker", "company", "sector", "legs", "mdd_126", "mean_dd_63", "xs_sec_126"] + SHOW
    out[cols].round(4).to_csv(OUT / "screen_latest.csv", index=False)
    print(f"{asof.date()}: {len(now)} firing, {len(near)} near misses"); print(out[cols].round(3).to_string(index=False)); print(rec.round(3).to_string())


if __name__ == "__main__":
    main()
