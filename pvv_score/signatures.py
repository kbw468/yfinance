"""Signature engine: conjunctions of threshold conditions, discovered then confirmed out of sample.

A condition = factor in its top or bottom quintile on the date (cross-sectional) or, for the ROC factors, a fixed
threshold. A signature = AND of 2 or 3 conditions. Outcome = the name's next-42-session path lands in the top
quartile of the universe on the smooth-climb score (Sharpe, max drawdown, straightness, up-day share).

Discovery window 2014-2021: keep signatures with >= MIN_N cases and lift >= MIN_LIFT (P / 0.25).
Confirmation window 2022-10 onward: keep only those whose lift is still >= CONFIRM_LIFT with >= MIN_N_CONFIRM cases.
Names today are ranked by the best confirmed signature they carry (its confirmation-window probability), then by how
many confirmed signatures fire. Nothing is averaged across signatures; a signature fires or it does not.
"""
import sys
import itertools
import numpy as np
import pandas as pd

from .config import CACHE_DIR, RESULTS_DIR, RECENT_START
from .run_eval import load_research

DISCOVER_END = "2021-12-31"
MIN_N, MIN_LIFT = 400, 1.20
MIN_N_CONFIRM, CONFIRM_LIFT = 150, 1.15

# curated factor set (price / volume / volatility; levels and states), quintile conditions on each
QUINTILE_FACTORS = [
    "mom_12_1", "ret_21", "ret_63", "rs_lead_126", "rs_spy_63", "dist_52w_high", "dist_52w_low", "base_depth_126", "days_since_20pct_dd",
    "pullback_21", "up_day_frac_63", "clv_21", "gap_share_21", "higher_low_cadence", "nr7_count_20", "range_comp_10_252", "bbw_pct_252",
    "vol_dry_20_250", "vol_dry_10_50", "updown_vol_ratio_50", "accum_dist_50", "obv_price_div_63", "net_vol_flow_63", "log_dvol_63", "dvol_trend_21_126",
    "ignition_mult_10", "days_since_ignition", "pullback_vol_21",
    "rv20_pct_252", "rv_ratio_5_20", "rv_ratio_20_60", "vol_of_vol_63", "dn_up_vol_asym_63", "dn_vol_63", "parkinson_cc_20", "skew_63", "idio_vol_63", "sharpe_126",
    "corr_spy_63", "beta_252", "up_capture_63", "capture_spread_63",
    "z_mom_12_1", "z_rs_lead_126", "z_vol_dry_20_250", "z_rv20_pct_252", "z_obv_price_div_63", "z_dist_52w_high", "z_log_dvol_63", "z_dn_up_vol_asym_63",
]
# fixed-threshold conditions (ROC family), name -> (column, op, value)
FIXED = {
    "volROC21>+50%": ("vol_roc_21", ">", 0.5), "volROC21>+25%": ("vol_roc_21", ">", 0.25), "volROC10>+50%": ("vol_roc_10", ">", 0.5),
    "rvROC21<-20%": ("rv20_roc_21", "<", -0.2), "rvROC21<-10%": ("rv20_roc_21", "<", -0.1), "rvROC10<-20%": ("rv20_roc_10", "<", -0.2),
    "rvROC21>+20%": ("rv20_roc_21", ">", 0.2), "px21>+3%": ("ret_21", ">", 0.03), "px21>+8%": ("ret_21", ">", 0.08), "px5<-3%": ("ret_5", "<", -0.03),
    "dvolROC21>+30%": ("dvol_roc_21", ">", 0.3), "volAccel5>0": ("vol_accel_5", ">", 0.0), "rvAccel5<0": ("rv_accel_5", "<", 0.0),
}


def smooth_top_quartile(df: pd.DataFrame, h: int = 42) -> pd.Series:
    parts = [df[f"fwd_sharpe_{h}"], -df[f"fwd_mdd_{h}"], df[f"fwd_r2_{h}"], df[f"fwd_up_{h}"]]
    u = sum(p.groupby(df.date).rank(pct=True) for p in parts) / 4
    return (u >= 0.75).where(u.notna())


def build_conditions(df: pd.DataFrame) -> dict:
    conds = {}
    rk = df.groupby("date")[QUINTILE_FACTORS].rank(pct=True)
    for f in QUINTILE_FACTORS:
        conds[f"{f}:TOP"] = (rk[f] >= 0.8).values
        conds[f"{f}:BOT"] = (rk[f] <= 0.2).values
    for name, (col, op, v) in FIXED.items():
        x = df[col].values
        conds[name] = (x > v) if op == ">" else (x < v)
    return {k: np.nan_to_num(v.astype(float), nan=0).astype(bool) for k, v in conds.items()}


def main():
    need = ["date", "ticker", "sector", "beta_252", "vol_roc_21", "vol_roc_10", "rv20_roc_21", "rv20_roc_10", "ret_21", "ret_5", "dvol_roc_21", "vol_accel_5", "rv_accel_5",
            "fwd_sharpe_42", "fwd_mdd_42", "fwd_r2_42", "fwd_up_42", "fwd_sharpe_63", "fwd_mdd_63", "fwd_r2_63", "fwd_up_63"] + QUINTILE_FACTORS
    df = load_research()[list(dict.fromkeys(need))].reset_index(drop=True)
    y42 = smooth_top_quartile(df, 42).values
    y63 = smooth_top_quartile(df, 63).values
    disc = (df.date <= DISCOVER_END).values & ~np.isnan(y42)
    conf = (df.date >= RECENT_START).values & ~np.isnan(y42)
    conds = build_conditions(df)
    names = list(conds)
    print(f"{len(names)} conditions, rows disc={disc.sum():,} conf={conf.sum():,}", file=sys.stderr)
    base_d, base_c = np.nanmean(y42[disc]), np.nanmean(y42[conf])
    y42d, y42c = np.nan_to_num(y42), np.nan_to_num(y42)

    def stats(mask):
        md, mc = mask & disc, mask & conf
        nd, nc = md.sum(), mc.sum()
        pd_ = y42d[md].mean() if nd else np.nan
        pc = y42c[mc].mean() if nc else np.nan
        return nd, pd_, nc, pc

    rows = []
    # pairs
    for a, b in itertools.combinations(names, 2):
        if a.split(":")[0] == b.split(":")[0]:
            continue
        m = conds[a] & conds[b]
        nd, pd_, nc, pc = stats(m)
        if nd >= MIN_N and pd_ / base_d >= MIN_LIFT:
            rows.append({"signature": f"{a} & {b}", "k": 2, "n_disc": nd, "p_disc": pd_, "lift_disc": pd_ / base_d, "n_conf": nc, "p_conf": pc, "lift_conf": pc / base_c if nc else np.nan})
    print(f"pairs passing discovery: {len(rows)}", file=sys.stderr)
    # triples: extend passing pairs by one more condition
    pair_pass = [r["signature"] for r in rows]
    seen = set()
    for sig in pair_pass:
        a, b = sig.split(" & ")
        for c in names:
            if c.split(":")[0] in (a.split(":")[0], b.split(":")[0]):
                continue
            key = tuple(sorted([a, b, c]))
            if key in seen:
                continue
            seen.add(key)
            m = conds[a] & conds[b] & conds[c]
            nd, pd_, nc, pc = stats(m)
            if nd >= MIN_N and pd_ / base_d >= MIN_LIFT + 0.1:
                rows.append({"signature": " & ".join(key), "k": 3, "n_disc": nd, "p_disc": pd_, "lift_disc": pd_ / base_d, "n_conf": nc, "p_conf": pc, "lift_conf": pc / base_c if nc else np.nan})
    res = pd.DataFrame(rows)
    res["confirmed"] = (res.n_conf >= MIN_N_CONFIRM) & (res.lift_conf >= CONFIRM_LIFT)
    # 63d check on confirmed
    res = res.sort_values(["confirmed", "p_conf"], ascending=False)
    res.to_csv(RESULTS_DIR / "signatures_all.csv", index=False)
    confirmed = res[res.confirmed].copy()
    print(f"discovered {len(res)}, confirmed {len(confirmed)}; base P(top-q smooth 42) disc={base_d:.3f} conf={base_c:.3f}", file=sys.stderr)

    # today's names: which confirmed signatures fire
    today_mask = (df.date == df.date.max()).values
    tickers = df.ticker.values[today_mask]
    fired = {t: [] for t in tickers}
    for _, r in confirmed.iterrows():
        parts = r.signature.split(" & ")
        m = np.ones(len(df), dtype=bool)
        for p in parts:
            m &= conds[p]
        for t in df.ticker.values[today_mask & m]:
            fired[t].append((r.p_conf, r.signature, int(r.n_conf)))
    out = []
    for t, lst in fired.items():
        lst.sort(reverse=True)
        out.append({"ticker": t, "n_signatures": len(lst), "best_p_top_q_42": lst[0][0] if lst else base_c, "best_signature": lst[0][1] if lst else "",
                    "best_signature_n": lst[0][2] if lst else 0, "all_signatures": " | ".join(f"{s} ({p:.2f})" for p, s, n in lst[:5])})
    out = pd.DataFrame(out).sort_values(["best_p_top_q_42", "n_signatures"], ascending=False)
    out.to_csv(RESULTS_DIR / "signatures_today.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print("\nconfirmed signatures (top 40 by confirmation-window probability):")
    print(confirmed[["signature", "n_disc", "p_disc", "n_conf", "p_conf", "lift_conf"]].head(40).round(3).to_string(index=False))
    print(f"\nnames firing >=1 confirmed signature today: {(out.n_signatures > 0).sum()} of {len(out)}")
    print(out.head(40).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
