"""Realised out-of-sample performance of the signature RULE as it is applied to the list:
on each confirmation-window date, for each name, take the confirmed signatures firing, pick the one with the highest
DISCOVERY probability, and record the realised outcome (top-quartile smooth path 42d / 63d). The mean of those outcomes
is the probability the list is entitled to print for "name carries a confirmed signature", by beta bucket.
Also writes plain-English descriptions of the conditions and tonight's firing names with the rule's probability."""
import numpy as np
import pandas as pd
from .config import RESULTS_DIR, RECENT_START
from .run_eval import load_research
from .composite import tercile_series
from .signatures import QUINTILE_FACTORS, FIXED, build_conditions, smooth_top_quartile

PLAIN = {
    "mom_12_1": "12-1m momentum", "ret_21": "21d return", "ret_63": "63d return", "rs_lead_126": "RS line leading price", "rs_spy_63": "63d RS vs SPY",
    "dist_52w_high": "distance to 52w high", "dist_52w_low": "gain off 52w low", "base_depth_126": "depth below 126d high", "days_since_20pct_dd": "time since 20% drawdown",
    "pullback_21": "pullback from 21d high", "up_day_frac_63": "share of up days 63d", "clv_21": "close location in range", "gap_share_21": "overnight share of moves",
    "higher_low_cadence": "higher-low cadence", "nr7_count_20": "NR7 count", "range_comp_10_252": "range compression", "bbw_pct_252": "Bollinger width vs norm",
    "vol_dry_20_250": "volume vs 250d", "vol_dry_10_50": "volume vs 50d", "updown_vol_ratio_50": "up/down volume", "accum_dist_50": "accumulation-distribution days",
    "obv_price_div_63": "OBV leading price", "net_vol_flow_63": "net volume flow", "log_dvol_63": "dollar volume", "dvol_trend_21_126": "dollar-volume trend",
    "ignition_mult_10": "ignition-day volume", "days_since_ignition": "time since ignition day", "pullback_vol_21": "volume on pullbacks",
    "rv20_pct_252": "20d RV vs norm", "rv_ratio_5_20": "5d/20d RV", "rv_ratio_20_60": "20d/60d RV", "vol_of_vol_63": "vol of vol", "dn_up_vol_asym_63": "downside/upside vol",
    "dn_vol_63": "downside vol", "parkinson_cc_20": "range vol / close vol", "skew_63": "skew", "idio_vol_63": "idiosyncratic vol", "sharpe_126": "126d Sharpe",
    "corr_spy_63": "corr to SPY", "beta_252": "beta", "up_capture_63": "up-capture", "capture_spread_63": "up minus down capture",
}


def plain(cond: str) -> str:
    if cond in FIXED:
        return cond
    f, side = cond.split(":")
    z = f.startswith("z_")
    name = PLAIN.get(f[2:] if z else f, f)
    return f"{name} {'vs own norm ' if z else ''}{'top' if side == 'TOP' else 'bottom'} 20%"


def main():
    sig = pd.read_csv(RESULTS_DIR / "signatures_all.csv")
    conf = sig[sig.confirmed].sort_values("p_disc", ascending=False).reset_index(drop=True)
    need = ["date", "ticker", "sector", "beta_252", "vol_roc_21", "vol_roc_10", "rv20_roc_21", "rv20_roc_10", "ret_21", "ret_5", "dvol_roc_21", "vol_accel_5", "rv_accel_5",
            "fwd_sharpe_42", "fwd_mdd_42", "fwd_r2_42", "fwd_up_42", "fwd_sharpe_63", "fwd_mdd_63", "fwd_r2_63", "fwd_up_63"] + QUINTILE_FACTORS
    df = load_research()[list(dict.fromkeys(need))].reset_index(drop=True)
    df["beta_bucket"] = tercile_series(df)
    y42 = smooth_top_quartile(df, 42).astype(float).values
    y63 = smooth_top_quartile(df, 63).astype(float).values
    conds = build_conditions(df)
    n = len(df)
    best_rank = np.full(n, np.inf)      # index into conf (lower = higher discovery p)
    for i, r in conf.iterrows():
        m = np.ones(n, dtype=bool)
        for p in r.signature.split(" & "):
            m &= conds[p]
        upd = m & (best_rank > i)
        best_rank[upd] = i
    df["best_sig_idx"] = np.where(np.isfinite(best_rank), best_rank, -1).astype(int)
    df["fires"] = df.best_sig_idx >= 0
    df["y42"] = y42; df["y63"] = y63
    rec = df[(df.date >= RECENT_START) & ~np.isnan(df.y42)]
    rows = []
    for b in ["all", "low", "mid", "high"]:
        s = rec if b == "all" else rec[rec.beta_bucket == b]
        f, nf = s[s.fires], s[~s.fires]
        rows.append({"bucket": b, "n_fire": len(f), "share_of_names_firing": len(f) / len(s), "P_topq42_fire": f.y42.mean(), "P_topq42_nofire": nf.y42.mean(),
                     "P_topq63_fire": f.y63.mean(), "P_topq63_nofire": nf.y63.mean()})
    rule = pd.DataFrame(rows).set_index("bucket")
    rule.to_csv(RESULTS_DIR / "signature_rule_oos.csv")
    # by number of confirmed signatures firing (depth), recent
    cnt = np.zeros(n, dtype=int)
    for i, r in conf.iterrows():
        m = np.ones(n, dtype=bool)
        for p in r.signature.split(" & "):
            m &= conds[p]
        cnt += m
    df["n_fire"] = cnt
    rec = df[(df.date >= RECENT_START) & ~np.isnan(df.y42)]
    depth = rec.groupby(pd.cut(rec.n_fire, [-1, 0, 2, 5, 10, 20, 50, 10000], labels=["0", "1-2", "3-5", "6-10", "11-20", "21-50", "50+"]), observed=True).agg(
        n=("y42", "size"), P_topq42=("y42", "mean"), P_topq63=("y63", "mean"))
    depth.to_csv(RESULTS_DIR / "signature_depth_oos.csv")
    # tonight
    today = df[df.date == df.date.max()].copy()
    today["best_signature"] = today.best_sig_idx.map(lambda i: conf.signature.iloc[i] if i >= 0 else "")
    today["best_signature_plain"] = today.best_signature.map(lambda s: " AND ".join(plain(c) for c in s.split(" & ")) if s else "")
    today["best_p_conf"] = today.best_sig_idx.map(lambda i: conf.p_conf.iloc[i] if i >= 0 else np.nan)
    today["best_n_conf"] = today.best_sig_idx.map(lambda i: conf.n_conf.iloc[i] if i >= 0 else np.nan)
    today = today[["ticker", "sector", "beta_bucket", "n_fire", "best_signature", "best_signature_plain", "best_p_conf", "best_n_conf"]].sort_values(["n_fire"], ascending=False)
    today.to_csv(RESULTS_DIR / "signatures_today.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 110)
    print("RULE as applied (recent OOS): P(top-quartile smooth path) when a confirmed signature fires vs not, by beta bucket")
    print(rule.round(3).to_string())
    print("\nby number of confirmed signatures firing:\n", depth.round(3).to_string())
    print(f"\ntonight: {int(today.n_fire.gt(0).sum())} names fire; top 30 by depth:")
    print(today.head(30)[["ticker", "sector", "beta_bucket", "n_fire", "best_p_conf", "best_signature_plain"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
