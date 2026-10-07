"""The buy list.

Final ordering = out-of-sample probability that the name's next 42 sessions are a TOP-QUARTILE smooth climb against
the WHOLE universe (Sharpe, max drawdown, straightness, up-day share, each ranked across all names on the date).
The probability is read off a (beta bucket x score tier) table built from the walk-forward predictions, so beta is
accounted for by realised history rather than by a quota: a bucket only ranks high if its top-tier names actually
produced top-quartile paths against everyone else. No forced representation of any bucket.
"""
import numpy as np
import pandas as pd
from .calib import BinnedIsotonic
from .config import CACHE_DIR, RESULTS_DIR, RECENT_START
from .run_eval import load_research
from .composite import tercile_series
from .regime import tag, today_regime, same_regime

TIERS = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
MIN_CELL_N = 500   # a (bucket, tier) cell thinner than this is replaced by the pooled "this tier and above" cell
TIER_LABELS = ["<50", "50-70", "70-80", "80-90", "90-95", "95-100"]


def universe_smooth(df: pd.DataFrame, h: int) -> pd.Series:
    parts = [df[f"fwd_sharpe_{h}"], -df[f"fwd_mdd_{h}"], df[f"fwd_r2_{h}"], df[f"fwd_up_{h}"]]
    return sum(p.groupby(df.date).rank(pct=True) for p in parts) / 4


def main():
    cols = ["date", "ticker", "beta_252", "vol_roc_21", "ret_21", "rv20_roc_21"] + [f"fwd_{k}_{h}" for h in (42, 63) for k in ("sharpe", "mdd", "r2", "up")]
    df = tag(load_research()[cols].reset_index(drop=True))
    df["beta_bucket"] = tercile_series(df)
    reg = today_regime()
    # the signature: participation rising while volatility compresses on an advance, all at once (a conjunction, kept intact)
    df["sig"] = ((df.vol_roc_21 > 0.5) & (df.ret_21 > 0.03) & (df.rv20_roc_21 < -0.2)).astype(int)
    for h in (42, 63):
        u = universe_smooth(df, h)
        ur = u.groupby(df.date).rank(pct=True)          # re-rank the combined score so exactly 25% / 50% qualify
        df[f"u_top_q_{h}"] = (ur >= 0.75).astype(float).where(u.notna())
        df[f"u_top_half_{h}"] = (ur >= 0.5).astype(float).where(u.notna())
    st = pd.read_parquet(CACHE_DIR / "composite_state_smooth_oos_preds.parquet")[["date", "ticker", "comp_bucketed"]].rename(columns={"comp_bucketed": "state"})
    lv = pd.read_parquet(CACHE_DIR / "composite_level_smooth_oos_preds.parquet")[["date", "ticker", "comp_bucketed"]].rename(columns={"comp_bucketed": "level"})
    d = df.merge(st, on=["date", "ticker"]).merge(lv, on=["date", "ticker"]).dropna(subset=["state", "level", "u_top_q_42"])
    for c in ["state", "level"]:
        d[c + "_p"] = d.groupby("date")[c].rank(pct=True)
    d["avg_p"] = (d.state_p + d.level_p) / 2
    d["tier"] = pd.cut(d.avg_p, TIERS, labels=TIER_LABELS, include_lowest=True)
    rec = d[(d.date >= RECENT_START) & same_regime(d, reg)]
    print(f"MOVE band today: {reg}; rate-sensitive sectors measured on {rec[rec.regime != 'all'].date.nunique()} same-band sessions, others on all {rec.date.nunique()} sessions since {RECENT_START}")
    tab = rec.groupby(["beta_bucket", "tier"], observed=True).agg(n=("u_top_q_42", "size"), p_top_q_42=("u_top_q_42", "mean"), p_top_half_42=("u_top_half_42", "mean"),
                                                                   p_top_q_63=("u_top_q_63", "mean"), p_top_half_63=("u_top_half_63", "mean"))
    # thin cells -> pooled downward within the bucket: widen the tier's lower edge until the pool holds MIN_CELL_N rows
    cols_p = ["p_top_q_42", "p_top_half_42", "p_top_q_63", "p_top_half_63"]
    src = {"p_top_q_42": "u_top_q_42", "p_top_half_42": "u_top_half_42", "p_top_q_63": "u_top_q_63", "p_top_half_63": "u_top_half_63"}
    tab["pooled_from"] = ""
    for (b_, lab), row in tab.iterrows():
        if row["n"] >= MIN_CELL_N:
            continue
        i = TIER_LABELS.index(lab)
        hi = TIERS[i + 1] if lab != TIER_LABELS[-1] else 1.01
        for j in range(i, -1, -1):
            pool = rec[(rec.beta_bucket == b_) & (rec.avg_p >= TIERS[j]) & (rec.avg_p < hi)]
            if len(pool) >= MIN_CELL_N or j == 0:
                for c in cols_p:
                    tab.loc[(b_, lab), c] = pool[src[c]].mean()
                tab.loc[(b_, lab), "pooled_from"] = f"{TIER_LABELS[j]}..{lab} (n={len(pool)})"
                break
    tab.to_csv(RESULTS_DIR / "buylist_probability_table.csv")
    # joint cells: (bucket, tier, signature) -> realised probability; fall back to the (bucket, tier) cell when the joint cell is thin
    jt = rec.groupby(["beta_bucket", "tier", "sig"], observed=True).agg(n=("u_top_q_42", "size"), p_top_q_42=("u_top_q_42", "mean"), p_top_half_42=("u_top_half_42", "mean"),
                                                                        p_top_q_63=("u_top_q_63", "mean"), p_top_half_63=("u_top_half_63", "mean"))
    jt.to_csv(RESULTS_DIR / "buylist_probability_table_signature.csv")
    sig_only = rec.groupby(["beta_bucket", "sig"], observed=True).agg(n=("u_top_q_42", "size"), p_top_q_42=("u_top_q_42", "mean"), p_top_half_42=("u_top_half_42", "mean"))
    print("signature alone (recent OOS) by bucket:\n", sig_only.round(3).to_string())

    # continuous calibration: isotonic fit of the realised outcome on the composite score, per bucket (same OOS rows)
    iso = {}
    for b_ in ["low", "mid", "high"]:
        s_ = rec[rec.beta_bucket == b_]
        iso[b_] = {}
        for c, src in [("iso_q42", "u_top_q_42"), ("iso_h42", "u_top_half_42"), ("iso_q63", "u_top_q_63"), ("iso_h63", "u_top_half_63")]:
            ok_ = s_[src].notna()
            iso[b_][c] = BinnedIsotonic(MIN_CELL_N).fit(s_.avg_p.values[ok_], s_[src].values[ok_])
    u = pd.read_csv(RESULTS_DIR / "universe_scores_smooth.csv")
    asof = u["asof"].iloc[0]
    for c in ["iso_q42", "iso_h42", "iso_q63", "iso_h63"]:
        u[c] = [iso[b_][c].predict([x])[0] if (isinstance(b_, str) and b_ in iso and pd.notna(x)) else np.nan for b_, x in zip(u.beta_bucket, u.avg_score)]
    u["tier"] = pd.cut(u.avg_score, TIERS, labels=TIER_LABELS, include_lowest=True)
    today = df[df.date == df.date.max()].set_index("ticker")
    u["signature"] = u.ticker.map(today["sig"]).fillna(0).astype(int)
    for c in ["p_top_q_42", "p_top_half_42", "p_top_q_63", "p_top_half_63"]:
        vals = []
        for b, t, g in zip(u.beta_bucket, u.tier.astype(str), u.signature):
            cell = jt[c].get((b, t, g), np.nan) if jt["n"].get((b, t, g), 0) >= 40 else np.nan
            vals.append(cell if pd.notna(cell) else tab[c].get((b, t), np.nan))
        u[c] = vals
    scored = u[u.p_top_q_42.notna()].sort_values(["p_top_q_42", "p_top_q_63", "avg_score"], ascending=False).reset_index(drop=True)
    scored.insert(0, "rank", scored.index + 1)
    out_cols = ["rank", "ticker", "Company", "Sector", "beta_bucket", "beta_252", "signature", "p_top_q_42", "p_top_half_42", "p_top_q_63", "p_top_half_63",
                "avg_score", "state_score_pooled", "level_score", "top_states"]
    scored[out_cols].round(3).to_csv(RESULTS_DIR / "buylist.csv", index=False)
    u.to_csv(RESULTS_DIR / "universe_scores_smooth.csv", index=False)   # carries the isotonic composite probabilities to final_list
    lines = [f"BUY LIST as of {asof} close. Every name ranked by the probability that its next 42 sessions are a top-quartile smooth climb",
             "against the whole universe (Sharpe + max drawdown + straightness + up-day share). Baseline for any name: 25% (Q) / 50% (H).",
             "Probabilities are realised out-of-sample frequencies (2022-10 onward) for names in the same beta bucket and score tier.", "",
             "SIG = volume ROC21 > +50% AND price +3% over 21d AND 20d-RV ROC21 < -20%, simultaneously. When set, the probability is the realised cell for names with the signature in that bucket and tier.", "",
             f"{'#':>3} {'tkr':<6} {'sector':<22} {'beta':<5} {'b252':>5} SIG {'Q42':>5} {'H42':>5} {'Q63':>5} {'H63':>5} {'score':>5}  what the tape is doing"]
    for _, r in scored.iterrows():
        lines.append(f"{int(r['rank']):>3} {r.ticker:<6} {str(r.Sector)[:22]:<22} {str(r.beta_bucket):<5} {r.beta_252:>5.2f}  {'Y' if r.signature else '-'}  {r.p_top_q_42:>5.2f} {r.p_top_half_42:>5.2f} {r.p_top_q_63:>5.2f} {r.p_top_half_63:>5.2f} {r.avg_score:>5.2f}  {r.top_states}")
    unsc = u[u.p_top_q_42.isna()]
    lines += ["", "not ranked:"] + [f"    {r.ticker:<6} {r.note if isinstance(r.note, str) and r.note else 'no probability cell (bucket/tier unavailable)'}" for _, r in unsc.iterrows()]
    (RESULTS_DIR / "buylist.txt").write_text("\n".join(lines))
    pd.set_option("display.width", 200)
    print("probability table (recent OOS): P(top-quartile universe-wide smooth path, 42d) by beta bucket x score tier")
    print(tab["p_top_q_42"].unstack().round(3).to_string())
    print("\n" + "\n".join(lines[:45]))


if __name__ == "__main__":
    main()
