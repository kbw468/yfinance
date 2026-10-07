"""Why a name sits where it does on THE LIST. Usage: python -m pvv_score.why JCI [NVDA ...]

For each ticker: composite score and tier, the quintile conditions live tonight (TOP / BOTTOM 20% of the universe),
the ROC thresholds met, the confirmed signatures firing, and the confirmed signatures it misses by exactly one
condition with the missing piece named (and the most common missing piece across all of them)."""
import sys
from collections import Counter
import numpy as np
import pandas as pd
from .config import CACHE_DIR, RESULTS_DIR
from .signatures import QUINTILE_FACTORS, FIXED
from .signature_rule import PLAIN


def plain(cond: str) -> str:
    if cond in FIXED:
        return cond
    f, side = cond.split(":")
    z = f.startswith("z_")
    return f"{PLAIN.get(f[2:] if z else f, f)}{' vs own norm' if z else ''} {'TOP' if side == 'TOP' else 'BOTTOM'} 20%"


def live_conditions(row, rk_row) -> set:
    live = set()
    for f in QUINTILE_FACTORS:
        if rk_row[f] >= 0.8:
            live.add(f"{f}:TOP")
        elif rk_row[f] <= 0.2:
            live.add(f"{f}:BOT")
    for k, (c, op, v) in FIXED.items():
        x = row[c]
        if pd.notna(x) and ((x > v) if op == ">" else (x < v)):
            live.add(k)
    return live


def main(tickers):
    df = pd.read_parquet(CACHE_DIR / "research_long.parquet")
    t = df[(df.date == df.date.max()) & df.eligible].set_index("ticker")
    rk = t[QUINTILE_FACTORS].rank(pct=True)
    L = pd.read_csv(RESULTS_DIR / "THE_LIST.csv").set_index("ticker")
    u = pd.read_csv(RESULTS_DIR / "universe_scores_smooth.csv").set_index("ticker")
    conf = pd.read_csv(RESULTS_DIR / "signatures_all.csv"); conf = conf[conf.confirmed].sort_values("p_conf", ascending=False)
    parts_list = [(r.signature.split(" & "), r.p_conf) for _, r in conf.iterrows()]
    for tk in tickers:
        if tk not in t.index:
            print(f"\n{tk}: not scored tonight"); continue
        row, rr = t.loc[tk], rk.loc[tk]
        live = live_conditions(row, rr)
        li = L.loc[tk]; ui = u.loc[tk]
        print(f"\n{'=' * 100}\n{tk}  {row.sector}  |  rank {int(li['rank'])}  tier {int(li.tier)}  P42 {li.P_topq_42d:.0%}  P63 {li.P_topq_63d:.0%}  |  beta {row.beta_252:.2f} ({ui.beta_bucket})  composite {ui.avg_score:.2f} (walk-forward; final-weight diagnostics: state {ui.state_score_pooled:.2f} level {ui.level_score:.2f})")
        print(f"state readings vs own norm: {ui.top_states}")
        top = sorted([f for f in QUINTILE_FACTORS if rr[f] >= 0.8], key=lambda f: -rr[f]); bot = sorted([f for f in QUINTILE_FACTORS if rr[f] <= 0.2], key=lambda f: rr[f])
        print("TOP 20% of universe:   " + ", ".join(f"{plain(f + ':TOP')[:-8]} ({rr[f]:.0%})" for f in top))
        print("BOTTOM 20% of universe: " + ", ".join(f"{plain(f + ':BOT')[:-11]} ({rr[f]:.0%})" for f in bot))
        roc_live = [k for k in FIXED if k in live]
        print("ROC thresholds met:    " + (", ".join(roc_live) if roc_live else "none") + "   | not met: " + ", ".join(f"{k} ({row[FIXED[k][0]]:+.2f})" for k in FIXED if k not in live))
        firing = [(p, s) for s, p in parts_list if all(c in live for c in s)]
        print(f"confirmed signatures firing: {len(firing)}" + (f"; strongest: " + " AND ".join(plain(c) for c in firing[0][1]) + f"  (P {firing[0][0]:.0%})" if firing else ""))
        near = [(p, s, [c for c in s if c not in live][0]) for s, p in parts_list if sum(c not in live for c in s) == 1]
        if near:
            miss = Counter(m for _, _, m in near).most_common(4)
            print(f"one condition away: {len(near)} confirmed signatures. most common missing piece: " + "; ".join(f"{plain(m)} ({n})" for m, n in miss))
            for p, s, m in near[:5]:
                print(f"   P {p:.0%}  " + " AND ".join(plain(c) for c in s) + f"   -> missing: {plain(m)}")
        else:
            print("one condition away: none")


if __name__ == "__main__":
    main(sys.argv[1:] or ["JCI"])
