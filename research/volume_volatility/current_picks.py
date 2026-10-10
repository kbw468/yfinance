"""Current long candidates from the 2023+ long-only setups, with historical odds.

For each setup it measures, from --odds-start on, how often a single qualifying stock beat the average stock
(stock-level win rate) and by how much, then lists the tickers that qualify on the latest date.
It also checks whether a more extreme reading inside a setup did better (top vs bottom half of the setup's
composite score), which decides whether the within-setup order means anything.

Setups (own-history percentile over the past 252 sessions):
  A  steady 50d volume (bottom 20%) + 1-week jump in volume swings (10/5, top 20%) + down day
  B  steady 20d volume (bottom 20%) + 1-week jump (15/5, top 20%) + down day
  C  steady 120d volume (bottom 20%) + 1-week jump (15/5, top 20%) + up day
  D  steady 120d volume (bottom 20%) + down day
  E  steady 120d volume (bottom 20%), any day

Usage: python current_picks.py <bars.parquet> <out_dir> [--odds-start D] [--min-history N]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt

SETUPS = {
    "A": dict(level=50, jump=(10, 5), day="DOWN", h=42),
    "B": dict(level=20, jump=(15, 5), day="DOWN", h=42),
    "C": dict(level=120, jump=(15, 5), day="UP", h=42),
    "D": dict(level=120, jump=None, day="DOWN", h=21),
    "E": dict(level=120, jump=None, day="ANY", h=21),
}


def rank(x):
    return x.replace([np.inf, -np.inf], np.nan).rolling(252, min_periods=126).rank(pct=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bars")
    ap.add_argument("out_dir")
    ap.add_argument("--odds-start", default="2023-01-01")
    ap.add_argument("--min-history", type=int, default=400)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bars = pd.read_parquet(a.bars)
    close = bars.pivot(index="date", columns="ticker", values="close").sort_index()
    close = close.where(close > 0)
    vol = bars.pivot(index="date", columns="ticker", values="volume").reindex(close.index)
    vol = vol.where(vol > 0)
    ret = close.pct_change(fill_method=None)
    sigma = np.log(close).diff().rolling(20, min_periods=18).std()
    adv = (close * vol).rolling(20, min_periods=18).median()
    tradable = (close >= bt.MIN_PRICE) & (adv >= bt.MIN_DOLLAR_ADV) & (sigma > 0) & vol.notna()
    seasoned = close.notna().cumsum() >= a.min_history
    ok = tradable & seasoned
    dlv = bt.log_volume_change(vol)
    vvs = {w: dlv.rolling(w, min_periods=w - 2).std() for w in (10, 15, 20, 50, 120)}
    lvl = {w: rank(vvs[w]) for w in (20, 50, 120)}
    jmp = {(w, l): rank(vvs[w] / vvs[w].shift(l) - 1) for w, l in [(10, 5), (15, 5)]}
    dayk = {"DOWN": ret < 0, "UP": ret > 0, "ANY": ret.notna()}

    last = close.index[-1]
    odds_rows_cal = []
    hist = np.asarray(close.index >= a.odds_start)
    odds_rows, pick_rows = [], []
    for name, s in SETUPS.items():
        m = ok & dayk[s["day"]] & (lvl[s["level"]] <= 0.2)
        score = 0.2 - lvl[s["level"]]  # how far into the steady zone
        if s["jump"]:
            m &= jmp[s["jump"]] > 0.8
            score = score + (jmp[s["jump"]] - 0.8)
        h = s["h"]
        f = (close.shift(-h) / close - 1).where(tradable)
        ex = f.sub(f.mean(axis=1), axis=0)
        exm = f.sub(f.median(axis=1), axis=0).values  # vs the typical (median) stock: coin flip = 50%
        fv = f.values
        mh = m.values & hist[:, None] & ex.notna().values
        allh = ok.values & dayk[s["day"]].values & hist[:, None] & ex.notna().values
        exv = ex.values
        sc = score.values
        med = np.nanmedian(np.where(mh, sc, np.nan))
        top, bot = mh & (sc >= med), mh & (sc < med)
        # strength terciles of the setup score, cut on history, give each name its bucket's real odds
        cuts = np.nanquantile(np.where(mh, sc, np.nan), [1 / 3, 2 / 3])
        terc = np.where(np.isnan(sc), np.nan, 1 + (sc >= cuts[0]) + (sc >= cuts[1]))
        cal = {}
        for k in (1, 2, 3):
            mk = mh & (terc == k)
            cal[k] = dict(p_beat_typical=float((exm[mk] > 0).mean()), p_beat_avg=float((exv[mk] > 0).mean()),
                          p_up=float((fv[mk] > 0).mean()), avg_excess=float(exv[mk].mean()), n=int(mk.sum()))
            odds_rows_cal.append(dict(setup=name, strength=k, **cal[k]))
        odds_rows.append(dict(
            setup=name, hold_days=h, signals=int(mh.sum()),
            win_rate=float((exv[mh] > 0).mean()), avg_excess=float(exv[mh].mean()),
            median_excess=float(np.median(exv[mh])),
            base_win_rate=float((exv[allh] > 0).mean()), base_avg_excess=float(exv[allh].mean()),
            p_beat_typical=float((exm[mh] > 0).mean()), base_p_beat_typical=float((exm[allh] > 0).mean()),
            p_up=float((fv[mh] > 0).mean()), base_p_up=float((fv[allh] > 0).mean()),
            top_half_win=float((exv[top] > 0).mean()), top_half_excess=float(exv[top].mean()),
            bottom_half_win=float((exv[bot] > 0).mean()), bottom_half_excess=float(exv[bot].mean())))
        today = m.loc[last]
        for tk in today[today].index:
            sv = float(score.loc[last, tk])
            k = int(1 + (sv >= cuts[0]) + (sv >= cuts[1]))
            pick_rows.append(dict(setup=name, ticker=tk, score=sv, strength=k, hold_days=h, **cal[k],
                                  steady_pct=float(lvl[s["level"]].loc[last, tk]),
                                  jump_pct=float(jmp[s["jump"]].loc[last, tk]) if s["jump"] else np.nan,
                                  day_return=float(ret.loc[last, tk]), close=float(close.loc[last, tk]),
                                  dollar_volume_20d=float(adv.loc[last, tk])))
    odds = pd.DataFrame(odds_rows)
    cal = pd.DataFrame(odds_rows_cal)
    picks = pd.DataFrame(pick_rows)
    odds.to_csv(out / "setup_odds.csv", index=False)
    cal.to_csv(out / "setup_odds_by_strength.csv", index=False)
    picks.to_csv(out / "picks_raw.csv", index=False)
    # one row per ticker: its best setup by historical odds of beating the typical stock
    best = picks.sort_values(["p_beat_typical", "avg_excess"], ascending=False).groupby("ticker").head(1)
    also = picks.groupby("ticker").setup.apply(lambda x: "".join(sorted(x)))
    best = best.assign(setups=best.ticker.map(also)).sort_values(["p_beat_typical", "avg_excess"], ascending=False)
    best.to_csv(out / "picks_ranked.csv", index=False)
    print(cal.round(4).to_string(index=False))
    print(f"latest date: {last.date()}  names with data: {int(close.loc[last].notna().sum())}")
    print(odds.round(4).to_string(index=False))
    print(picks.groupby("setup").ticker.count())


if __name__ == "__main__":
    main()
