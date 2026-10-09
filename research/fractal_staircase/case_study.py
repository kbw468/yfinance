"""Case study for one ticker (in or outside the universe), usage: python case_study.py TICKER.
Same feature engine, ranked against the study universe each week. Not used by the study."""
import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import pandas as pd
import yfinance as yf

import features as F
from common import DATA_DIR, RES_DIR, add_targets

TICK = sys.argv[1] if len(sys.argv) > 1 else "TD"

raw = yf.download([TICK, "SPY"], start="2004-01-01", auto_adjust=True, progress=False, group_by="column")
spy = np.log(raw["Close"]["SPY"]).diff()
spy.index = pd.to_datetime(spy.index).tz_localize(None)
df = pd.DataFrame({c.lower(): raw[c][TICK] for c in ["Open", "High", "Low", "Close", "Volume"]}).dropna()
df.index = pd.to_datetime(df.index).tz_localize(None)
df = df.reset_index().rename(columns={"Date": "date", "index": "date"})

cols = ["date", "ticker", "mar63", "mdd126", "roc126", "rng63", "bo_gain126", "mdd63", "hh_steps63", "kt63", "d_kt63",
        "kt_roc", "gp63", "mar252", "d_mdd63_rel", "dq_mdd63", "dq_kt63", "fr21", "fdd21", "fr42", "fdd42", "fr63", "fdd63"]
pnl = pd.read_parquet(f"{DATA_DIR}/panel.parquet", columns=cols)
pnl = add_targets(pnl)
dates = np.sort(pnl["date"].unique())

f = F.ticker_features(df, set(dates), spy)
f = f.sort_values("date").reset_index(drop=True)
# derived (weekly-row shifts, as in xsec.derived)
for c in ("kt63", "mdd63_rel"):
    f[f"d_{c}"] = f[c] - f[c].shift(4)
f["stair_b63"] = ((f["roc63"] >= np.log(1.10)) & (f["mdd63"] <= f["roc63"] / 3)).astype(float)
f = add_targets(f)

g = {d: grp for d, grp in pnl.groupby("date")}


def pct_vs_universe(row, col):
    u = g[row["date"]][col].dropna().to_numpy()
    v = row[col]
    if np.isnan(v) or len(u) == 0:
        return np.nan
    return (u <= v).mean()


for c in ["mar63", "mdd126", "roc126", "rng63", "bo_gain126", "mdd63", "hh_steps63", "kt_roc", "gp63", "mar252", "d_kt63",
          "d_mdd63_rel", "kt63"]:
    f["q_" + c] = f.apply(lambda r: pct_vs_universe(r, c), axis=1)
f["dq_mdd63"] = f["q_mdd63"] - f["q_mdd63"].shift(4)
f["dq_kt63"] = f["q_kt63"] - f["q_kt63"].shift(4)
for c in ["dq_mdd63", "dq_kt63"]:
    f["q_" + c] = f.apply(lambda r: pct_vs_universe(r, c), axis=1)

# quiet base rungs
qb = [f["q_mar63"] <= 0.2, f["q_mdd126"] <= 0.3, f["q_roc126"] <= 0.4, f["ddh252"] > -0.105]
alive = np.ones(len(f), bool)
f["qb_rungs"] = 0
for c in qb:
    alive &= c.fillna(False).to_numpy()
    f["qb_rungs"] += alive
f["qb_strict"] = (f["q_mar63"] <= 0.2) & (f["q_mdd126"] <= 0.2) & (f["q_roc126"] <= 0.3) & (f["ddh252"] > -0.05)

# top out-of-sample rules (as used in the scan), skipping the one that needs industry-level data
import re
rules = pd.read_csv(f"{RES_DIR}/scan_rules_used.csv")
hits = []
for i, r in rules.iterrows():
    conds = []
    ok = True
    for part in r["rule"].split(" & "):
        m = re.match(r"(.+?)(<=|>=)(-?[\d.]+)$", part.strip())
        col, op, v = m.group(1), m.group(2), float(m.group(3))
        if col not in f.columns:
            ok = False
            break
        conds.append((col, op, v))
    if not ok:
        continue
    mask = np.ones(len(f), bool)
    for col, op, v in conds:
        x = f[col].to_numpy()
        mask &= (x <= v) if op == "<=" else (x >= v)
    f[f"R{i + 1}"] = mask
rcols = [c for c in f.columns if re.fullmatch(r"R\d+", c)]
f["rules_hit"] = f[rcols].sum(axis=1)
f["rules_list"] = f[rcols].apply(lambda r: ",".join(c for c in rcols if r[c]), axis=1)
f["thesis"] = ((f["roc63"] > 0) & (f["vroc21"] < 0) & (f["rv10_126"] > 0) & (f["q_mdd63"] < 0.5) & (f["acc21"] > 0)
               & (f["vacc21"] < 0) & (f["vlroc21"] > 0))

# regime-neutral lift of TD's own staircase frequency
wb = pnl.groupby("date")["y63"].mean()
f["wb"] = f["date"].map(wb)
lab = f[f["y63"].notna() & (f["date"] >= "2006-01-01")]
print(f"{TICK}: weeks {len(lab)}, staircase weeks {int(lab['y63'].sum())} ({lab['y63'].mean():.1%}), "
      f"universe same weeks {lab['wb'].mean():.1%}, lift {lab['y63'].sum() / lab['wb'].sum():.2f}")

# episodes: runs of consecutive staircase weeks
lab = lab.reset_index(drop=True)
ep = (lab["y63"] != lab["y63"].shift()).cumsum()
rows = []
for k, e in lab[lab["y63"] == 1].groupby(ep[lab["y63"] == 1]):
    first = e.iloc[0]
    rows.append(dict(start=first["date"].date(), weeks=len(e), fr63_first=first["fr63"], best_fr63=e["fr63"].max(),
                     fdd63_first=first["fdd63"], qb_rungs=int(first["qb_rungs"]), rules=first["rules_list"],
                     thesis=bool(first["thesis"]), already_stair=bool(first["stair_b63"] == 1),
                     q_vol=first["q_mar63"], q_mdd126=first["q_mdd126"], q_roc126=first["q_roc126"], ddh252=first["ddh252"]))
eps = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\nstaircase episodes (first qualifying week):")
print(eps.round(3).to_string(index=False))

w = f[f["date"] >= "2024-06-01"][["date", "close" if "close" in f.columns else "logp", "roc63", "roc126", "ddh252", "q_mar63",
                                  "q_mdd126", "q_roc126", "q_rng63", "qb_rungs", "rules_list", "thesis", "stair_b63", "vroc21",
                                  "rv10_126", "y63", "fr63", "fdd63"]].copy()
w["price"] = np.exp(f.loc[w.index, "logp"])
print("\nweekly view since 2024-06:")
print(w.drop(columns=[c for c in ["logp", "close"] if c in w.columns]).round(3).to_string(index=False))
f.to_csv(f"{DATA_DIR}/case_{TICK}.csv", index=False)
