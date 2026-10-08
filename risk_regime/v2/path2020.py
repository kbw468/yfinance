#!/usr/bin/env python3
"""v2 Stage 10: the 2020 path. Run from risk_regime/work after decision.py.

Every statistic in v2 excludes Feb 1 to Jul 31 2020 by design. The fits and the trailing thresholds never saw
those sessions as labels, but the models did produce probabilities on them and the state machines ran through
them. This script scores the frozen rules, fits and thresholds unchanged, on the 2006-to-date path WITH the
window included, and prints the result next to the excluded one. Writes PATH2020.pkl.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); pd.set_option('display.width', 320); pd.set_option('display.max_rows', 100)
F = pd.read_pickle('F.pkl'); idx = F.index; O = pd.read_pickle('OHLCV.pkl'); spy = O['T']['SPY']['Close'].reindex(idx); r = np.log(spy).diff()
DEC = pd.read_pickle('DECISION.pkl'); ST = dict(DEC['states']); HEAD = 'C: OUT P_off>q0.95, IN P_off<q0.8'
EXM = (idx >= pd.Timestamp('2020-02-01')) & (idx <= pd.Timestamp('2020-07-31')); m_all = idx >= pd.Timestamp('2006-01-01'); m_ex = m_all & ~EXM
a, b = pd.Timestamp('2020-02-19'), pd.Timestamp('2020-03-23')
def stats(x):
    c = x.cumsum(); dd = c - c.cummax(); return dict(ann=round(x.mean() * 252 * 100, 2), sharpe=round(x.mean() / x.std() * np.sqrt(252), 2), maxDD_log=round(dd.min() * 100, 1), maxDD_price=round((np.exp(dd.min()) - 1) * 100, 1))
rows = []
for name, S in [('SPY buy and hold', pd.Series(1, index=idx))] + list(ST.items()):
    w = S.shift(1); row = {'rule': name}
    for tag, m in (('ex-2020', m_ex), ('incl-2020', m_all)):
        s = stats((w * r)[m]); row.update({f'{tag} ann': s['ann'], f'{tag} sharpe': s['sharpe'], f'{tag} maxDD log': s['maxDD_log'], f'{tag} maxDD price': s['maxDD_price']})
    seg = r.loc[a:b].iloc[1:]; row['2020 crash taken'] = round(float((w.loc[seg.index] * seg).sum() / seg.sum()), 2)
    so = S.loc[a:b]; o = so[so == 0]; row['first OUT in crash'] = f'{o.index[0].date()} (SPY {spy.loc[o.index[0]] / spy.loc[a] * 100 - 100:.1f}%)' if len(o) else ('already OUT' if S.loc[a] == 0 else 'never')
    y20 = (idx.year == 2020); row['2020 book %'] = round(float((w * r)[y20].sum() * 100), 1); row['2020 SPY %'] = round(float(r[y20].sum() * 100), 1)
    back = S.loc['2020-03-23':'2020-12-31']; i = back[back == 1]; row['back IN'] = str(i.index[0].date()) if len(i) and S.loc[b] == 0 else ('never OUT' if (S.loc[a:b] == 1).all() else 'stayed IN')
    rows.append(row)
T = pd.DataFrame(rows); pd.to_pickle({'table': T}, 'PATH2020.pkl')
print('=== the frozen rules scored with Feb-Jul 2020 included, next to the excluded path (2006 to date, log returns) ===')
print(T.to_string(index=False))
h = T[T.rule == HEAD].iloc[0]; print(f"\nreference rule: max DD {h['ex-2020 maxDD log']} log ({h['ex-2020 maxDD price']}% price) excluding the window, {h['incl-2020 maxDD log']} log ({h['incl-2020 maxDD price']}% price) including it; took {h['2020 crash taken']} of the Feb 19 to Mar 23 2020 decline, first OUT {h['first OUT in crash']}, back IN {h['back IN']}; 2020 book {h['2020 book %']}% vs SPY {h['2020 SPY %']}%")
