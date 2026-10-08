#!/usr/bin/env python3
"""What happens if the rule is chosen to avoid the 2024-2026 drawdowns. Run from risk_regime/work after onset.py.
Pool: every rule already built (decision grid, 15% layer rules, onset grid; about 200). Choose on 2024-01-01 to date
by (a) smallest worst drawdown, (b) best return per unit of worst drawdown; then score the chosen rules on 2006-2023,
which they never saw, against the reference rule and buy and hold. Feb-Jul 2020 included in the later scoring."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); pd.set_option('display.width', 260)
F = pd.read_pickle('F.pkl'); idx = F.index; spy = pd.read_pickle('OHLCV.pkl')['T']['SPY']['Close'].reindex(idx); r = np.log(spy).diff()
ST = dict(pd.read_pickle('DECISION.pkl')['states']); ST.update(pd.read_pickle('STATES15.pkl')); ST.update(pd.read_pickle('STATES_ONSET.pkl'))
REF = 'C: OUT P_off>q0.95, IN P_off<q0.8'
def score(S, a, b):
    m = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)); w = S.shift(1).fillna(1); x = (w * r)[m]; c = x.cumsum(); dd = c - c.cummax()
    o = pd.Series(((S == 0) & m).to_numpy(dtype=bool), index=idx)
    return {'ann %': round(x.mean() * 252 * 100, 1), 'worst DD %': round((np.exp(dd.min()) - 1) * 100, 1), 'times OUT': int((o & ~o.shift(1, fill_value=False)).sum())}
rows = []
for k, S in ST.items():
    a = score(S, '2024-01-01', '2026-12-31'); b = score(S, '2006-01-01', '2023-12-31'); rows.append({'rule': k, **{f'24-26 {x}': v for x, v in a.items()}, **{f'06-23 {x}': v for x, v in b.items()}})
T = pd.DataFrame(rows); T['24-26 return per DD'] = T['24-26 ann %'] / T['24-26 worst DD %'].abs().clip(lower=1)
bh = {'rule': 'SPY buy and hold', **{f'24-26 {x}': v for x, v in score(pd.Series(1, index=idx), '2024-01-01', '2026-12-31').items()}, **{f'06-23 {x}': v for x, v in score(pd.Series(1, index=idx), '2006-01-01', '2023-12-31').items()}}
cols = ['rule', '24-26 ann %', '24-26 worst DD %', '24-26 times OUT', '06-23 ann %', '06-23 worst DD %', '06-23 times OUT']
print(f'{len(T)} rules in the pool')
print('\n=== the 8 rules with the smallest 2024-26 drawdown (ties broken by return) ===')
print(pd.concat([pd.DataFrame([bh]), T.sort_values(['24-26 worst DD %', '24-26 ann %'], ascending=[False, False]).head(8), T[T.rule == REF]])[cols].to_string(index=False))
print('\n=== the 8 rules with the best 2024-26 return per unit of drawdown ===')
print(pd.concat([pd.DataFrame([bh]), T.sort_values('24-26 return per DD', ascending=False).head(8), T[T.rule == REF]])[cols].to_string(index=False))
top = T.sort_values('24-26 return per DD', ascending=False).head(20); rest = T
print(f"\nrank correlation between 2024-26 return-per-drawdown and 2006-23 annual return across all {len(T)} rules: {T['24-26 return per DD'].rank().corr(T['06-23 ann %'].rank()):.2f}")
print(f"top 20 by 2024-26: mean 2006-23 ann {top['06-23 ann %'].mean():.1f}%, mean worst DD {top['06-23 worst DD %'].mean():.1f}% | all rules: {T['06-23 ann %'].mean():.1f}%, {T['06-23 worst DD %'].mean():.1f}%")
