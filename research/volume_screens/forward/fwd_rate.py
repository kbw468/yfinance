"""Forward score 1-10: walk-forward-validated model of forward 63d (primary) and 21d risk-adjusted return,
built only from price, volume and volatility. Data: Nasdaq API. Usage: python fwd_rate.py TICKER [TICKER ...]"""
import sys, os, pickle, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from fwdfeat import features
from rate import history, _get, _num
M = pickle.load(open(os.path.join(HERE,'fwd_model.pkl'),'rb'))
FEATS, REF = M['feats'], M['ref']
LG = {h: lgb.Booster(model_file=os.path.join(HERE, f'lgbm{h}.txt')) for h in (21,63)}
REFSORT = {f: np.sort(REF[f].dropna().values) for f in FEATS}
def pct_in(sorted_arr, x):
    if x is None or np.isnan(x) or len(sorted_arr)==0: return np.nan
    lo = np.searchsorted(sorted_arr, x, 'left'); hi = np.searchsorted(sorted_arr, x, 'right')
    return (lo + 0.5*(hi-lo) + 0.5)/(len(sorted_arr)+1)
def shares_out(t, last_close):
    d = _get(f'https://api.nasdaq.com/api/quote/{t}/summary?assetclass=stocks')
    try: return _num(d['summaryData']['MarketCap']['value'])/last_close
    except Exception: return np.nan
def score_vector(x):
    """x: Series of raw features -> dict of scores."""
    z = np.array([[pct_in(REFSORT[f], x.get(f, np.nan)) - 0.5 for f in FEATS]]); z = np.nan_to_num(z, nan=0.0)
    out = {}
    for h in (21,63):
        pr = float((z@M[f'ridge{h}'])[0]); pl = float(LG[h].predict(z)[0])
        b = pct_in(M[f'ref_pred_ridge{h}'], pr) + pct_in(M[f'ref_pred_lgbm{h}'], pl)
        p = pct_in(M[f'ref_pred_blend{h}'], b) if h==63 else pct_in(M[f'ref_pred_ridge{h}'], pr)
        out[h] = round(1 + 9*p, 1)
    return out
def rate(T, asof=None):
    frames = {t: history(t, years=4) for t in (T,'SPY')}
    d = pd.concat(frames, axis=1).swaplevel(0,1,axis=1).sort_index(axis=1).dropna(subset=[('Close','SPY')])
    if asof is not None: d = d.loc[:asof]
    elif d['Volume'][T].iloc[-1] < 0.6*d['Volume'][T].iloc[-22:-1].mean(): d = d.iloc[:-1]
    O,H,L,C,V = [d[k] for k in ['Open','High','Low','Close','Volume']]
    f = features(O,H,L,C,V, {T: shares_out(T, C[T].iloc[-1])}).loc[T]
    s = score_vector(f)
    flag = '' if (C[T].iloc[-1]>=3 and np.exp(f['e_logadv'])>=1e6) else ' (below price/liquidity floor used in training)'
    return C.index[-1].date(), s, flag
if __name__=='__main__':
    for t in sys.argv[1:]:
        asof, s, flag = rate(t.upper())
        print(f'{t.upper()} asof {asof} forward63 {s[63]} forward21 {s[21]} ref {M["ref_date"]}{flag}')
