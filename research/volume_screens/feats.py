import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
d = pd.read_pickle('px.pkl')
meta = pd.read_pickle('meta.pkl')
# drop partial last bar (intraday Monday)
d = d.iloc[:-1]
O,H,L,C,V = [d[k] for k in ['Open','High','Low','Close','Volume']]
drop = ['FDXF','HONA','Q','VYLR']
O,H,L,C,V = [x.drop(columns=drop, errors='ignore') for x in (O,H,L,C,V)]
V = V.replace(0, np.nan)
r = np.log(C/C.shift(1))
ar = r.abs()
up = (r>0).astype(float); dn = (r<0).astype(float)
W=21; LONG=252

def rsum(x,w=W): return x.rolling(w, min_periods=int(w*0.8)).sum()
def rmean(x,w=W): return x.rolling(w, min_periods=int(w*0.8)).mean()
def rstd(x,w=W): return x.rolling(w, min_periods=int(w*0.8)).std()

F = {}
# ---------- VOLUME ----------
vt = rsum(V)
F['v_updown_share'] = (rsum(V*up) - rsum(V*dn)) / vt
prior_hi = H.shift(1).rolling(W).max()
bo = (C > prior_hi).astype(float)
F['v_breakout_share'] = rsum(V*bo) / vt
F['v_vw_minus_simple'] = (rsum(V*r)/vt - rmean(r)) * W
ret21 = np.log(C/C.shift(W))
vroc = rmean(V) / rmean(V, LONG)
F['v_roc_252'] = vroc
F['v_roc_per_abs_price'] = (vroc - 1) / (ret21.abs() + 0.02)   # absorption/distribution vs expansion
F['v_down_share'] = rsum(V*dn)/vt
# down-volume concentrated in 3 biggest down days
def top3_share(a):
    a = a[~np.isnan(a)]
    if len(a) < 10 or a.sum()<=0: return np.nan
    return np.sort(a)[-3:].sum()/a.sum()
dnv = (V*dn).fillna(0)
# biggest down days by return magnitude: rank by r within window -> approximate by vol on the 3 most-negative-return days
def big_down_share(rw, vw):
    pass
# vectorised: for each window, volume on 3 most negative returns / total down volume
def roll2(fn, A, B, w=W):
    out = pd.DataFrame(np.nan, index=A.index, columns=A.columns)
    a = A.values; b = B.values
    for i in range(w-1, len(A)):
        out.iloc[i] = fn(a[i-w+1:i+1], b[i-w+1:i+1])
    return out
def f_bigdown(rw, vw):
    res = np.full(rw.shape[1], np.nan)
    for j in range(rw.shape[1]):
        rr = rw[:,j]; vv = vw[:,j]
        m = ~np.isnan(rr) & ~np.isnan(vv)
        rr=rr[m]; vv=vv[m]
        if len(rr)<15: continue
        dmask = rr<0
        if dmask.sum()<3 or vv[dmask].sum()<=0: continue
        idx = np.argsort(rr)[:3]
        res[j] = vv[idx].sum()/vv[dmask].sum()
    return res
F['v_bigdown_conc'] = roll2(f_bigdown, r, V)
quiet = (ar < 0.005).astype(float)
F['v_quiet_ratio'] = (rsum(V*quiet)/rsum(quiet).replace(0,np.nan)) / rmean(V, LONG)
F['v_top3_conc'] = V.rolling(W).apply(top3_share, raw=True)
mcap = pd.to_numeric(meta['Market Cap'], errors='coerce').reindex(C.columns) * 1e6
dv = C*V
shares = mcap / C.iloc[-1]
F['v_dollar_turnover'] = rmean(V) / shares      # share turnover of float proxy (shares held const)
F['v_dturn_roc'] = rmean(dv)/rmean(dv,LONG)
F['v_absret_vol_corr'] = ar.rolling(W).corr(V)
F['v_ret_vol_corr'] = r.rolling(W).corr(V)

# ---------- VOLATILITY ----------
rv = lambda w: rstd(r,w)*np.sqrt(252)
rv5, rv10, rv21, rv63, rv126, rv252 = rv(5), rv(10), rv(21), rv(63), rv(126), rv(252)
F['rv_5_21']=rv5/rv21; F['rv_10_21']=rv10/rv21; F['rv_21_63']=rv21/rv63
F['rv_21_126']=rv21/rv126; F['rv_21_252']=rv21/rv252; F['rv_63_252']=rv63/rv252
F['rv21']=rv21
park = np.sqrt(rmean(np.log(H/L)**2)/(4*np.log(2)))*np.sqrt(252)
gk = np.sqrt(rmean(0.5*np.log(H/L)**2 - (2*np.log(2)-1)*np.log(C/O)**2))*np.sqrt(252)
F['x_cc_over_park'] = rv21/park
F['x_cc_over_gk'] = rv21/gk
on = np.log(O/C.shift(1)); idr = np.log(C/O)
F['x_overnight_share'] = rsum(on**2)/(rsum(on**2)+rsum(idr**2))
F['x_overnight_share_chg'] = F['x_overnight_share'] - F['x_overnight_share'].shift(W)
upv = np.sqrt(rsum((r.clip(lower=0))**2)/W); dnv_ = np.sqrt(rsum((r.clip(upper=0))**2)/W)
F['x_semivol_ratio'] = upv/dnv_
F['x_vol_of_vol'] = rstd(rv5)/rv21
rng = (H-L)/C
t = np.arange(W); tw = (t - t.mean())/((t-t.mean())**2).sum()
F['x_range_slope'] = rng.rolling(W).apply(lambda a: np.dot(tw,a), raw=True) / rmean(rng)
F['x_last5_narrowest'] = rmean(rng,5) / rmean(rng)
F['x_clv'] = rmean(((C-L)/(H-L)).replace([np.inf,-np.inf],np.nan))
r2 = r**2
F['x_maxdown_var_share'] = (r.clip(upper=0)**2).rolling(W).max()/rsum(r2)
F['x_rv21_ex_event'] = np.sqrt((rsum(r2) - (r.clip(upper=0)**2).rolling(W).max())/(W-1))*np.sqrt(252)
F['x_rv21_ex_event_21_252'] = F['x_rv21_ex_event']/rv252
F['x_rv_pct_252'] = rv21.rolling(LONG, min_periods=200).rank(pct=True)
F['x_rv_pct_504'] = rv21.rolling(504, min_periods=350).rank(pct=True)
F['x_range_21_252'] = rmean(rng)/rmean(rng,LONG)

# ---------- COMBINATIONS ----------
F['c_absorption'] = vroc / F['x_range_21_252']           # vol up, range down -> high
F['c_absorb_flag'] = ((vroc>1)&(F['x_range_21_252']<1)).astype(float)
dacc = F['v_updown_share'] - F['v_updown_share'].shift(10)
F['c_acc_roc10'] = dacc
F['c_acc_roc10_x_volcomp'] = dacc * (F['rv_21_63']<1)
F['c_acc_x_volcomp'] = F['v_updown_share'] * (1 - F['rv_21_63'].clip(upper=1.5))

# ---------- MOMENTUM controls ----------
F['m_ret21'] = ret21
F['m_ret63'] = np.log(C/C.shift(63))
F['m_ret252_ex21'] = np.log(C.shift(21)/C.shift(252))
F['m_dist_52whi'] = C/H.rolling(252).max() - 1

# ---------- SPY-relative ----------
spy_rel = ['v_updown_share','v_breakout_share','x_clv','x_semivol_ratio','rv_21_63','rv_21_252','v_roc_252','x_cc_over_park']
for k in spy_rel:
    F['rel_'+k] = F[k].sub(F[k]['SPY'], axis=0)

# ---------- FORWARD TARGETS ----------
T = {}
for h in (21,63):
    fr = np.log(C.shift(-h)/C)
    fv = r.rolling(h).std().shift(-h)*np.sqrt(h)
    T[f'fwd{h}'] = fr
    T[f'fwd{h}_spy'] = fr.sub(fr['SPY'],axis=0)
    T[f'fwd{h}_ra'] = fr/fv                 # per-name horizon Sharpe
    T[f'fwd{h}_ra_x'] = fr.sub(fr['SPY'],axis=0)/fv
    T[f'fwd{h}_maxdd'] = (C.rolling(h).min().shift(-h)/C - 1)

# ---------- STACK ----------
dates = C.index
rebal = dates[LONG::5]                 # weekly, after 252d warm-up
rebal = rebal.append(pd.Index([dates[-1]])).unique()
def stack(D, name, idx):
    s = D.loc[idx].drop(columns='SPY').stack(future_stack=True); s.name=name; return s
Fp = pd.concat([stack(v,k,rebal) for k,v in F.items()], axis=1)
Tp = pd.concat([stack(v,k,rebal) for k,v in T.items()], axis=1)
P = Fp.join(Tp); P.index.names=['date','ticker']
P.to_pickle('panel.pkl')
# current snapshot (full daily, last date) incl. SPY
snap = pd.DataFrame({k:v.iloc[-1] for k,v in F.items()})
snap['close']=C.iloc[-1]; snap['hi21']=H.iloc[-21:].max(); snap['lo21']=L.iloc[-21:].min()
snap['hi63']=H.iloc[-63:].max(); snap['lo63']=L.iloc[-63:].min(); snap['hi252']=H.iloc[-252:].max()
snap['lo10']=L.iloc[-10:].min(); snap['hi10']=H.iloc[-10:].max()
snap['adv21_$M']=rmean(dv).iloc[-1]/1e6
snap.to_pickle('snap.pkl')
meta['Sector'].to_pickle('sector.pkl')
print('asof', C.index[-1].date(), 'panel', P.shape, 'rebal dates', len(rebal), rebal[0].date(), rebal[-2].date())
print(P.describe().T[['count','mean','50%']].to_string())
