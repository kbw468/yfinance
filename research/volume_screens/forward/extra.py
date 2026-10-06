# Extra price/volume/vol features, vectorised, sampled at each panel's rebalance dates.
import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings('ignore')
def extra(px_path, panel_path, out):
    d = pd.read_pickle(px_path).iloc[:-1]
    O,H,L,C,V = [d[k] for k in ['Open','High','Low','Close','Volume']]
    V = V.replace(0,np.nan)
    P = pd.read_pickle(panel_path)
    dates = P.index.get_level_values('date').unique(); names = P.index.get_level_values('ticker').unique()
    keep = [c for c in names if c in C.columns]
    O,H,L,C,V = [x[keep+['SPY']] for x in (O,H,L,C,V)]
    r = np.log(C/C.shift(1)); dv = C*V
    m = lambda x,w: x.rolling(w, min_periods=int(w*0.8)).mean()
    s = lambda x,w: x.rolling(w, min_periods=int(w*0.8)).std()
    F = {}
    F['e_ret5'] = np.log(C/C.shift(5))
    F['e_ret126'] = np.log(C/C.shift(126))
    F['e_mom12_1'] = np.log(C.shift(21)/C.shift(252))
    F['e_dist_52wlo'] = C/L.rolling(252).min()-1
    F['e_max21'] = r.rolling(21).max()
    F['e_min21'] = r.rolling(21).min()
    F['e_skew63'] = r.rolling(63).skew()
    F['e_kurt63'] = r.rolling(63).kurt()
    F['e_logadv'] = np.log(m(dv,21))
    F['e_amihud'] = np.log(m((r.abs()/dv).replace([np.inf],np.nan),63)*1e9)
    F['e_vol_5_21'] = m(V,5)/m(V,21)
    F['e_vol_63_252'] = m(V,63)/m(V,252)
    F['e_rv63'] = s(r,63)*np.sqrt(252)
    F['e_rv252'] = s(r,252)*np.sqrt(252)
    rs = r['SPY']
    cov = r.rolling(252, min_periods=200).cov(rs); var = rs.rolling(252, min_periods=200).var()
    beta = cov.div(var, axis=0); F['e_beta'] = beta
    resid = r - beta.mul(rs, axis=0)
    F['e_ivol63'] = s(resid,63)*np.sqrt(252)
    F['e_ivol_share'] = (s(resid,63)/s(r,63))**2
    on = np.log(O/C.shift(1)); idr = np.log(C/O)
    F['e_on_ret21'] = on.rolling(21).sum(); F['e_id_ret21'] = idr.rolling(21).sum()
    F['e_on_ret63'] = on.rolling(63).sum(); F['e_id_ret63'] = idr.rolling(63).sum()
    F['e_logprice'] = np.log(C)
    F['e_hl_spread'] = m(2*(np.sqrt(np.exp(np.log(H/L)))-1)/(1+np.sqrt(np.exp(np.log(H/L)))),21)  # range-based spread proxy
    F['e_ret21_x_vol'] = np.log(C/C.shift(21)) * (m(V,21)/m(V,252))   # high-volume return (volume-confirmed move)
    F['e_upday_frac'] = (r>0).astype(float).rolling(21).mean()
    F['e_rel_ret63'] = np.log(C/C.shift(63)).sub(np.log(C['SPY']/C['SPY'].shift(63)), axis=0)
    out_parts=[]
    for k,v in F.items():
        x = v.reindex(dates).drop(columns='SPY').stack(future_stack=True); x.name=k; out_parts.append(x)
    E = pd.concat(out_parts, axis=1); E.index.names=['date','ticker']
    E = E.reindex(P.index)
    E.to_pickle(out); print(out, E.shape, E.notna().mean().round(2).min())
extra(sys.argv[1], sys.argv[2], sys.argv[3])
