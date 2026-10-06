"""Single-name feature builder. Formulas are identical to the training panel (feats.py + extra.py).
Input: wide OHLCV frames (date x ticker) that include a 'SPY' column; shares outstanding per ticker.
Output: DataFrame (ticker x feature) at the last date."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
W, LONG = 21, 252
def features(O,H,L,C,V, shares):
    V = V.replace(0, np.nan)
    r = np.log(C/C.shift(1)); ar = r.abs()
    up = (r>0).astype(float); dn = (r<0).astype(float)
    rsum = lambda x,w=W: x.rolling(w, min_periods=int(w*0.8)).sum()
    rmean = lambda x,w=W: x.rolling(w, min_periods=int(w*0.8)).mean()
    rstd = lambda x,w=W: x.rolling(w, min_periods=int(w*0.8)).std()
    F = {}
    vt = rsum(V)
    F['v_updown_share'] = (rsum(V*up) - rsum(V*dn)) / vt
    bo = (C > H.shift(1).rolling(W).max()).astype(float)
    F['v_breakout_share'] = rsum(V*bo) / vt
    F['v_vw_minus_simple'] = (rsum(V*r)/vt - rmean(r)) * W
    ret21 = np.log(C/C.shift(W)); vroc = rmean(V) / rmean(V, LONG)
    F['v_roc_252'] = vroc
    F['v_roc_per_abs_price'] = (vroc - 1) / (ret21.abs() + 0.02)
    F['v_down_share'] = rsum(V*dn)/vt
    bd = {}
    for c in C.columns:
        rr = r[c].values[-W:]; vv = V[c].values[-W:]; m = ~np.isnan(rr) & ~np.isnan(vv); rr, vv = rr[m], vv[m]
        dmask = rr < 0
        bd[c] = np.nan if (len(rr)<15 or dmask.sum()<3 or vv[dmask].sum()<=0) else vv[np.argsort(rr)[:3]].sum()/vv[dmask].sum()
    F['v_bigdown_conc'] = pd.DataFrame([bd], index=[C.index[-1]]).reindex(columns=C.columns)
    quiet = (ar < 0.005).astype(float)
    F['v_quiet_ratio'] = (rsum(V*quiet)/rsum(quiet).replace(0,np.nan)) / rmean(V, LONG)
    def top3(a):
        a = a[~np.isnan(a)]
        return np.nan if (len(a)<10 or a.sum()<=0) else np.sort(a)[-3:].sum()/a.sum()
    F['v_top3_conc'] = V.iloc[-W-5:].rolling(W).apply(top3, raw=True)
    dv = C*V
    F['v_dollar_turnover'] = rmean(V) / pd.Series(shares).reindex(C.columns)
    F['v_dturn_roc'] = rmean(dv)/rmean(dv,LONG)
    F['v_absret_vol_corr'] = ar.rolling(W).corr(V)
    F['v_ret_vol_corr'] = r.rolling(W).corr(V)
    rv = lambda w: rstd(r,w)*np.sqrt(252)
    rv5, rv10, rv21, rv63, rv126, rv252 = rv(5), rv(10), rv(21), rv(63), rv(126), rv(252)
    F['rv_5_21']=rv5/rv21; F['rv_10_21']=rv10/rv21; F['rv_21_63']=rv21/rv63
    F['rv_21_126']=rv21/rv126; F['rv_21_252']=rv21/rv252; F['rv_63_252']=rv63/rv252; F['rv21']=rv21
    park = np.sqrt(rmean(np.log(H/L)**2)/(4*np.log(2)))*np.sqrt(252)
    gk = np.sqrt(rmean(0.5*np.log(H/L)**2 - (2*np.log(2)-1)*np.log(C/O)**2))*np.sqrt(252)
    F['x_cc_over_park'] = rv21/park; F['x_cc_over_gk'] = rv21/gk
    on = np.log(O/C.shift(1)); idr = np.log(C/O)
    F['x_overnight_share'] = rsum(on**2)/(rsum(on**2)+rsum(idr**2))
    F['x_overnight_share_chg'] = F['x_overnight_share'] - F['x_overnight_share'].shift(W)
    F['x_semivol_ratio'] = np.sqrt(rsum((r.clip(lower=0))**2)/W)/np.sqrt(rsum((r.clip(upper=0))**2)/W)
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
    F['c_absorption'] = vroc / F['x_range_21_252']
    F['c_absorb_flag'] = ((vroc>1)&(F['x_range_21_252']<1)).astype(float)
    dacc = F['v_updown_share'] - F['v_updown_share'].shift(10)
    F['c_acc_roc10'] = dacc
    F['c_acc_roc10_x_volcomp'] = dacc * (F['rv_21_63']<1)
    F['c_acc_x_volcomp'] = F['v_updown_share'] * (1 - F['rv_21_63'].clip(upper=1.5))
    F['m_ret21'] = ret21; F['m_ret63'] = np.log(C/C.shift(63))
    F['m_ret252_ex21'] = np.log(C.shift(21)/C.shift(252)); F['m_dist_52whi'] = C/H.rolling(252).max() - 1
    # extras
    m = lambda x,w: x.rolling(w, min_periods=int(w*0.8)).mean()
    s = lambda x,w: x.rolling(w, min_periods=int(w*0.8)).std()
    F['e_ret5'] = np.log(C/C.shift(5)); F['e_ret126'] = np.log(C/C.shift(126))
    F['e_mom12_1'] = np.log(C.shift(21)/C.shift(252)); F['e_dist_52wlo'] = C/L.rolling(252).min()-1
    F['e_max21'] = r.rolling(21).max(); F['e_min21'] = r.rolling(21).min()
    F['e_skew63'] = r.rolling(63).skew(); F['e_kurt63'] = r.rolling(63).kurt()
    F['e_logadv'] = np.log(m(dv,21))
    F['e_amihud'] = np.log(m((r.abs()/dv).replace([np.inf],np.nan),63)*1e9)
    F['e_vol_5_21'] = m(V,5)/m(V,21); F['e_vol_63_252'] = m(V,63)/m(V,252)
    F['e_rv63'] = s(r,63)*np.sqrt(252); F['e_rv252'] = s(r,252)*np.sqrt(252)
    rs = r['SPY']
    beta = r.rolling(252, min_periods=200).cov(rs).div(rs.rolling(252, min_periods=200).var(), axis=0)
    F['e_beta'] = beta; resid = r - beta.mul(rs, axis=0)
    F['e_ivol63'] = s(resid,63)*np.sqrt(252); F['e_ivol_share'] = (s(resid,63)/s(r,63))**2
    F['e_on_ret21'] = on.rolling(21).sum(); F['e_id_ret21'] = idr.rolling(21).sum()
    F['e_on_ret63'] = on.rolling(63).sum(); F['e_id_ret63'] = idr.rolling(63).sum()
    F['e_logprice'] = np.log(C)
    F['e_hl_spread'] = m(2*(np.sqrt(np.exp(np.log(H/L)))-1)/(1+np.sqrt(np.exp(np.log(H/L)))),21)
    F['e_ret21_x_vol'] = np.log(C/C.shift(21)) * (m(V,21)/m(V,252))
    F['e_upday_frac'] = (r>0).astype(float).rolling(21).mean()
    F['e_rel_ret63'] = np.log(C/C.shift(63)).sub(np.log(C['SPY']/C['SPY'].shift(63)), axis=0)
    out = pd.DataFrame({k: v.iloc[-1] for k,v in F.items()})
    return out.replace([np.inf,-np.inf], np.nan)
