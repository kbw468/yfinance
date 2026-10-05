import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
P = pd.read_pickle('panel.pkl').replace([np.inf,-np.inf],np.nan)
sec = pd.read_pickle('sector.pkl'); P['sector']=P.index.get_level_values('ticker').map(sec)
Pb = P[P['fwd63'].notna()].copy()
def nw_t(x, lag):
    x=np.asarray(x,float); n=len(x); m=x.mean(); e=x-m; v=(e@e)/n
    for l in range(1,lag+1): v+=2*(1-l/(lag+1))*(e[l:]@e[:-l])/n
    return m/np.sqrt(v/n)
# demean targets by date (universe) and by date-sector
for t in ['fwd21_ra','fwd63_ra','fwd21_spy','fwd63_spy']:
    Pb[t+'_dm'] = Pb[t]-Pb.groupby(level='date')[t].transform('mean')
    Pb[t+'_sdm'] = Pb[t]-Pb.groupby([Pb.index.get_level_values('date'),'sector'])[t].transform('mean')
pct = lambda c: Pb.groupby(level='date')[c].rank(pct=True)
S = {
 'absorption (vol>1.1x & range<0.9x)': (Pb.v_roc_252>1.1)&(Pb.x_range_21_252<0.9),
 'absorption + accum>0.15': (Pb.v_roc_252>1.1)&(Pb.x_range_21_252<0.9)&(Pb.v_updown_share>0.15),
 'accum rising 10d & rv21/63<0.9': (Pb.c_acc_roc10>0.1)&(Pb.rv_21_63<0.9),
 'quiet-day vol>1.2x & rv21/63<0.9': (Pb.v_quiet_ratio>1.2)&(Pb.rv_21_63<0.9),
 'clean compression (rv21/63<.8, cc/park<1, vvol bottom30%)': (Pb.rv_21_63<0.8)&(Pb.x_cc_over_park<1.0)&(pct('x_vol_of_vol')<0.3),
 'coil (range slope<0 & last5 narrowest<0.85)': (Pb.x_range_slope<0)&(Pb.x_last5_narrowest<0.85),
 'buyers close it (clv>0.58)': Pb.x_clv>0.58,
 'upside semivol dom (>1.25) & rv21/63<1': (Pb.x_semivol_ratio>1.25)&(Pb.rv_21_63<1),
 'downside semivol dom (<0.8) & rv21/63<1': (Pb.x_semivol_ratio<0.8)&(Pb.rv_21_63<1),
 'breakout-day vol share>0.3': Pb.v_breakout_share>0.3,
 'overnight share top20% & rv21/63<1': (pct('x_overnight_share')>0.8)&(Pb.rv_21_63<1),
 'gap-driven vol (cc/park>1.3)': Pb.x_cc_over_park>1.3,
 'range-driven vol (cc/park<0.9)': Pb.x_cc_over_park<0.9,
 'rv pct<0.2 (own 252d history)': Pb.x_rv_pct_252<0.2,
 'rv pct>0.8': Pb.x_rv_pct_252>0.8,
 'forced seller (bigdown conc>0.5)': Pb.v_bigdown_conc>0.5,
 'steady participation (top3<0.2 & vol>1x)': (Pb.v_top3_conc<0.2)&(Pb.v_roc_252>1),
 'one-event vol (maxdown var share>0.5)': Pb.x_maxdown_var_share>0.5,
 'updown share>0.3': Pb.v_updown_share>0.3,
 'updown share<-0.3': Pb.v_updown_share<-0.3,
 'vw-ret > simple by 2%+ ': Pb.v_vw_minus_simple>0.02,
 'expansion quadrant (vol>1.3x & |ret21|>8%)': (Pb.v_roc_252>1.3)&(Pb.m_ret21.abs()>0.08),
 'absorb quadrant (vol>1.3x & |ret21|<3%)': (Pb.v_roc_252>1.3)&(Pb.m_ret21.abs()<0.03),
 'acc x volcomp top10%': pct('c_acc_x_volcomp')>0.9,
 'acc x volcomp top10% & sector-neutral': pct('c_acc_x_volcomp')>0.9,
 'turnover top20%': pct('v_dollar_turnover')>0.8,
 'turnover bottom20%': pct('v_dollar_turnover')<0.2,
}
rows=[]
for k,m in S.items():
    sub = Pb[m]
    rec={'screen':k,'avg_n/date':sub.groupby(level='date').size().mean()}
    for t,H in (('fwd21_ra',21),('fwd63_ra',63)):
        col = t+'_sdm'
        bydate = sub.groupby(level='date')[col].mean().dropna()
        rec[f'ex{H}ra']=bydate.mean(); rec[f't{H}']=nw_t(bydate,H//5); rec[f'hit{H}']=(bydate>0).mean()
    # raw excess vs SPY 63d, annualised-ish
    bydate = sub.groupby(level='date')['fwd63_spy_sdm'].mean().dropna()
    rec['ex63_vsSPY_sn']=bydate.mean()
    rows.append(rec)
R = pd.DataFrame(rows).set_index('screen')
pd.set_option('display.width',250)
print(R.round(3).sort_values('t63',ascending=False).to_string())
