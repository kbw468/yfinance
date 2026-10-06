import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
s = pd.read_pickle('snap.pkl').replace([np.inf,-np.inf],np.nan)
cg = pd.read_pickle('cg.pkl'); info = pd.read_pickle('info.pkl')
spy = s.loc['SPY']
buys = cg[(cg.active_usd>0)&(cg.sh1>0)]
X = buys.join(s, how='left').join(info, how='left')
X['is_new'] = X.active_pct=='NEW'
X['days_to_earn'] = (pd.to_datetime(X.earnings_date, errors='coerce', utc=True).dt.tz_localize(None) - pd.Timestamp('2026-10-05')).dt.days
X['short_pct'] = pd.to_numeric(X.shortPercentOfFloat, errors='coerce')*100

X['adv_sh'] = X['adv21_$M']*1e6/X.close
X['turnover_pct'] = X.adv_sh / pd.to_numeric(X.floatShares, errors='coerce') * 100
X['cg_sh_days'] = (X.sh1-X.sh0)/X.adv_sh          # CG net share change in days of ADV
X['rel_updown'] = X.v_updown_share - spy.v_updown_share
X['rel_rv2163'] = X.rv_21_63 - spy.rv_21_63
X['pos_in_21'] = (X.close-X.lo21)/(X.hi21-X.lo21)
comp = {
 'acc':      X.v_updown_share>0.10,
 'acc_vSPY': X.rel_updown>0.20,
 'volcomp':  X.rv_21_63<0.90,
 'rangevol': X.x_cc_over_park<1.15,
 'clv':      X.x_clv>0.52,
 'noforced': (X.v_bigdown_conc<0.45)&(X.x_maxdown_var_share<0.40),
 'quietvol': X.v_quiet_ratio>1.0,
 'nearhigh': X.m_dist_52whi>-0.08,
 'coil':     (X.x_last5_narrowest<0.9)|(X.x_range_slope<0),
 'cgconv':   (X['add']>=2)|(X.active_usd>=50),
}
for k,v in comp.items(): X['c_'+k]=v.fillna(False).astype(int)
CC=['c_'+k for k in comp]; X['setup'] = X[CC].sum(axis=1)
def quad(r):
    v=r.v_roc_252; p=abs(r.m_ret21)
    if np.isnan(v): return 'n/a'
    if v>1.15 and p<0.03: return 'ABSORB'
    if v>1.15 and p>=0.08: return 'EXPAND'
    if v<0.85 and p<0.03: return 'DEAD'
    return 'neutral'
X['quadrant']=X.apply(quad,axis=1)
X['flow']=np.where(X.v_updown_share>0.15,'ACC',np.where(X.v_updown_share<-0.15,'DIST','flat'))
X['voltype']=np.where(X.x_cc_over_park>1.25,'GAP',np.where(X.x_cc_over_park<0.9,'RANGE','mixed'))
X['flags'] = X[CC].apply(lambda r: ''.join('1' if v else '.' for v in r), axis=1)
cols=['setup','flags','close','sector','active_usd','active_pct','add','own','is_new','cg_sh_days','quadrant','flow','voltype','v_updown_share','rel_updown','v_breakout_share','v_vw_minus_simple','v_roc_252','v_quiet_ratio','v_top3_conc','v_bigdown_conc','rv21','rv_21_63','x_rv_pct_252','x_cc_over_park','x_overnight_share','x_semivol_ratio','x_vol_of_vol','x_range_slope','x_last5_narrowest','x_clv','x_maxdown_var_share','m_ret21','m_ret63','m_dist_52whi','pos_in_21','hi21','lo21','lo10','hi252','adv21_$M','turnover_pct','short_pct','earnings_date','days_to_earn']
X = X.sort_values(['setup','v_updown_share'],ascending=False)
X[cols].round(3).to_csv('cg_buys_setup_2026-10-05.csv')
pd.set_option('display.width',400); pd.set_option('display.max_rows',300); pd.set_option('display.max_columns',80)
print('SPY: updown %.2f rv21/63 %.2f rv21 %.2f clv %.2f volroc %.2f cc/park %.2f'%(spy.v_updown_share,spy.rv_21_63,spy.rv21,spy.x_clv,spy.v_roc_252,spy.x_cc_over_park))
print('flags order: acc accSPY volcomp rangevol clv noforced quietvol nearhigh coil cgconv')
print(X[['setup','flags','close','sector','active_usd','active_pct','add','own','cg_sh_days','quadrant','flow','voltype','v_updown_share','v_breakout_share','v_roc_252','v_quiet_ratio','v_bigdown_conc','rv_21_63','x_rv_pct_252','x_cc_over_park','x_clv','x_semivol_ratio','x_maxdown_var_share','m_ret21','m_ret63','m_dist_52whi','hi21','lo10','hi252','short_pct','earnings_date']].round(2).to_string())
