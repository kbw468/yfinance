import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
s = pd.read_pickle('snap.pkl').replace([np.inf,-np.inf],np.nan)
meta = pd.read_pickle('meta.pkl')
spy = s.loc['SPY']; s = s.drop('SPY')
s['sector']=meta['Sector'].reindex(s.index); s['short_float']=meta['Short Float'].reindex(s.index)
s['mcap_$B']=pd.to_numeric(meta['Market Cap'],errors='coerce').reindex(s.index)/1000
# quadrant labels
def quad(r):
    v = r.v_roc_252; p = abs(r.m_ret21)
    if v>1.15 and p<0.03: return 'ABSORB/DISTRIB'
    if v>1.15 and p>=0.08: return 'EXPANSION'
    if v<0.85 and p<0.03: return 'DEAD'
    return 'neutral'
s['quadrant']=s.apply(quad,axis=1)
s['flow'] = np.where(s.v_updown_share>0.15,'ACC',np.where(s.v_updown_share<-0.15,'DIST','flat'))
s['vol_type'] = np.where(s.x_cc_over_park>1.25,'GAP',np.where(s.x_cc_over_park<0.9,'RANGE','mixed'))
s['rel_updown']=s.v_updown_share-spy.v_updown_share
s['rel_rv2163']=s.rv_21_63-spy.rv_21_63
s['absorb_q'] = (s.v_roc_252>1.3)&(s.m_ret21.abs()<0.03)
s['quiet_comp'] = (s.v_quiet_ratio>1.2)&(s.rv_21_63<0.9)
s['acc_vc_top10'] = s.c_acc_x_volcomp.rank(pct=True)>0.9
s['one_event'] = s.x_maxdown_var_share>0.5
s['absorb_accum'] = (s.v_roc_252>1.1)&(s.x_range_21_252<0.9)&(s.v_updown_share>0.15)
s['coil'] = (s.x_range_slope<0)&(s.x_last5_narrowest<0.85)
print('SPY: updown %.2f  rv21/63 %.2f  rv21 %.2f  clv %.2f  volroc %.2f  cc/park %.2f  semivol %.2f  rvpct %.2f' % (spy.v_updown_share,spy.rv_21_63,spy.rv21,spy.x_clv,spy.v_roc_252,spy.x_cc_over_park,spy.x_semivol_ratio,spy.x_rv_pct_252))
print('\nquadrant counts:', s.quadrant.value_counts().to_dict()); print('flow:', s.flow.value_counts().to_dict())
cols=['close','sector','mcap_$B','quadrant','flow','vol_type','v_updown_share','rel_updown','v_breakout_share','v_vw_minus_simple','v_roc_252','v_quiet_ratio','v_top3_conc','v_bigdown_conc','v_dollar_turnover','rv21','rv_21_63','rv_21_252','x_rv_pct_252','x_cc_over_park','x_overnight_share','x_semivol_ratio','x_vol_of_vol','x_range_slope','x_last5_narrowest','x_clv','x_maxdown_var_share','c_absorption','c_acc_x_volcomp','m_ret21','m_ret63','m_dist_52whi','hi21','lo21','hi63','lo63','hi252','lo10','hi10','adv21_$M','short_float']
pd.set_option('display.width',300); pd.set_option('display.max_rows',500); pd.set_option('display.max_columns',60)
for nm in ['absorb_q','quiet_comp','absorb_accum','acc_vc_top10','coil','one_event']:
    sub = s[s[nm]].sort_values('c_acc_x_volcomp',ascending=False)
    print(f'\n=== {nm} ({len(sub)}) ===')
    print(sub[['close','sector','quadrant','flow','vol_type','v_updown_share','v_roc_252','v_quiet_ratio','rv_21_63','x_rv_pct_252','x_cc_over_park','x_clv','x_semivol_ratio','x_range_slope','c_acc_x_volcomp','m_ret21','m_ret63','m_dist_52whi','hi21','lo21','lo10','short_float']].round(3).to_string())
s[cols+['absorb_q','quiet_comp','absorb_accum','acc_vc_top10','coil','one_event']].round(4).to_csv('volume_vol_screens_2026-10-02.csv')
# combined state score = count of positive-evidence states minus avoid
s['state_score']= s.absorb_q.astype(int)+s.quiet_comp.astype(int)+s.acc_vc_top10.astype(int)+s.absorb_accum.astype(int)-s.one_event.astype(int)
print('\n=== multi-state (score>=2) ===')
print(s[s.state_score>=2].sort_values('state_score',ascending=False)[['close','sector','quadrant','flow','vol_type','state_score','v_updown_share','v_roc_252','v_quiet_ratio','rv_21_63','x_rv_pct_252','x_clv','m_ret21','m_ret63','m_dist_52whi','hi21','lo21','lo10','hi252']].round(3).to_string())
