import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
P = pd.read_pickle('panel.pkl').replace([np.ininf if False else np.inf,-np.inf],np.nan)
sec = pd.read_pickle('sector.pkl'); P['sector']=P.index.get_level_values('ticker').map(sec)
Pb = P[P['fwd63'].notna()].copy()
def nw_t(x, lag):
    x=np.asarray(x,float); n=len(x); m=x.mean(); e=x-m; v=(e@e)/n
    for l in range(1,lag+1): v+=2*(1-l/(lag+1))*(e[l:]@e[:-l])/n
    return m/np.sqrt(v/n) if v>0 else np.nan
for t in ['fwd21_ra','fwd63_ra','fwd63_spy']:
    Pb[t+'_sdm'] = Pb[t]-Pb.groupby([Pb.index.get_level_values('date'),'sector'])[t].transform('mean')
gd = Pb.groupby(level='date')
Pb['acc_vc_fixed'] = Pb.v_updown_share * (1-Pb.rv_21_63).clip(0,1)
Pb['acc_vc_rank']  = gd['v_updown_share'].rank(pct=True) * (1-gd['rv_21_63'].rank(pct=True))
Pb['acc_only_pos_comp'] = Pb.v_updown_share.where(Pb.rv_21_63<1)   # updown share, only among compressing names
Pb['acc_vc_old'] = Pb.c_acc_x_volcomp
d = Pb.index.get_level_values('date')
periods = {'20-22': d<'2023-01-01', '23-24': (d>='2023-01-01')&(d<'2024-10-01'), '24-26': d>='2024-10-01', 'ALL': d==d}
rows=[]
for f in ['acc_vc_old','acc_vc_fixed','acc_vc_rank','acc_only_pos_comp']:
    pr = gd[f].rank(pct=True)
    top = pr>0.9
    for pn,pm in periods.items():
        sub = Pb[top&pm]; rec={'feat':f,'period':pn,'n/date':sub.groupby(level='date').size().mean()}
        for t,H in (('fwd21_ra',21),('fwd63_ra',63),('fwd63_spy',63)):
            b = sub.groupby(level='date')[t+'_sdm'].mean().dropna(); rec[f'ex_{t}']=b.mean(); rec[f't_{t}']=nw_t(b,H//5)
        # also rank IC vs fwd63_ra over period
        g = Pb[pm][[f,'fwd63_ra']].dropna()
        ic = g.groupby(level='date').apply(lambda x: x[f].corr(x['fwd63_ra'],method='spearman'))
        rec['IC63']=ic.mean(); rec['tIC63']=nw_t(ic,12)
        rows.append(rec)
R=pd.DataFrame(rows).set_index(['feat','period']); pd.set_option('display.width',250)
print(R.round(3).to_string())
# snapshot under fixed definition
s = pd.read_csv('volume_vol_screens_2026-10-02.csv', index_col=0)
s['acc_vc_fixed'] = s.v_updown_share*(1-s.rv_21_63).clip(0,1)
s['acc_vc_rank'] = s.v_updown_share.rank(pct=True)*(1-s.rv_21_63.rank(pct=True))
s['tdf']=s.acc_vc_fixed.rank(pct=True)>0.9; s['tdr']=s.acc_vc_rank.rank(pct=True)>0.9
c=['close','sector','quadrant','vol_type','v_updown_share','v_breakout_share','v_vw_minus_simple','v_roc_252','v_quiet_ratio','rv21','rv_21_63','x_rv_pct_252','x_cc_over_park','x_overnight_share','x_clv','x_semivol_ratio','x_maxdown_var_share','acc_vc_fixed','acc_vc_rank','m_ret21','m_ret63','m_dist_52whi','hi21','lo21','lo10','hi252','adv21_$M','short_float']
print('\n=== FIXED top decile (both defs flagged: tdf/tdr), ex one-event ===')
sub = s[(s.tdf|s.tdr)&~s.one_event].sort_values('acc_vc_fixed',ascending=False)
sub['both']=sub.tdf&sub.tdr
pd.set_option('display.max_rows',500)
print(sub[['both']+c].round(3).to_string())
sub.to_csv('top_decile_acc_volcomp.csv')
