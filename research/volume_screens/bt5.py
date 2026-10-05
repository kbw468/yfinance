import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
P = pd.read_pickle('panel.pkl').replace([np.inf,-np.inf],np.nan)
sec = pd.read_pickle('sector.pkl'); P['sector']=P.index.get_level_values('ticker').map(sec)
Pb = P[P['fwd63'].notna()].copy()
def nw_t(x, lag):
    x=np.asarray(x,float); n=len(x); m=x.mean(); e=x-m; v=(e@e)/n
    for l in range(1,lag+1): v+=2*(1-l/(lag+1))*(e[l:]@e[:-l])/n
    return m/np.sqrt(v/n) if v>0 else np.nan
for t in ['fwd21_ra','fwd63_ra','fwd21_spy','fwd63_spy']:
    Pb[t+'_sdm'] = Pb[t]-Pb.groupby([Pb.index.get_level_values('date'),'sector'])[t].transform('mean')
pct = lambda c: Pb.groupby(level='date')[c].rank(pct=True)
S = {
 'absorb quadrant': (Pb.v_roc_252>1.3)&(Pb.m_ret21.abs()<0.03),
 'quiet-vol>1.2 & rv21/63<0.9': (Pb.v_quiet_ratio>1.2)&(Pb.rv_21_63<0.9),
 'acc x volcomp top10%': pct('c_acc_x_volcomp')>0.9,
 'one-event vol>0.5 (AVOID)': Pb.x_maxdown_var_share>0.5,
 'turnover bottom20%': pct('v_dollar_turnover')<0.2,
 'ret21 bottom20% (reversal)': pct('m_ret21')<0.2,
 'clv bottom20% (reversal)': pct('x_clv')<0.2,
}
d = Pb.index.get_level_values('date')
periods = {'2020-10→2022-12': d<'2023-01-01', '2023-01→2024-09': (d>='2023-01-01')&(d<'2024-10-01'), '2024-10→2026-07': d>='2024-10-01', 'ALL': d==d}
rows=[]
for k,m in S.items():
    for pn,pm in periods.items():
        sub = Pb[m&pm]; rec={'screen':k,'period':pn,'n/date':sub.groupby(level='date').size().mean()}
        for t,H in (('fwd21_ra',21),('fwd63_ra',63),('fwd63_spy',63)):
            b = sub.groupby(level='date')[t+'_sdm'].mean().dropna()
            rec[f'ex_{t}']=b.mean(); rec[f't_{t}']=nw_t(b,H//5)
        rows.append(rec)
R=pd.DataFrame(rows).set_index(['screen','period'])
pd.set_option('display.width',250)
print(R.round(3).to_string())
