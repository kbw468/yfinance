import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
P = pd.read_pickle('panel.pkl').replace([np.inf,-np.inf],np.nan)
sec = pd.read_pickle('sector.pkl')
P['sector'] = P.index.get_level_values('ticker').map(sec)
feats = [c for c in P.columns if not c.startswith('fwd') and c!='sector' and not c.startswith('rel_')]
Pb = P[P['fwd21'].notna()].copy()

def nw_t(x, lag):
    x = np.asarray(x); n=len(x); m=x.mean(); e=x-m
    v = (e@e)/n
    for l in range(1,lag+1):
        w = 1-l/(lag+1); v += 2*w*(e[l:]@e[:-l])/n
    return m/np.sqrt(v/n)

# sector-neutral rank transform
def sn_rank(df, cols):
    return df.groupby([df.index.get_level_values('date'), df['sector']])[cols].rank(pct=True)
R_raw = Pb.groupby(level='date')[feats].rank(pct=True)
R_sn  = sn_rank(Pb, feats)
T_rank = Pb.groupby(level='date')[['fwd21_ra','fwd63_ra']].rank(pct=True)

def ic_series(R, f, tgt):
    g = pd.concat([R[f], T_rank[tgt]], axis=1).dropna()
    return g.groupby(level='date').apply(lambda x: x.iloc[:,0].corr(x.iloc[:,1]) if len(x)>50 else np.nan).dropna()

rows=[]
for f in feats:
    rec={'feature':f}
    for nm,R in (('raw',R_raw),('sn',R_sn)):
        for tgt,H in (('fwd21_ra',21),('fwd63_ra',63)):
            ic = ic_series(R,f,tgt)
            rec[f'{nm}_IC{H}']=ic.mean(); rec[f'{nm}_t{H}']=nw_t(ic, H//5)
    rows.append(rec)
res = pd.DataFrame(rows).set_index('feature')
res['maxT'] = res[[c for c in res.columns if '_t' in c]].abs().max(axis=1)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
print(res.sort_values('maxT',ascending=False).round(3).to_string())
res.to_pickle('ic2.pkl')
R_sn.to_pickle('R_sn.pkl'); R_raw.to_pickle('R_raw.pkl')
