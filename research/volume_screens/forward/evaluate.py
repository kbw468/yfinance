import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
def nw_t(x, lag):
    x=np.asarray(x,float); x=x[~np.isnan(x)]; n=len(x); m=x.mean(); e=x-m; v=(e@e)/n
    for l in range(1,lag+1): v+=2*(1-l/(lag+1))*(e[l:]@e[:-l])/n
    return m/np.sqrt(v/n)
rows=[]; dec={}
for h in (21,63):
    O = pd.read_pickle(f'oos_h{h}.pkl'); O = O[O['ridge'].notna()]
    tgt=f'fwd{h}_ra'; ex=f'fwd{h}_spy'
    O = O[O[tgt].notna()]
    lag=h//5
    for model in ('sign','ridge','lgbm','blend'):
        for univ in ('ALL','LC','SM'):
            S = O if univ=='ALL' else O[O.univ==univ]
            g = S.groupby(level='date')
            ic = g.apply(lambda z: z[model].corr(z[tgt], method='spearman'))
            def q(z):
                k = pd.qcut(z[model].rank(method='first'), 10, labels=False)
                return pd.Series({'d10_ra':z[tgt][k==9].mean(),'d1_ra':z[tgt][k==0].mean(),'u_ra':z[tgt].mean(),
                                  'd10_ex':z[ex][k==9].mean(),'d1_ex':z[ex][k==0].mean(),'u_ex':z[ex].mean(),
                                  'd10_hit':(z[ex][k==9]>z[ex].median()).mean()})
            Q = g.apply(q)
            rec = dict(h=h, model=model, univ=univ, IC=ic.mean(), t=nw_t(ic,lag), ic_hit=(ic>0).mean(),
                       d10_ra=Q.d10_ra.mean(), d1_ra=Q.d1_ra.mean(), u_ra=Q.u_ra.mean(), t_d10_vs_u=nw_t(Q.d10_ra-Q.u_ra,lag),
                       d10_ex_ann=Q.d10_ex.mean()*252/h, d1_ex_ann=Q.d1_ex.mean()*252/h, u_ex_ann=Q.u_ex.mean()*252/h,
                       d10_beat_median=Q.d10_hit.mean())
            rows.append(rec)
            if univ=='ALL':
                yr = ic.groupby(ic.index.year).mean()
                for y,v in yr.items(): rows[-1][f'IC_{y}']=v
R = pd.DataFrame(rows).set_index(['h','model','univ'])
pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
print(R.round(3).to_string())
R.to_pickle('eval.pkl')
