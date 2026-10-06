import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
def nw_t(x, lag):
    x=np.asarray(x,float); x=x[~np.isnan(x)]; n=len(x); m=x.mean(); e=x-m; v=(e@e)/n
    for l in range(1,lag+1): v+=2*(1-l/(lag+1))*(e[l:]@e[:-l])/n
    return m/np.sqrt(v/n)
sec = pd.concat([pd.read_pickle('../sector.pkl'), pd.read_pickle('../u6/sector.pkl')]); sec = sec[~sec.index.duplicated()]
rows=[]
for h in (21,63):
    tgt=f'fwd{h}_ra'; ex=f'fwd{h}_spy'; lag=h//5
    for src, path in (('pooled-model','oos_h%d.pkl'%h), ('sector-neutral-model','oos_sn_h%d.pkl'%h)):
        O = pd.read_pickle(path); O = O[O['ridge'].notna() & O[tgt].notna()].copy()
        O['sector'] = O.index.get_level_values('ticker').map(sec).fillna('Other')
        key = [O.index.get_level_values('date'), O['sector']]
        O['t_sn'] = O.groupby(key)[tgt].rank(pct=True)
        O['ex_dm'] = O[ex] - O.groupby(key)[ex].transform('mean'); O['ra_dm'] = O[tgt] - O.groupby(key)[tgt].transform('mean')
        for model in ('ridge','lgbm','blend'):
            O['p_sn'] = O.groupby(key)[model].rank(pct=True)
            for univ in ('ALL','LC','SM'):
                S = O if univ=='ALL' else O[O.univ==univ]
                gd = S.groupby(level='date')
                ic = gd.apply(lambda z: z['p_sn'].corr(z['t_sn']))
                top = S[S.p_sn>=0.9]; bot = S[S.p_sn<=0.1]
                tr = top.groupby(level='date')['ra_dm'].mean(); te = top.groupby(level='date')['ex_dm'].mean(); be = bot.groupby(level='date')['ex_dm'].mean()
                rows.append(dict(h=h, src=src, model=model, univ=univ, IC=ic.mean(), t=nw_t(ic,lag), ic_hit=(ic>0).mean(),
                    top_ra_vs_sector=tr.mean(), t_top=nw_t(tr,lag), top_ex_vs_sector_ann=te.mean()*252/h, bot_ex_vs_sector_ann=be.mean()*252/h,
                    **({f'IC_{y}':v for y,v in ic.groupby(ic.index.year).mean().items()} if univ=='ALL' else {})))
R = pd.DataFrame(rows).set_index(['h','src','model','univ'])
pd.set_option('display.width',300); pd.set_option('display.max_columns',30)
print(R.round(3).to_string()); R.to_pickle('eval_sn.pkl')
