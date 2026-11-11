import numpy as np, pandas as pd
X="/work/users/d/y/dyy12/XuLab"; G=f"{X}/complex_screen/pair_motif_composite/motif_valid"
A=pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_all3081.tsv",sep="\t")
K=A[(A.known.astype(str)=="True") & ~A.known_reason.isin(["NEUROD4","STAG2_class"])].copy()
Zd=pd.read_csv(f"{G}/scores_panel_disc.tsv",sep="\t",index_col=0); Zv=pd.read_csv(f"{G}/scores_panel_val.tsv",sep="\t",index_col=0); Fd=pd.read_csv(f"{G}/scores_fisher_disc.tsv",sep="\t",index_col=0)
gen=(Fd>=3).sum(1)>10; tfs=list(Zd.columns)
allp=[(a,b) for i,a in enumerate(tfs) for b in tfs[i+1:]]; ba=np.array([np.nanmax(np.fmin(Zd[a],Zd[b])[~gen]) for a,b in allp])
rows=[]
for r in K.itertuples():
    a,b=r.tf_a,r.tf_b
    if a not in tfs or b not in tfs: continue
    md,mv=np.fmin(Zd[a],Zd[b]),np.fmin(Zv[a],Zv[b]); sel=(md>=2)&(mv>0)&~gen; pb=float(np.nanmax(md[~gen]))
    top=md[sel].sort_values(ascending=False).index[:2]
    rows.append(dict(pair=r.pair,reason=r.known_reason,corr_pct=r.best_pct,corr_ge060=r.best_pct>=.6,n_suggestive=int(sel.sum()),pair_pct=round(float(np.mean(ba<pb)),2),top=";".join(top)))
T=pd.DataFrame(rows).sort_values("corr_pct",ascending=False); pd.set_option("display.width",250); pd.set_option("display.max_colwidth",80)
print(len(T),"known pairs (excl. NEUROD4 / STAG2-class);", int(T.corr_ge060.sum()),"with correlation pct >= 0.60;", int((T.n_suggestive>0).sum()),"with >=1 suggestive shared motif")
print(T.to_string(index=False)); T.to_csv(f"{G}/known_pairs_check.tsv",sep="\t",index=False)
