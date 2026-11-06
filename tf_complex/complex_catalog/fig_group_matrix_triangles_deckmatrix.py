import numpy as np, json
exec(open("matfig.py").read().split('if __name__=="__main__":')[0])
exec(open("groups_def.py").read())
L,R=load("mats/counts_spqn_pct.tsv"); o=np.arange(len(L))
Ln,Rn=load("mats/noqS.tsv"); on=order_of(Rn); idxmap=[Ln.index(t) for t in L]
off=Rn[~np.eye(len(Ln),dtype=bool)]
stats={}
for key,name,mem in GROUPS:
    pos,rr=draw(L,R,o,mem,f"mat2_{key}.png",0.0,1.0,"Percentile of TF–TF Pearson r",cmap="RdYlBu_r")
    idx=[L.index(m) for m in mem]; s=R[np.ix_(idx,idx)]; mp=s[~np.eye(len(idx),dtype=bool)].mean()
    stats[key]=round(float(mp),2)
    print(key,"runs",rr,"mean pct %.2f"%mp)
json.dump(stats,open("pct_counts.json","w"))
