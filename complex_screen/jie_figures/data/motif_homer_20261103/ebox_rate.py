import csv,glob,os,re,sys
import pandas as pd,numpy as np
os.chdir(sys.argv[1])
HN=["E-box","TFE3","c-Myc","n-Myc","Max","USF1","Usf2","CLOCK","BMAL1","NPAS","bHLHE40","bHLHE41","MNT","MITF"]
AN=["MLX","MAX","MYCN","MYC","USF1","USF2","TFEB","TFE3","MITF","BHE40","BHE41"]
rows=[]
for run in("motif","motif_bins"):
  for bg in("homer","homer_ocr"):
    for f in glob.glob(f"{run}/*/{bg}/knownResults.txt"):
        t=f.split("/")[1]; R=list(csv.reader(open(f),delimiter="\t"))[1:]; n=len(R)
        hit=[(i+1,r[0].split("(")[0],float(r[4]),float(r[3]),r[6],r[8],r[1]) for i,r in enumerate(R) if r[0].split("(")[0] in HN and "bHLH" in r[0]]
        sig=[h for h in hit if h[2]<.05]; best=min(hit,key=lambda h:h[3]) if hit else None
        srep=[(i+1,float(r[4])) for i,r in enumerate(R) if r[0].startswith("Srebp")]
        rows.append(dict(kind="homer",run=run,bg=bg,tf=t,n_sig=len(sig),n=len(hit),best=best[1] if best else "",best_q=best[2] if best else 1,best_rank=best[0] if best else 999,best_T=best[4] if best else "",best_B=best[5] if best else "",sre_q=min(x[1] for x in srep)))
  for bg in("ame","ame_ocr"):
    for f in glob.glob(f"{run}/*/{bg}/*/ame.tsv"):
        t=f.split("/")[1]
        if os.path.getsize(f)==0: continue
        try: d=pd.read_csv(f,sep="\t",comment="#")
        except pd.errors.EmptyDataError: continue
        if "motif_ID" not in d: continue
        d=d[d["motif_ID"].notna()]
        d["fam"]=d.motif_ID.str.split("_").str[0]; k=d[d.fam.isin(AN)]
        rows.append(dict(kind="ame",run=run,bg=bg,tf=t,n_sig=len(k),n=len(AN),best=k.fam.iloc[0] if len(k) else "",best_q=k["adj_p-value"].astype(float).min() if len(k) else 1,best_rank=int(k["rank"].iloc[0]) if len(k) else 999,tot=len(d),mlx_model=(k.fam=="MLX").any(),sre=d.fam.isin(["SRBP1","SRBP2"]).any()))
T=pd.DataFrame(rows); T.to_csv(os.path.join(sys.argv[2],"ebox_rate.tsv"),sep="\t",index=False)
for (k,r,b),g in T.groupby(["kind","run","bg"]):
    pos=g[g.n_sig>0]
    m=g[g.tf.isin(["mlx","srebf2"])][["tf","n_sig","best","best_q","best_rank","best_T","best_B"]] if k=="homer" else g[g.tf.isin(["mlx","srebf2"])][["tf","n_sig","best","best_q","best_rank","tot","mlx_model"]]
    print(f"\n{k} {r} {b}: TFs tested {len(g)}; with >=1 E-box family significant {len(pos)} ({len(pos)/len(g):.0%}); >=5 members {(g.n_sig>=5).sum()}; median n_sig {g.n_sig.median()}")
    print(m.to_string(index=False))
    if k=="homer": print("SRE q<.05 TFs:",(g.sre_q<.05).sum(),g[g.tf.isin(['mlx','srebf2'])][['tf','sre_q']].values.tolist())
    print("rank of mlx/srebf2 by best p among TFs:",{t:int((g.sort_values('best_q').tf.values==t).argmax())+1 for t in('mlx','srebf2') if t in g.tf.values},"of",len(g))
