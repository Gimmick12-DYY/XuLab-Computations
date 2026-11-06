import numpy as np, csv, collections, json, sys
r=list(csv.reader(open("../data/TF1000cells.meta.csv")));i=r[0].index("TF")
cc=collections.Counter(x[i].lower() for x in r[1:])
def load(p):
    h=open(p).readline().rstrip("\n").split("\t")[1:]
    return h,np.loadtxt(p,delimiter="\t",skiprows=1,usecols=range(1,len(h)+1))
MATS={"all":"results_chromhmm_elements_counts_spqn/spqn_similarity_genome.tsv",
      "noquies":"results_chromhmm_elements_counts_spqn_noquies/spqn_similarity_noquies.tsv",
      "tss":"results_chromhmm_elements_counts_spqn_tss/spqn_similarity_tss.tsv",
      "enhancer":"results_chromhmm_elements_counts_spqn_enhancer/spqn_similarity_enhancer.tsv",
      "compartment":"results_compartment_counts_spqn_domain_sm2/spqn_similarity_genome.tsv"}
G=json.loads(sys.stdin.readline())
out={"cells":{},"res":{}}
for tag,f in MATS.items():
    L,R=load(f); n=len(L); out["cells"]={t:cc[t] for t in L}
    off=R[~np.eye(n,dtype=bool)]
    order=np.argsort([-cc[t] for t in L]); rank=np.empty(n,int); rank[order]=np.arange(n)
    rng=np.random.default_rng(1)
    def g(idx):
        s=R[np.ix_(idx,idx)]; return s[~np.eye(len(idx),dtype=bool)].mean()
    res={}
    for key,mem in G.items():
        idx=[L.index(t) for t in mem]; obs=g(idx); nl=np.empty(5000)
        for b in range(5000):
            pick=[]
            for t in idx:
                cand=[order[k] for k in range(max(0,rank[t]-4),min(n,rank[t]+5)) if order[k] not in pick]
                pick.append(rng.choice(cand))
            nl[b]=g(pick)
        res[key]={"r":round(float(obs),4),"pct":round(float((off<obs).mean()),3),"p":round(float((1+(nl>=obs).sum())/5001),4),"null":round(float(nl.mean()),4)}
    out["res"][tag]=res
    print(tag,{k:(v["pct"],v["p"]) for k,v in res.items()},file=sys.stderr)
print(json.dumps(out))
