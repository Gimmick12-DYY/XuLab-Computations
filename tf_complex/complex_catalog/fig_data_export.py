import numpy as np, sys, csv, collections, json
sys.path.insert(0,"scripts")
from plot_correlation_matrix import load_similarity_tsv
r=list(csv.reader(open("../data/TF1000cells.meta.csv")));i=r[0].index("TF")
cc=collections.Counter(x[i].lower() for x in r[1:])
out={}
for tag,f in {"noq":"results_chromhmm_elements_counts_spqn_noquies/tf_similarity_noquies.tsv","noqS":"results_chromhmm_elements_counts_spqn_noquies/spqn_similarity_noquies.tsv"}.items():
    L,R=load_similarity_tsv(f); n=len(L)
    m=(R.sum(1)-1)/(n-1)
    out[tag]={"tf":L,"meanr":m.round(4).tolist(),"cells":[cc[t] for t in L]}
    rng=np.random.default_rng(1)
    order=np.argsort([-cc[t] for t in L]); rank=np.empty(n,int); rank[order]=np.arange(n)
    def g(idx):
        s=R[np.ix_(idx,idx)]; return s[~np.eye(len(idx),dtype=bool)].mean()
    nulls={}
    for name,grp in {"WIZ/ZNF644":["wiz","znf644"],"NFYA/MLX/SREBF2":["nfya","mlx","srebf2"],"NuRD core":["rbbp4","rbbp7","mbd3"],"CTCF/STAG2":["ctcf","stag2"]}.items():
        idx=[L.index(t) for t in grp]; nl=[]
        for _ in range(5000):
            pick=[]
            for t in idx:
                cand=[order[k] for k in range(max(0,rank[t]-4),min(n,rank[t]+5)) if order[k] not in pick]
                pick.append(rng.choice(cand))
            nl.append(g(pick))
        nulls[name]={"obs":float(g(idx)),"null":np.round(nl,4).tolist()}
    out[tag]["nulls"]=nulls
    print("rho", tag, np.corrcoef(np.log10([cc[t] for t in L]), m)[0,1], file=sys.stderr)
print(json.dumps(out))
