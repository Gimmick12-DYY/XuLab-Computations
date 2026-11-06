import numpy as np, sys, csv, collections
sys.path.insert(0,"scripts")
from plot_correlation_matrix import load_similarity_tsv
exec(open("/dev/stdin").read()) if False else None
r=list(csv.reader(open("../data/TF1000cells.meta.csv")));i=r[0].index("TF")
cc=collections.Counter(x[i].lower() for x in r[1:])
rng=np.random.default_rng(1)
mats={"noq":"results_chromhmm_elements_counts_spqn_noquies/tf_similarity_noquies.tsv",
      "noqS":"results_chromhmm_elements_counts_spqn_noquies/spqn_similarity_noquies.tsv",
      "allS":"results_chromhmm_elements_counts_spqn/spqn_similarity_genome.tsv",
      "enhS":"results_chromhmm_elements_counts_spqn_enhancer/spqn_similarity_enhancer.tsv"}
known={"NuRD core":["rbbp4","rbbp7","mbd3"],"NuRD+ZNF512B":["rbbp4","rbbp7","mbd3","znf512b"],
 "PRC2":["ezh2","mtf2","rbbp4","rbbp7"],"EZH2/MTF2":["ezh2","mtf2"],"RBBP4/RBBP7":["rbbp4","rbbp7"],
 "WIZ/ZNF644":["wiz","znf644"],"CTCF/STAG2":["ctcf","stag2"],"MEF2A/MEF2D":["mef2a","mef2d"],
 "ZNF512/ZNF512B":["znf512","znf512b"],"NFYA/MLX/SREBF2":["nfya","mlx","srebf2"],"posterior HOX":["hoxa10","hoxb9","hoxc9","hoxd10"],
 "ADNP/HP1":["adnp","hp1"],"HP1/SETDB1":["hp1","setdb1"]}
def g(R,idx):
    s=R[np.ix_(idx,idx)]; return s[~np.eye(len(idx),dtype=bool)].mean()
for tag,f in mats.items():
    L,R=load_similarity_tsv(f); n=len(L)
    order=np.argsort([-cc[t] for t in L]); rank=np.empty(n,int); rank[order]=np.arange(n)
    print("==",tag)
    for name,grp in known.items():
        idx=[L.index(t) for t in grp]; obs=g(R,idx); null=[]
        for _ in range(5000):
            pick=[]
            for t in idx:  # depth-matched: random TF within +-4 ranks, no repeats
                cand=[order[k] for k in range(max(0,rank[t]-4),min(n,rank[t]+5)) if order[k] not in pick]
                pick.append(rng.choice(cand))
            null.append(g(R,pick))
        null=np.array(null); print(f"  {name}: r={obs:.3f} depth-matched p={(1+(null>=obs).sum())/5001:.4f} null_mean={null.mean():.3f}")
    iu=np.triu_indices(n,1); v=R[iu]; o=np.argsort(-v)[:30]
    print("  TOP30: "+"; ".join(f"{L[iu[0][k]]}-{L[iu[1][k]]} {v[k]:.3f}" for k in o))
