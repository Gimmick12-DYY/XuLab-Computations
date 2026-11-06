# Sharing rate after dropping motifs enriched in more than half of the TFs, known-motif route, main foreground; shown as a background rate only.
import itertools, json, pandas as pd, numpy as np
O = "/work/users/d/y/dyy12/XuLab/complex_screen/motif_shared_20261103"; tfs = [t.strip() for t in open(f"{O}/tfs.txt")]; H = {}
for t in tfs:
    d = pd.read_csv(f"{O}/sea/{t}/sea.tsv", sep="\t", comment="#"); d = d[d.ID.notna()]; H[t] = set(d.ID[(d.QVALUE < .05) & (d.ENR_RATIO > 1)])
ids = sorted(set().union(*H.values())); ub = pd.Series({m: sum(m in H[t] for t in tfs) / 79 for m in ids}); drop = set(ub.index[ub > .5]); TT = pd.read_csv(f"{O}/tomtom_db/tomtom.tsv", sep="\t", comment="#"); TT = TT[TT["q-value"] < .05]; sim = {}
for r in TT.itertuples(): sim.setdefault(r.Query_ID, set()).add(r.Target_ID); sim.setdefault(r.Target_ID, set()).add(r.Query_ID)
def shared(a, b, D, S):
    A, B = H[a] - D, H[b] - D
    return bool(A & B) or (S and any((sim.get(m, set()) & B) for m in A))
n = len(tfs) * 78 / 2
print("motifs >50%:", len(drop), "of", len(ids), "| same or similar:", round(sum(shared(a, b, drop, True) for a, b in itertools.combinations(tfs, 2)) / n, 3), "| same motif only:", round(sum(shared(a, b, drop, False) for a, b in itertools.combinations(tfs, 2)) / n, 3), "| median motifs left per TF:", np.median([len(H[t] - drop) for t in tfs]))
for p in (["mlx", "srebf2"], ["rbbp7", "srcap"], ["ezh2", "mtf2"], ["maz", "znf143"], ["mbd3", "rbbp4"]): print(p, sorted(H[p[0]] & H[p[1]] - drop)[:6], shared(p[0], p[1], drop, True))
print(open(f"{O}/background_rates.json").read())
