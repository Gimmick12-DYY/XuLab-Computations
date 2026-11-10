# m2b (ATAC x H3K27ac activity) vs m2_abc (ATAC only), both ABC v1.1.2 power-law, ABC >= 0.02, self-promoter in.
# Element activity agreement; links kept / lost / gained overall and by element class (promoter element vs distal_active / bivalent, self-promoter
# separately); Spearman / Pearson of ABC.Score on shared links; per-gene element rank shifts (shared links) and top-element changes.
# Writes results/compare/{summary.tsv, by_class.tsv, per_gene.tsv}. usage: compare_abc.py
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
X = "/work/users/d/y/dyy12/XuLab/regulatome"; OLD = f"{X}/m2_abc/results"; NEW = f"{X}/m2b_abc_k27ac/results"; O = f"{NEW}/compare"; os.makedirs(O, exist_ok=True)
cls = pd.read_csv(f"{X}/m2_abc/inputs/candidate_class.tsv", sep="\t").set_index("name")["class"]
cand = pd.read_csv(f"{X}/m2_abc/inputs/candidates_active.bed", sep="\t", header=None, names=["chr", "start", "end", "eid"])
k2c = dict(zip(cand.chr + ":" + cand.start.astype(str) + "-" + cand.end.astype(str), cand.eid.map(cls)))
def links(d):
    a = pd.read_csv(f"{d}/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score", "isSelfPromoter", "activity_base", "distance"])
    a["el"] = a.chr + ":" + a.start.astype(str) + "-" + a.end.astype(str); a["key"] = a.el + ">" + a.TargetGene
    a["self"] = a.isSelfPromoter.astype(str).str.lower().eq("true"); a["ecls"] = np.where(a.self, "self_promoter", a.el.map(k2c).fillna("unmapped")); return a.set_index("key")
a, b = links(OLD), links(NEW)
out = []
eo = pd.read_csv(f"{OLD}/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["name", "activity_base", "isPromoterElement"]).set_index("name")
en = pd.read_csv(f"{NEW}/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["name", "activity_base"]).set_index("name")
j = eo.join(en, rsuffix="_new").dropna()
out += [("elements", len(j)), ("activity Spearman all elements", spearmanr(j.activity_base, j.activity_base_new)[0]),
        ("activity Spearman promoter elements", spearmanr(j.activity_base[j.isPromoterElement], j.activity_base_new[j.isPromoterElement])[0]),
        ("activity Spearman non-promoter elements", spearmanr(j.activity_base[~j.isPromoterElement], j.activity_base_new[~j.isPromoterElement])[0])]
sh = a.index.intersection(b.index); lost = a.index.difference(b.index); gain = b.index.difference(a.index)
out += [("links old", len(a)), ("links new", len(b)), ("links shared", len(sh)), ("links lost (old only)", len(lost)), ("links gained (new only)", len(gain)),
        ("Jaccard", len(sh) / len(a.index.union(b.index))),
        ("ABC Spearman shared", spearmanr(a.loc[sh, "ABC.Score"], b.loc[sh, "ABC.Score"])[0]), ("ABC Pearson shared", pearsonr(a.loc[sh, "ABC.Score"], b.loc[sh, "ABC.Score"])[0]),
        ("ABC Spearman shared non-self", spearmanr(a.loc[sh][~a.loc[sh].self]["ABC.Score"], b.loc[sh][~a.loc[sh].self]["ABC.Score"])[0]),
        ("ABC Pearson shared non-self", pearsonr(a.loc[sh][~a.loc[sh].self]["ABC.Score"], b.loc[sh][~a.loc[sh].self]["ABC.Score"])[0]),
        ("genes with >=1 link old", a.TargetGene.nunique()), ("genes with >=1 link new", b.TargetGene.nunique()),
        ("median distance non-self old (bp)", a[~a.self].distance.abs().median()), ("median distance non-self new (bp)", b[~b.self].distance.abs().median())]
rows = []
for c in ["self_promoter", "promoter", "distal_active", "bivalent", "unmapped"]:
    ia, ib = set(a.index[a.ecls == c]), set(b.index[b.ecls == c])
    if not ia and not ib: continue
    s = sorted(ia & ib); rows.append(dict(element_class=c, old=len(ia), new=len(ib), shared=len(s), lost=len(ia - ib), gained=len(ib - ia),
        spearman_shared=spearmanr(a.loc[s, "ABC.Score"], b.loc[s, "ABC.Score"])[0] if len(s) > 2 else np.nan, pearson_shared=pearsonr(a.loc[s, "ABC.Score"], b.loc[s, "ABC.Score"])[0] if len(s) > 2 else np.nan))
BC = pd.DataFrame(rows); BC.to_csv(f"{O}/by_class.tsv", sep="\t", index=False, float_format="%.4f")
# per gene: rank shifts of shared non-self links; top non-self element (over each run's own thresholded links) changed?
pg = []
an, bn = a[~a.self], b[~b.self]
for g, x in an.groupby("TargetGene"):
    if g not in set(bn.TargetGene): pg.append(dict(gene=g, n_old=len(x), n_new=0, n_shared=0, med_abs_drank=np.nan, top_changed=np.nan)); continue
    y = bn[bn.TargetGene == g]; s = x.index.intersection(y.index)
    ra = x.loc[s, "ABC.Score"].rank(ascending=False); rb = y.loc[s, "ABC.Score"].rank(ascending=False)
    pg.append(dict(gene=g, n_old=len(x), n_new=len(y), n_shared=len(s), med_abs_drank=float((ra - rb).abs().median()) if len(s) else np.nan,
                   top_changed=float(x["ABC.Score"].idxmax() != y["ABC.Score"].idxmax())))
PG = pd.DataFrame(pg)
for g in set(bn.TargetGene) - set(an.TargetGene): PG.loc[len(PG)] = dict(gene=g, n_old=0, n_new=int((bn.TargetGene == g).sum()), n_shared=0, med_abs_drank=np.nan, top_changed=np.nan)
PG.to_csv(f"{O}/per_gene.tsv", sep="\t", index=False, float_format="%.3f")
both = PG[(PG.n_old > 0) & (PG.n_new > 0)]
out += [("genes with non-self links in both", len(both)), ("median over genes of median |delta rank| (shared links, genes with >=2 shared)", both[both.n_shared >= 2].med_abs_drank.median()),
        ("mean over genes of median |delta rank| (genes with >=2 shared)", both[both.n_shared >= 2].med_abs_drank.mean()),
        ("fraction of genes whose top non-self element changes", both.top_changed.mean()),
        ("fraction of genes whose top element changes, genes with >=2 links in both", both[(both.n_old >= 2) & (both.n_new >= 2)].top_changed.mean()),
        ("genes losing all non-self links", int(((PG.n_old > 0) & (PG.n_new == 0)).sum())), ("genes gaining non-self links from none", int(((PG.n_old == 0) & (PG.n_new > 0)).sum()))]
S = pd.DataFrame(out, columns=["metric", "value"]); S.to_csv(f"{O}/summary.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250); print(S.to_string(index=False)); print(BC.to_string(index=False))
