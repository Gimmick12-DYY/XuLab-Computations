# m10_promoter_regulome Amendment 1, follow-ups F1-F3 (diagnostic, benchmark already read; no model fitting).
# usage: m10_followup.py -> F1_testable.tsv, F1_counts.tsv, F2_thinning.tsv, F3_all_tfs.tsv, F3_spearman.tsv
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m10_promoter_regulome"; M9 = f"{RG}/m9_abc_diagnosis"; LAB = "diagnostic, benchmark already read"; SEED = 20261111
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]; FL = pd.read_csv(f"{RG}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf")
CT = {t: np.load(f"{O}/work/counts/{t}.npz", allow_pickle=True) for t in tfs79}; pg = pd.Index(CT[tfs79[0]]["genes"].astype(str))
G = pd.read_csv(f"{O}/gap_per_tf.tsv", sep="\t")
# ---- F1
REF = ["motif", "bulk_chip_HEK293orK562", "bulk_chip_HEK293T", "chromnitron"]; rows = []; cnt = []
for arm in ("arm2", "arm1"):
    A = G[(G.arm == "arm2") & (G.reading == "r1") & (G.N == 1000)]; testable = sorted(set(A[A.definition.isin(REF) & (A.OR_lo > A.OR_matched)].tf))
    B = G[(G.arm == arm) & (G.reading == "r1") & (G.N == 1000)]
    for t in testable:
        refs = A[(A.tf == t) & A.definition.isin(REF) & (A.OR_lo > A.OR_matched)]; z = CT[t]
        for d in ("raw_tss500", "imp_tss500"):
            q = B[(B.tf == t) & (B.definition == d)]
            if not len(q): continue
            q = q.iloc[0]; ok = bool(q.OR_lo > q.OR_matched and q.OR_lo > q.OR_pseudoTF and q.OR_lo > q.get("OR_swapped", np.nan))
            rows.append(dict(label=LAB, arm=arm, tf=t, testable_by=";".join(f"{r.definition} {r.OR:.2f} [{r.OR_lo:.2f}-{r.OR_hi:.2f}]" for r in refs.itertuples()), ours=d, OR=q.OR, OR_lo=q.OR_lo, OR_hi=q.OR_hi, OR_matched=q.OR_matched,
                             OR_pseudoTF=q.OR_pseudoTF, OR_swapped=q.get("OR_swapped", np.nan), passes=ok, cells=FL.cells.get(t, np.nan), total_reads=float(z["total"]), reads_within_1kb_tss=float(z["in_tss1k_any"])))
    R = pd.DataFrame([r for r in rows if r["arm"] == arm])
    for d in ("raw_tss500", "imp_tss500"):
        Q = R[R.ours == d]; cnt.append(dict(label=LAB, arm=arm, ours=d, n_testable=len(Q), n_pass=int(Q.passes.sum()), passing=",".join(Q.tf[Q.passes])))
pd.DataFrame(rows).to_csv(f"{O}/F1_testable.tsv", sep="\t", index=False, float_format="%.4f"); pd.DataFrame(cnt).to_csv(f"{O}/F1_counts.tsv", sep="\t", index=False)
# ---- F2 thinning
Rz = np.load(f"{M9}/work/responses_arm2.npz", allow_pickle=True); tt = list(Rz["tfs"]); ug = pd.Index(Rz["genes"].astype(str)); n = len(ug); pidx = pg.get_indexer(ug); assert (pidx >= 0).all()
def or_(tg, resp): a = (tg & resp).sum(-1); b = (tg & ~resp).sum(-1); c = resp.sum() - a; d = n - a - b - c; return (a + .5) * (d + .5) / ((b + .5) * (c + .5))
def resp_of(t): y = Rz["r1"][tt.index(t)]; m = np.zeros(n, bool); m[np.argsort(-np.abs(y), kind="stable")[:int(round(n * .05))]] = True; return m
def top(v, rng, N=1000): o = np.lexsort((rng.random(len(v)), -v)); m = np.zeros(len(v), bool); m[o[:N]] = True; return m
BK = dict(np.load(f"{M9}/work/scores_bulk.npz", allow_pickle=True)); bg = pd.Index(BK.pop("genes").astype(str)); CN = dict(np.load(f"{M9}/work/scores_chromnitron.npz", allow_pickle=True)); cg = pd.Index(CN.pop("genes").astype(str))
FM = pd.read_csv(f"{M9}/work/fimo/NRF1.tsv", sep="\t", header=None, names=["motif", "gene", "score", "p"]); FM = FM[~FM.motif.str.contains("_shuf")].groupby("gene").score.max().reindex(ug)
F2 = []; rng = np.random.default_rng(SEED)
def thin(t, prob, name):
    p = np.nan_to_num(prob, nan=0.0); p = np.clip(p, 0, None); p = p / p.sum(); k = int(CT[t]["w500"][pidx[pidx >= 0]].sum()); resp = resp_of(t)
    full = float(or_(top(np.nan_to_num(prob, nan=-1e9), rng), resp)); ours = float(or_(top(CT[t]["w500"][pidx].astype(float), rng), resp))
    ors = np.array([or_(top(rng.multinomial(k, p).astype(float), rng), resp) for _ in range(200)])
    nz = np.array([(rng.multinomial(k, p) > 0).sum() for _ in range(20)]).mean()
    F2.append(dict(label=LAB, tf=t, probability_vector=name, promoter_fragments_ours=k, full_depth_OR=full, our_raw_OR=ours, thinned_OR_median=float(np.median(ors)), thinned_OR_p2_5=float(np.percentile(ors, 2.5)), thinned_OR_p97_5=float(np.percentile(ors, 97.5)), thinned_frac_ge_full_minus_0_1=float((ors >= full - .1).mean()), mean_genes_with_ge1_read_thinned=float(nz), genes_with_ge1_read_ours=int((CT[t]["w500"][pidx] > 0).sum())))
thin("CTCF", pd.Series(BK["CTCF_293T"], index=bg).reindex(ug).values, "bulk ChIP HEK293T p-value track")
thin("CTCF", pd.Series(BK["CTCF"], index=bg).reindex(ug).values, "bulk ChIP HEK293 CPM (ENCODE)")
thin("NRF1", pd.Series(CN["NRF1"], index=cg).reindex(ug).values, "Chromnitron")
thin("NRF1", (FM - FM.min()).fillna(0).values, "motif score shifted to >= 0")
pd.DataFrame(F2).to_csv(f"{O}/F2_thinning.tsv", sep="\t", index=False, float_format="%.4f")
# ---- F3
A = G[(G.arm == "arm2") & (G.reading == "r1") & (G.N == 1000) & (G.definition == "raw_tss500")].set_index("tf"); testable = set(pd.DataFrame(rows).query("arm == 'arm2'").tf) if rows else set(); T = []
for t in tfs79:
    z = CT[t]; T.append(dict(label=LAB, tf=t, cells=FL.cells.get(t, np.nan), tertile=FL.tertile.get(t, ""), total_fragments=float(z["total"]), fragments_within_1kb_tss=float(z["in_tss1k_any"]), frac_within_1kb_tss=float(z["in_tss1k_any"]) / float(z["total"]),
                         promoter_fragments_tss500_universe=float(z["w500"][pidx[pidx >= 0]].sum()), raw_promoter_OR=A.OR.get(t, np.nan), raw_promoter_OR_minus_matched=(A.OR - A.OR_matched).get(t, np.nan), testable=t in testable))
T = pd.DataFrame(T); T.to_csv(f"{O}/F3_all_tfs.tsv", sep="\t", index=False, float_format="%.4f"); S = []; rng = np.random.default_rng(SEED + 3)
for sn, Q in (("all_tested", T.dropna(subset=["raw_promoter_OR_minus_matched"])), ("testable", T[T.testable].dropna(subset=["raw_promoter_OR_minus_matched"]))):
    x, y = Q.promoter_fragments_tss500_universe.values, Q.raw_promoter_OR_minus_matched.values; r = spearmanr(x, y)[0]; b = []
    for _ in range(2000): i = rng.integers(0, len(x), len(x)); b.append(spearmanr(x[i], y[i])[0])
    S.append(dict(label=LAB, set=sn, n_tfs=len(Q), spearman=r, lo=np.nanpercentile(b, 2.5), hi=np.nanpercentile(b, 97.5)))
pd.DataFrame(S).to_csv(f"{O}/F3_spearman.tsv", sep="\t", index=False, float_format="%.4f"); print(pd.DataFrame(cnt).to_string(index=False)); print(pd.DataFrame(F2).to_string(index=False)); print(pd.DataFrame(S).to_string(index=False))
