# m17_fig5_panels P4 (SPEC.md Amendment 2): many-TF gene example candidates with element-level binding. Illustration / selection on an already-read benchmark.
# usage: m17_p4.py -> out/gene_example/p4_candidates.tsv, p4_gene_tf.tsv, p4_elements.tsv, p4_links.tsv
import os, re, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
X_ = "/work/users/d/y/dyy12/XuLab"; RG_ = f"{X_}/regulatome"; OUT4 = f"{RG_}/m17_fig5_panels/out/gene_example"; os.makedirs(OUT4, exist_ok=True)
LAB4 = "illustration / selection on an already-read benchmark (diagnostic)"; V = "/vast/som/xujie_lab/TAGATG-293T"; QMIN = -np.log10(0.05)
sys.argv = ["m15.py", "dc"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); exec(src[:src.index("# ---- matched-gene strata")])  # TF, UG, Z1, Z2, ZA (dc)
tfs79 = [l.strip() for l in open(f"{RG_}/m1_binding/v2/tfs.txt") if l.strip()]
STRONG = {"CTCF", "DNMT1", "E4F1", "MEF2A", "NRF1", "PA2G4", "RBBP7", "RBPJ", "SAFB", "SETDB1"}
Wsc = pd.read_csv(f"{RG_}/m11_peak_regulome/out/wide_score_q0.05.tsv.gz", sep="\t", index_col=0).reindex(index=tfs79, columns=UG).fillna(0)
PCTB = pd.DataFrame(np.vstack([rankdata(r) / len(UG) for r in Wsc.values]), index=tfs79, columns=UG)
M15o = f"{RG_}/m15_gene_to_tf/out"; CONF = {K: pd.read_csv(f"{M15o}/dc_pairs_K{K}.tsv", sep="\t")[["gene", "tf"]].drop_duplicates() for K in (2000, 10000)}
CS = {K: set(map(tuple, C.values)) for K, C in CONF.items()}
cand = set()
for K, C in CONF.items(): n = C.groupby("gene").tf.nunique(); cand |= set(n.index[n >= 3])
DECL = ["PGAM1", "HSPB1", "TXNIP"]; genes = sorted(cand | set(DECL)); print("candidates", len(cand), flush=True)
# canonical ABC links for these genes
A = pd.read_csv(f"{RG_}/m2c_abc_canonical/results/HEK293T_hicfixed/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["chr", "start", "end", "class", "TargetGene", "TargetGeneTSS", "ABC.Score", "isSelfPromoter"])
A = A[A.TargetGene.isin(genes)].copy(); A["self"] = A.isSelfPromoter.astype(str).str.lower().eq("true"); A.insert(0, "label", LAB4); A.to_csv(f"{OUT4}/p4_links.tsv", sep="\t", index=False)
EL = A[["chr", "start", "end"]].drop_duplicates().sort_values(["chr", "start"]).reset_index(drop=True); CH = {c: (g.start.values, g.end.values, g.index.values) for c, g in EL.groupby("chr")}
def elem_of(c, p):
    r = np.full(len(p), -1)
    for ch in np.unique(c):
        if ch not in CH: continue
        st, en, ix = CH[ch]; m = c == ch; k = np.searchsorted(st, p[m], side="right") - 1; ok = (k >= 0) & (p[m] < en[np.maximum(k, 0)]); rr = np.full(m.sum(), -1); rr[ok] = ix[k[ok]]; r[m] = rr
    return r
PK = []
for t in tfs79:
    f = f"{V}/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
    if not os.path.exists(f) or os.path.getsize(f) == 0: continue
    P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 6, 8, 9]); P.columns = ["chr", "pstart", "pend", "signal", "q", "off"]; P = P[P.q >= QMIN]; P["summit"] = P.pstart + P.off
    P["e"] = elem_of(P.chr.values.astype(str), P.summit.values); P = P[P.e >= 0].sort_values("signal", ascending=False).drop_duplicates("e"); P["tf"] = t; PK.append(P)
PK = pd.concat(PK).drop(columns="chr"); PK = PK.merge(EL.reset_index().rename(columns={"index": "e", "chr": "echr", "start": "estart", "end": "eend"}), on="e")
E = A.merge(PK, left_on=["chr", "start", "end"], right_on=["echr", "estart", "eend"]); E["link_score"] = E.signal * E["ABC.Score"]
E = E[["TargetGene", "tf", "chr", "start", "end", "class", "self", "ABC.Score", "signal", "q", "summit", "link_score"]].rename(columns={"TargetGene": "gene", "signal": "peak_signalValue", "q": "peak_minus_log10_q", "summit": "peak_summit", "start": "element_start", "end": "element_end"})
E.insert(0, "label", LAB4); E.sort_values(["gene", "tf", "link_score"], ascending=[True, True, False]).to_csv(f"{OUT4}/p4_elements.tsv", sep="\t", index=False, float_format="%.5g")
ug = {g: k for k, g in enumerate(UG)}; ti = {t: k for k, t in enumerate(TF)}; ROWS = []
for g in genes:
    Eg = E[E.gene == g]
    for t in tfs79:
        e = Eg[Eg.tf == t]; kd = t in ti and g in ug
        r = dict(label=LAB4, gene=g, tf=t, n_linked_elements_with_peak=len(e), promoter_peak=bool(e["self"].any()) if len(e) else False, best_element=f"{e.iloc[0].chr}:{e.iloc[0].element_start}-{e.iloc[0].element_end}" if len(e) else "",
                 best_abc=e.iloc[0]["ABC.Score"] if len(e) else np.nan, best_signal=e.iloc[0].peak_signalValue if len(e) else np.nan, binding_pct=float(PCTB.loc[t, g]) if g in PCTB.columns else np.nan, has_knockdown=kd,
                 z_half1=Z1[ti[t], ug[g]] if kd else np.nan, z_half2=Z2[ti[t], ug[g]] if kd else np.nan, z_all=ZA[ti[t], ug[g]] if kd else np.nan, confirmed_K2000=(g, t) in CS[2000], confirmed_K10000=(g, t) in CS[10000], strong_phenotype=t in STRONG)
        r["direction_on_kd"] = "" if not kd or not np.isfinite(r["z_all"]) else ("down" if r["z_all"] < 0 else "up"); r["bound"] = r["n_linked_elements_with_peak"] > 0 and r["binding_pct"] >= .75; ROWS.append(r)
GT = pd.DataFrame(ROWS); GT.to_csv(f"{OUT4}/p4_gene_tf.tsv", sep="\t", index=False, float_format="%.4g")
HK = re.compile(r"^(RPL|RPS|MRPL|MRPS|EIF|EEF|ATP5|NDUF|COX|UQCR|SNRP|PSM|HSP|TUB|ACT)"); HKX = {"GAPDH", "PGK1", "ENO1", "LDHA", "PPIA", "PGAM1"}
C = []
for g, H in GT.groupby("gene"):
    d = dict(label=LAB4, gene=g, declared=g in DECL and g not in cand, housekeeping_flag=bool(HK.match(g)) or g in HKX, n_tfs_peak_linked=int((H.n_linked_elements_with_peak > 0).sum()))
    for K in (10000, 2000):
        cf = H[f"confirmed_K{K}"]; d[f"n_confirmed_K{K}"] = int(cf.sum()); d[f"n_both_K{K}"] = int((cf & H.bound).sum()); d[f"n_bound_only_K{K}"] = int((~cf & H.bound).sum()); d[f"n_confirmed_only_K{K}"] = int((cf & ~H.bound).sum())
        d[f"both_tfs_K{K}"] = ",".join(H.tf[cf & H.bound]); d[f"confirmed_tfs_K{K}"] = ",".join(H.tf[cf])
    d["sum_absz_both_K10000"] = float(H.z_all[H.confirmed_K10000 & H.bound].abs().sum()); d["confirmed_only_EN_D"] = set(H.tf[H.confirmed_K10000 | H.confirmed_K2000]) <= {"E4F1", "NRF1", "DNMT1"}; C.append(d)
C = pd.DataFrame(C).sort_values(["n_both_K10000", "n_both_K2000", "sum_absz_both_K10000"], ascending=False).reset_index(drop=True); C.insert(1, "rank", np.arange(1, len(C) + 1))
pick = C[(~C.housekeeping_flag) & (~C.confirmed_only_EN_D) & (~C.declared)]; C["figure_top_candidate"] = False; C.loc[(pick.index[0] if len(pick) else 0), "figure_top_candidate"] = True
C.to_csv(f"{OUT4}/p4_candidates.tsv", sep="\t", index=False, float_format="%.4g")
print(C.drop(columns=["label", "confirmed_tfs_K2000", "both_tfs_K2000"]).head(15).to_string(index=False)); print("top candidate:", C.gene[C.figure_top_candidate].tolist())
