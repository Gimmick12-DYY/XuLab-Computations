# Worked example of the ABC -> TF x gene conversion for NRF1 and CTCF (Arm 2), recomputing the reported odds ratios from the exported rows.
# Step 1 = m2c_abc_canonical/tf_targets_overlap.py (binary targets, responders on the double-centred z, scipy Fisher OR);
# m10 = m10_promoter_regulome/m10_eval.py definition "abc_hicfixed_sum" (summed ABC score of links from bound elements, top 1,000, responders on raw per-TF z, OR with +0.5).
# usage: worked_example.py -> worked_example/<TF>_genes.tsv, <TF>_links.tsv, join_summary.tsv, check.tsv
import os
import numpy as np, pandas as pd
from scipy import stats
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m10_promoter_regulome/worked_example"; os.makedirs(O, exist_ok=True)
R = f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed"; PK = "/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2"; QMIN = -np.log10(0.05)
ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; panel = lambda t: ALIAS.get(t, t)
E = pd.read_csv(f"{R}/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end"]); E["key"] = E.chr + ":" + E.start.astype(str) + "-" + E.end.astype(str); NE = len(E); ek = dict(zip(E.key, range(NE)))
L = pd.read_csv(sorted(f"{R}/Predictions/{f}" for f in os.listdir(f"{R}/Predictions") if f.startswith("EnhancerPredictionsFull_threshold0.02") and f.endswith(".tsv"))[0], sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score", "isSelfPromoter"])
L["e"] = (L.chr + ":" + L.start.astype(str) + "-" + L.end.astype(str)).map(ek).astype(int); L["self"] = L.isSelfPromoter.astype(str).str.lower().eq("true")
GL = pd.read_csv(f"{R}/Neighborhoods/GeneList.txt", sep="\t", usecols=["name"]).name.astype(str).unique(); genes = np.array(sorted(set(GL)))
# knockdown matrix, Arm 2, as tf_targets_overlap.py load(): genes with NTC detection >= 0.10, kept primary units averaged per TF, double-centred, panel-collapsed
d = np.load(f"{RG}/m4_validation/kd_response/out_v2/arm2_all_cell/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{RG}/m4_validation/kd_response/out_v2/arm2_all_cell/guide_qc.tsv", sep="\t")
u = d["units"].tolist(); ui = {x: k for k, x in enumerate(u)}; keep = d["ntc_detect"] >= .1; g = d["genes"][keep].astype(str); Z = d["z"][:, keep].astype(float)
prim = q[(q.kept == 1) & q.kind.isin(["all", "construct"])]; tf_ = sorted(prim.tf.unique()); P_ = np.nan_to_num(np.vstack([np.nanmean(Z[[ui[x] for x in prim.unit[prim.tf == t]]], axis=0) for t in tf_]))
Rr = P_ - P_.mean(0); Rr = Rr - Rr.mean(1, keepdims=True); pts = sorted({panel(t) for t in tf_})
col = lambda M: np.vstack([M[[i for i, t in enumerate(tf_) if panel(t) == p]].mean(0) for p in pts]); Rp, Pp = col(Rr), col(P_)
kg = {x: k for k, x in enumerate(g)}  # duplicate symbols: last occurrence (as tf_targets_overlap.py line 57)
uni = np.array([x in kg for x in genes]); ug = genes[uni]; ukix = np.array([kg[x] for x in ug]); nu = len(ug)
# join summary
gs = pd.Series(g); ens = gs.str.match(r"^ENSG\d+")
J = [("ABC GeneList genes", len(genes)), ("KD genes >= 10% NTC detection (rows, incl. duplicates)", len(g)), ("KD unique symbols", gs.nunique()), ("KD duplicate symbols", int(gs.duplicated().sum())),
     ("KD genes named by Ensembl ID (no symbol)", int(ens.sum())), ("universe = GeneList & KD (symbol match)", nu), ("KD genes lost (not in GeneList)", int((~gs.drop_duplicates(keep="last").isin(set(genes))).sum())),
     ("  of which Ensembl-ID-named", int((~gs[~gs.duplicated(keep="last")].isin(set(genes)) & ens[~gs.duplicated(keep="last")]).sum())), ("GeneList genes not in KD >= 10%", int((~uni).sum())),
     ("HP1 = mean of KD rows", ",".join(t for t in tf_ if panel(t) == "HP1")), ("ZUFSP = mean of KD rows", ",".join(t for t in tf_ if panel(t) == "ZUFSP"))]
pd.DataFrame(J, columns=["item", "value"]).to_csv(f"{O}/join_summary.tsv", sep="\t", index=False)
CC = np.load(f"{RG}/m7_step2/features/canon_counts/raw_counts.npy"); tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]
CH = {c: (G.start.values, G.end.values, G.index.values) for c, G in E.sort_values(["chr", "start"]).groupby("chr")}; checks = []
m2c = pd.read_csv(f"{RG}/m2c_abc_canonical/overlap/HEK293T_hicfixed/per_tf.tsv", sep="\t"); gap = pd.read_csv(f"{RG}/m10_promoter_regulome/gap_per_tf.tsv", sep="\t")
for t in ("NRF1", "CTCF"):
    P = pd.read_csv(f"{PK}/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 2, 8, 9]); P.columns = ["chr", "start", "end", "q", "off"]; P = P[P.q >= QMIN]; summit = {}
    for c, s in zip(P.chr.values, (P.start + P.off).values):
        if c not in CH: continue
        st, en, ix = CH[c]; k = np.searchsorted(st, s, side="right") - 1
        if k >= 0 and s < en[k]: summit[ix[k]] = summit.get(ix[k], 0) + 1
    Lu = L[L.TargetGene.isin(set(ug))].copy(); Lu["n_summits_q05_in_element"] = Lu.e.map(summit).fillna(0).astype(int); Lu["bound"] = Lu.n_summits_q05_in_element > 0; Lu["tf_reads_at_element"] = CC[Lu.e.values, tfs79.index(t)].astype(int)
    Lu[["chr", "start", "end", "TargetGene", "ABC.Score", "self", "n_summits_q05_in_element", "bound", "tf_reads_at_element"]].rename(columns={"self": "is_self_promoter"}).to_csv(f"{O}/{t}_links.tsv", sep="\t", index=False)
    G = pd.DataFrame(index=ug); grp = Lu.groupby("TargetGene"); G["n_linked_elements"] = grp.size(); G["n_bound_elements"] = grp.bound.sum()
    bp = Lu[Lu.bound & Lu.self].TargetGene.unique(); bd = Lu[Lu.bound & ~Lu.self].TargetGene.unique()
    G["bound_via"] = np.where(G.index.isin(bp) & G.index.isin(bd), "both", np.where(G.index.isin(bp), "promoter", np.where(G.index.isin(bd), "distal", "none")))
    G["sum_ABC_of_bound_links"] = Lu[Lu.bound].groupby("TargetGene")["ABC.Score"].sum(); G = G.fillna({"n_linked_elements": 0, "n_bound_elements": 0, "sum_ABC_of_bound_links": 0})
    G["target_step1"] = G.n_bound_elements > 0
    sc = G.sum_ABC_of_bound_links.values; top = np.zeros(nu, bool); top[np.argsort(-sc, kind="stable")[:1000]] = True; G["in_top1000_m10"] = top
    G["n_genes_tied_at_top1000_cutoff"] = int((sc == np.sort(sc)[::-1][999]).sum())
    pi = pts.index(t); zd = Rp[pi][ukix]; zr = Pp[pi][ukix]; k5 = int(round(nu * .05))
    G["z_doublecentred_step1"] = zd; G["absz_rank_doublecentred"] = pd.Series(-np.abs(zd)).rank(method="first").astype(int).values; G["responder_step1"] = False; G.iloc[np.argsort(-np.abs(zd), kind="stable")[:k5], G.columns.get_loc("responder_step1")] = True
    G["z_raw_m10"] = zr; G["absz_rank_raw"] = pd.Series(-np.abs(zr)).rank(method="first").astype(int).values; G["responder_m10"] = False; G.iloc[np.argsort(-np.abs(zr), kind="stable")[:k5], G.columns.get_loc("responder_m10")] = True
    G["is_knocked_down_gene_itself"] = G.index == t
    G.index.name = "gene"; G.to_csv(f"{O}/{t}_genes.tsv", sep="\t", float_format="%.5g")
    a = int((G.target_step1 & G.responder_step1).sum()); b = int((G.target_step1 & ~G.responder_step1).sum()); c = int((~G.target_step1 & G.responder_step1).sum()); dd = int((~G.target_step1 & ~G.responder_step1).sum())
    o1 = stats.fisher_exact([[a, b], [c, dd]])[0]; rep1 = m2c[(m2c.arm == "arm2") & (m2c.tf == t)].OR.iloc[0]
    a2 = int((G.in_top1000_m10 & G.responder_m10).sum()); b2 = int((G.in_top1000_m10 & ~G.responder_m10).sum()); c2 = int((~G.in_top1000_m10 & G.responder_m10).sum()); d2 = int((~G.in_top1000_m10 & ~G.responder_m10).sum())
    o2 = (a2 + .5) * (d2 + .5) / ((b2 + .5) * (c2 + .5)); rep2 = gap[(gap.arm == "arm2") & (gap.reading == "r1") & (gap.N == 1000) & (gap.definition == "abc_hicfixed_sum") & (gap.tf == t)].OR.iloc[0]
    checks += [dict(tf=t, analysis="Step 1 (m2c): target = any bound linked element; responder = top 5% |double-centred z|; scipy Fisher OR", universe=nu, target_resp=a, target_nonresp=b, nontarget_resp=c, nontarget_nonresp=dd, OR_recomputed=o1, OR_reported=rep1, knocked_down_gene_in_universe=t in set(ug), kd_gene_is_target=bool(G.target_step1.get(t, False)), kd_gene_is_responder=bool(G.responder_step1.get(t, False))),
               dict(tf=t, analysis="m10 abc_hicfixed_sum: top 1,000 by summed ABC of bound links; responder = top 5% |raw per-TF z|; OR with +0.5", universe=nu, target_resp=a2, target_nonresp=b2, nontarget_resp=c2, nontarget_nonresp=d2, OR_recomputed=o2, OR_reported=rep2, knocked_down_gene_in_universe=t in set(ug), kd_gene_is_target=bool(G.in_top1000_m10.get(t, False)), kd_gene_is_responder=bool(G.responder_m10.get(t, False)))]
C = pd.DataFrame(checks); C.to_csv(f"{O}/check.tsv", sep="\t", index=False, float_format="%.4f"); print(C.to_string(index=False)); print(pd.DataFrame(J).to_string(index=False))
