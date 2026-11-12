# m22 post-build: summaries, background-column check (SPEC 5), agreement check (SPEC 8, diagnostic), Jaccard description (SPEC 9).
import json
import numpy as np, pandas as pd
from scipy.stats import rankdata
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m22_edge_table"; OUT = f"{O}/out"; rng = np.random.default_rng(20261112)
E = pd.read_csv(f"{OUT}/regulome_edges.tsv.gz", sep="\t", low_memory=False); W = np.load(f"{O}/work_matrices.npz", allow_pickle=True); tfs = list(W["tfs"]); genes = W["genes"]; gi = {g: k for k, g in enumerate(genes)}; NG = len(genes)
ZC, H1, H2, RP = "observed_kd_z [external: X-Atlas/Orion HEK293T]", "observed_kd_z_half1 [external]", "observed_kd_z_half2 [external]", "observed_kd_reproducible [external]"
# summaries
pt = E.groupby("tf").agg(n_target_genes=("gene", "size"), n_promoter_bound=("promoter_peak", "sum"), median_shared_by=("shared_by", "median"), n_with_module=("coordinated_binding_module", lambda v: int((v.fillna("") != "").sum())), n_reproducible_kd_external=(RP, "sum")).reset_index()
pt["tier"] = pt.tf.map(E.groupby("tf").tier.first()); pt.to_csv(f"{OUT}/per_tf_summary.tsv", sep="\t", index=False)
pg = E.groupby("gene").agg(n_tfs_bound=("tf", "size"), n_tier1=("tier", lambda v: int((v == "Tier 1").sum())), n_tier2=("tier", lambda v: int((v == "Tier 2").sum())), top_tf_by_share=("tf", "first")).reset_index()
top = E.sort_values("predicted_impact_share_at_gene", ascending=False).drop_duplicates("gene").set_index("gene").tf; pg["top_tf_by_share"] = pg.gene.map(top); pg.to_csv(f"{OUT}/per_gene_summary.tsv", sep="\t", index=False)
# SPEC 5: does enrichment_over_background improve own-motif and ChIP agreement over raw signalValue x ABC?
def boot_med(d): d = np.asarray(d, float); d = d[np.isfinite(d)]; b = np.median(d[rng.integers(0, len(d), (2000, len(d)))], 1); return float(np.median(d)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(d)
CHK = []
for t, g in E.groupby("tf"):
    if g.own_motif_promoter.isna().all() or len(g) < 200: continue
    n = min(1000, len(g) // 2); a = g.nlargest(n, "enrichment_over_background").own_motif_promoter.astype(bool).mean(); b = g.nlargest(n, "score_signal_x_abc").own_motif_promoter.astype(bool).mean()
    CHK.append(dict(tf=t, check="own motif at promoter", n_top=n, frac_by_enrichment=a, frac_by_signal=b, diff=a - b))
def chip_genes(f):
    C = pd.read_csv(f"{X}/data/{f}", sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"])
    EL = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene"])
    hit = np.zeros(len(EL), bool)
    for ch, G in EL.groupby("chr"):
        H = C[C.c == ch]
        if not len(H): continue
        o = np.argsort(H.s.values); hs, he = H.s.values[o], np.maximum.accumulate(H.e.values[o]); k = np.searchsorted(hs, G.end.values, side="left") - 1; ok = k >= 0; ok[ok] = he[k[ok]] > G.start.values[ok]; hit[G.index.values] = ok
    return set(EL.TargetGene[hit])
CHIP = {"CTCF": "CTCF_HEK293T_GSE103651_peaks.bed", "NFYA": "NFYA_K562_peaks.bed", "MAZ": "MAZ_HEK293_peaks.bed", "ZBTB7A": "ZBTB7A_HEK293_peaks.bed", "ZIC2": "ZIC2_HEK293_peaks.bed", "ZNF282": "ZNF282_HEK293_peaks.bed", "ZNF777": "ZNF777_HEK293_peaks.bed"}
for t, f in CHIP.items():
    g = E[E.tf == t]
    if len(g) < 200: continue
    cg = chip_genes(f); n = min(1000, len(g) // 2); a = g.nlargest(n, "enrichment_over_background").gene.isin(cg).mean(); b = g.nlargest(n, "score_signal_x_abc").gene.isin(cg).mean()
    CHK.append(dict(tf=t, check=f"bulk ChIP genes ({f})", n_top=n, frac_by_enrichment=a, frac_by_signal=b, diff=a - b))
CK = pd.DataFrame(CHK); CK.to_csv(f"{OUT}/check_background_column.tsv", sep="\t", index=False, float_format="%.4g")
VER = []
for nm, d in (("own motif", CK[CK.check == "own motif at promoter"]["diff"]), ("bulk ChIP", CK[CK.check.str.startswith("bulk")]["diff"])):
    m, lo, hi, n = boot_med(d) if len(d) else (np.nan, np.nan, np.nan, 0); VER.append(dict(check=nm, n_tfs=n, median_diff_enrichment_minus_signal=m, lo=lo, hi=hi, passes=bool(lo > 0)))
VER = pd.DataFrame(VER); VER.to_csv(f"{OUT}/check_background_verdict.tsv", sep="\t", index=False, float_format="%.4g"); validated = bool(VER.passes.all())
json.dump(dict(enrichment_over_background_validated=validated, predicted_impact_uses="score_signal_x_abc" if not validated else "score_signal_x_abc (background-corrected version also valid)"), open(f"{OUT}/background_column_status.json", "w"), indent=1)
print(VER.to_string(index=False), flush=True)
# SPEC 8: agreement of predicted impact with external knockdown (diagnostic), with swapped-TF control
MIS = dict(zip([l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()], json.load(open(f"{RG}/m7_step2/features/raw_hicfixed/feature_qc.json"))["mis_paired_map"]["mis0"]))
S = W["share"]; SC = W["score"]; ti = {t: k for k, t in enumerate(tfs)}
def sp(a, b): m = np.isfinite(a) & np.isfinite(b); return np.nan if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0 else float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])
Eu = E[E[ZC].notna()]; within_gene, within_gene_sw = [], []
for g, d in Eu.groupby("gene"):
    if len(d) < 3: continue
    k = gi[g]; within_gene.append(sp(d.predicted_impact_share_at_gene.values, d[ZC].abs().values))
    sw = np.array([S[ti[MIS[t]], k] if MIS.get(t) in ti else np.nan for t in d.tf]); within_gene_sw.append(sp(sw, d[ZC].abs().values))
within_tf, within_tf_sw = [], []
for t, d in Eu.groupby("tf"):
    if len(d) < 20: continue
    within_tf.append(sp(d.predicted_impact.values, d[ZC].abs().values)); sw = np.array([SC[ti[MIS[t]], gi[g]] for g in d.gene]) if MIS.get(t) in ti else np.full(len(d), np.nan); within_tf_sw.append(sp(sw, d[ZC].abs().values))
def bmean(v): v = np.asarray(v, float); v = v[np.isfinite(v)]; b = v[rng.integers(0, len(v), (2000, len(v)))].mean(1); return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(v)
AG = []
for nm, a, b in (("within genes (>= 3 bound TFs with knockdown): Spearman(share, |z|)", within_gene, within_gene_sw), ("within TFs (bound genes): Spearman(score, |z|)", within_tf, within_tf_sw)):
    m, lo, hi, n = bmean(a); m2, lo2, hi2, _ = bmean(b); d_ = np.array(a) - np.array(b); m3, lo3, hi3, _ = bmean(d_)
    AG.append(dict(label="diagnostic (external knockdown used as a check, not a definition)", level=nm, n=n, mean_spearman=m, lo=lo, hi=hi, swapped_tf_mean=m2, sw_lo=lo2, sw_hi=hi2, real_minus_swapped=m3, d_lo=lo3, d_hi=hi3))
AG = pd.DataFrame(AG); AG.to_csv(f"{OUT}/agreement_check.tsv", sep="\t", index=False, float_format="%.4g"); print(AG.drop(columns="label").round(4).to_string(index=False), flush=True)
# SPEC 9: Jaccard of target lists (description)
B = {t: set(d.gene) for t, d in E.groupby("tf")}; J = pd.DataFrame(index=tfs, columns=tfs, dtype=float)
for a in tfs:
    for b in tfs: A_, B_ = B.get(a, set()), B.get(b, set()); J.loc[a, b] = len(A_ & B_) / len(A_ | B_) if (A_ | B_) else np.nan
J.to_csv(f"{OUT}/tf_x_tf_jaccard.tsv", sep="\t", float_format="%.4f"); print("edges", len(E), "TFs", E.tf.nunique(), "genes", E.gene.nunique(), flush=True)
