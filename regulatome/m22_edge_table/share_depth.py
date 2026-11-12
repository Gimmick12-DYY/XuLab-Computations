# m22 Amendment 3: predicted share vs depth; alternative impact versions; within-gene check against external knockdown. Tables only. Diagnostic.
import json
import numpy as np, pandas as pd
from scipy.stats import rankdata
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m22_edge_table"; OUT = f"{O}/out"; rng = np.random.default_rng(20261112)
ZC, H1, H2, RP = "observed_kd_z [external: X-Atlas/Orion HEK293T]", "observed_kd_z_half1 [external]", "observed_kd_z_half2 [external]", "observed_kd_reproducible [external]"
E = pd.read_csv(f"{OUT}/regulome_edges.tsv.gz", sep="\t", low_memory=False, usecols=["tf", "gene", "score_signal_x_abc", "predicted_impact_share_at_gene", ZC, H1, H2, RP])
FL = pd.read_csv(f"{RG}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf"); RT = pd.read_csv(f"{RG}/m7_step2/features/canon_counts/read_totals.tsv", sep="\t").set_index("tf").reads
# 2. alternative versions
E["within_tf_percentile"] = E.groupby("tf").score_signal_x_abc.rank(pct=True); med = E.groupby("tf").score_signal_x_abc.median(); E["scaled"] = E.score_signal_x_abc / E.tf.map(med)
E["depth_scaled_share"] = E.scaled / E.groupby("gene").scaled.transform("sum")
# 1. per-TF depth relations
P = E.groupby("tf").agg(n_bound=("gene", "size"), mean_share=("predicted_impact_share_at_gene", "mean"), median_share=("predicted_impact_share_at_gene", "median"), mean_score=("score_signal_x_abc", "mean"), mean_depth_scaled_share=("depth_scaled_share", "mean"))
for col, nm in (("predicted_impact_share_at_gene", "n_top_share"), ("within_tf_percentile", "n_top_within_tf_pct"), ("depth_scaled_share", "n_top_depth_scaled")):
    top = E.sort_values(col, ascending=False).drop_duplicates("gene"); P[nm] = top.tf.value_counts().reindex(P.index).fillna(0).astype(int)
P["log10_cells"] = np.log10(FL.cells.reindex(P.index).astype(float)); P["log10_fragments"] = np.log10(RT.reindex(P.index).astype(float)); P.reset_index().to_csv(f"{OUT}/share_vs_depth_per_tf.tsv", sep="\t", index=False, float_format="%.4g")
def spb(a, b):
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]; f = lambda i: np.corrcoef(rankdata(a[i]), rankdata(b[i]))[0, 1]; bt = [f(rng.integers(0, len(a), len(a))) for _ in range(2000)]; return f(np.arange(len(a))), np.nanpercentile(bt, 2.5), np.nanpercentile(bt, 97.5), int(m.sum())
D = []
for col in ("mean_share", "median_share", "mean_score", "n_top_share", "mean_depth_scaled_share", "n_top_depth_scaled", "n_top_within_tf_pct", "n_bound"):
    for dep in ("log10_cells", "log10_fragments"):
        r, lo, hi, n = spb(P[col].values, P[dep].values); D.append(dict(label="diagnostic", quantity=col, depth=dep, n_tfs=n, spearman=r, lo=lo, hi=hi))
D = pd.DataFrame(D); D.to_csv(f"{OUT}/share_vs_depth_summary.tsv", sep="\t", index=False, float_format="%.4g"); print(D.drop(columns="label").round(3).to_string(index=False))
print("top-share TFs:", P.n_top_share.sort_values(ascending=False).head(8).to_dict()); print("top within-TF percentile TFs:", P.n_top_within_tf_pct.sort_values(ascending=False).head(8).to_dict()); print("top depth-scaled TFs:", P.n_top_depth_scaled.sort_values(ascending=False).head(8).to_dict())
# 3. check vs external knockdown
MIS = dict(zip([l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()], json.load(open(f"{RG}/m7_step2/features/raw_hicfixed/feature_qc.json"))["mis_paired_map"]["mis0"]))
Ek = E[E[ZC].notna()].copy(); Ek["absz"] = Ek[ZC].abs()
key = {(t, g): k for k, (t, g) in enumerate(zip(E.tf, E.gene))}
def sp(a, b): m = np.isfinite(a) & np.isfinite(b); return np.nan if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0 else float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])
R = []
for col in ("predicted_impact_share_at_gene", "within_tf_percentile", "depth_scaled_share"):
    vals = E[col].values; per, persw, hit, chance = [], [], [], []
    for g, d in Ek.groupby("gene"):
        if len(d) < 3: continue
        per.append(sp(d[col].values, d.absz.values)); sw = np.array([vals[key[(MIS.get(t), g)]] if (MIS.get(t), g) in key else 0.0 for t in d.tf]); persw.append(sp(sw, d.absz.values))
        rp = d[RP].fillna(False).astype(bool).values; hit.append(bool(rp[np.argmax(d[col].values)])); chance.append(rp.mean())
    def bm(v): v = np.asarray(v, float); v = v[np.isfinite(v)]; b = v[rng.integers(0, len(v), (2000, len(v)))].mean(1); return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(v)
    m, lo, hi, n = bm(per); m2, lo2, hi2, _ = bm(persw); dd = np.array(per) - np.array(persw); m3, lo3, hi3, _ = bm(dd)
    R.append(dict(label="diagnostic (external knockdown as a check)", version=col, n_genes=n, within_gene_spearman=m, lo=lo, hi=hi, swapped_tf=m2, sw_lo=lo2, sw_hi=hi2, real_minus_swapped=m3, d_lo=lo3, d_hi=hi3,
                  frac_top_tf_reproducible=float(np.mean(hit)), chance_rate=float(np.mean(chance)), n_genes_with_any_reproducible=int(np.sum(np.array(chance) > 0))))
R = pd.DataFrame(R); R.to_csv(f"{OUT}/share_versions_check.tsv", sep="\t", index=False, float_format="%.4g"); print(R.drop(columns="label").round(4).to_string(index=False))
E[["tf", "gene", "within_tf_percentile", "depth_scaled_share"]].to_csv(f"{OUT}/edges_alternative_impact.tsv.gz", sep="\t", index=False, float_format="%.5g")
