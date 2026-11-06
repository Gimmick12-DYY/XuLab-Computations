# Per-panel-TF knockdown vs NTC from streamed cells, merged with guide stats
import sys, glob
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

stream_dir, guide_stats, out = sys.argv[1:4]
df = pd.concat([pd.read_csv(f, sep="\t", compression=None) for f in sorted(glob.glob(stream_dir + "/*.tsv.gz"))], ignore_index=True)
df = df[df.pass_guide_filter == 1]
gs = pd.read_csv(guide_stats, sep="\t")
ntc = df[df.gene_target == "Non-Targeting"]
rows = []
for g in gs.gene_symbol:
    pert = df[df.gene_target == g]
    cp_p = pert[g] / pert.total_counts * 1e4
    # batch-matched NTC: NTC cells from the GEM batches holding this perturbation
    nt = ntc[ntc["sample"].isin(pert["sample"].unique())]
    cp_n = nt[g] / nt.total_counts * 1e4
    r = dict(gene_symbol=g, n_cells_stream=len(pert), n_ntc_matched=len(nt),
             ntc_frac_cells_detected=(nt[g] > 0).mean(), ntc_mean_cp10k=cp_n.mean(),
             pert_mean_cp10k=cp_p.mean(), median_n_genes_pert=pert.n_genes_by_counts.median(),
             median_n_genes_ntc=nt.n_genes_by_counts.median())
    r["knockdown_pct"] = 100 * (1 - r["pert_mean_cp10k"] / r["ntc_mean_cp10k"]) if r["ntc_mean_cp10k"] > 0 else np.nan
    r["kd_mwu_p"] = mannwhitneyu(cp_p, cp_n, alternative="less").pvalue if len(pert) and cp_n.sum() > 0 else np.nan
    rows.append(r)
kd = pd.DataFrame(rows)
res = gs.merge(kd, on="gene_symbol", how="left")
ok = res.kd_mwu_p.notna()
p = res.loc[ok, "kd_mwu_p"].values; o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
q = np.minimum.accumulate(q[::-1])[::-1]; qq = np.empty_like(q); qq[o] = np.minimum(q, 1)
res.loc[ok, "kd_q_bh"] = qq
res["paper_classifier_q"] = "not_released"
res.to_csv(out, sep="\t", index=False, float_format="%.4g")
print(res.to_string())
