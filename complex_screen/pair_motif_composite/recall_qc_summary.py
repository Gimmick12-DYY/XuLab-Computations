# Peak re-calling QC summary. Criteria fixed before results (2026-11-11, agreed with Yuyang):
# adopt a variant only if (1) more TFs pass the own-motif gate (rank <= 10 on disc and val) than raw_lab, (2) bulk specificity is not lower
# than raw_lab (CTCF top-1,000 overlap with HEK293T CTCF ChIP-nexus divided by the median overlap of the other 78 TFs; same for ZNF282 with
# HEK293T ChIP-exo), and (3) shared open-chromatin signal goes down vs raw_lab (median frac in HOT bins, median mean ATAC CPM, and mean
# pairwise Jaccard of top-1,000 1 kb bins across TFs all lower). Also reported: n significant peaks, FRiP, Spearman(n significant, cells).
# Output recall/qc_per_tf.tsv, recall/qc_summary.tsv
import json, glob, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/complex_screen/pair_motif_composite/recall"
cells = pd.read_csv(f"{X}/complex_screen/pair_motif_lookup/tf_cells.tsv", sep="\t").set_index("tf").n_cells
J = {f.split("/")[-1][:-5]: json.load(open(f)) for f in glob.glob(f"{R}/qc/*.json")}; rows = []
for t, d in J.items():
    for v, r in d.items(): rows.append(dict(tf=t, variant=v, **{k: x for k, x in r.items() if k != "bins"}))
T = pd.DataFrame(rows); T.to_csv(f"{R}/qc_per_tf.tsv", sep="\t", index=False)
G = pd.read_csv(f"{R}/gate_summary.tsv", sep="\t").set_index("variant"); out = []
for v, g in T.groupby("variant"):
    g = g.set_index("tf"); r = dict(variant=v, n_tf=len(g))
    for ref, tf in [("CTCF_HEK293T", "CTCF"), ("ZNF282_HEK293T", "ZNF282"), ("CTCF_HEK293", "CTCF")]:
        if tf in g.index: r[f"spec_{ref}"] = round(g.loc[tf, f"bulk_{ref}"] / max(g.drop(tf)[f"bulk_{ref}"].median(), 1e-3), 2); r[f"own_{ref}"] = g.loc[tf, f"bulk_{ref}"]; r[f"others_median_{ref}"] = g.drop(tf)[f"bulk_{ref}"].median()
    for k in ["frac_in_atac", "frac_hot", "mean_atac_cpm", "frac_tss1kb", "frip"]:
        if k in g: r[f"median_{k}"] = round(g[k].median(), 3)
    sets = {t: set(J[t][v]["bins"]) for t in g.index if v in J[t]}; jac = [len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in itertools.combinations(sorted(sets), 2)]; r["mean_pairwise_jaccard"] = round(float(np.mean(jac)), 4)
    if "n_sig" in g and g.n_sig.notna().sum() > 5: r["median_n_sig"] = int(g.n_sig.median()); r["rho_nsig_cells"] = round(spearmanr(g.n_sig, cells[g.index])[0], 2)
    if v in G.index: r["gate_strict"] = int(G.loc[v, "strict_le10"]); r["gate_loose"] = int(G.loc[v, "loose_le25"]); r["gate_tfs"] = G.loc[v, "tfs_strict"]
    out.append(r)
S = pd.DataFrame(out).set_index("variant"); b = S.loc["raw_lab"]
S["c1_motif"] = S.gate_strict > b.gate_strict; S["c2_bulk"] = (S.spec_CTCF_HEK293T >= b.spec_CTCF_HEK293T) & (S.spec_ZNF282_HEK293T >= b.spec_ZNF282_HEK293T)
S["c3_open"] = (S.median_frac_hot < b.median_frac_hot) & (S.median_mean_atac_cpm < b.median_mean_atac_cpm) & (S.mean_pairwise_jaccard < b.mean_pairwise_jaccard); S["adopt"] = S.c1_motif & S.c2_bulk & S.c3_open
S.reset_index().to_csv(f"{R}/qc_summary.tsv", sep="\t", index=False); pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50); pd.set_option("display.max_colwidth", 120); print(S.T.to_string())
