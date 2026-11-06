# M1 sanity checks: bulk-ChIP AUROC (7 TFs), CTCF motif in top-1000 elements, per-element TF counts (HOT correction)
import sys, subprocess, io
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

elements, out, motif = sys.argv[1:4]
D = "/work/users/d/y/dyy12/XuLab/data"
BULK = {"CTCF": "CTCF_majority2of3.bed", "MAZ": "MAZ_HEK293_peaks.bed", "NFYA": "NFYA_K562_peaks.bed",
        "ZBTB7A": "ZBTB7A_HEK293_peaks.bed", "ZIC2": "ZIC2_HEK293_peaks.bed", "ZNF282": "ZNF282_HEK293_peaks.bed",
        "ZNF777": "ZNF777_HEK293_peaks.bed"}
z = np.load(f"{out}/binding.npz", allow_pickle=True)
tfs = list(z["tfs"]); eid = z["element_id"]; n = len(eid)
pos = {e: i for i, e in enumerate(eid)}

def hits(bed, extra=()):
    o = subprocess.run(["bedtools", "intersect", "-u", "-a", elements, "-b", bed, *extra], capture_output=True, text=True, check=True).stdout
    y = np.zeros(n, bool)
    for line in o.splitlines(): y[pos[line.split("\t")[3]]] = True
    return y

scores = {"B_lfc": z["B_lfc"], "B_pr": z["B_pr"], "B_imp": z["B_imp"], "B_blend_k10000": z["B_blend_k10000"],
          "B_blend_k3000": z["B_blend_k3000"], "B_blend_k30000": z["B_blend_k30000"], "raw_cpm": z["raw_cpm"]}
atac = z["atac_cpm"]
rows = []
for tf, bed in BULK.items():
    j = tfs.index(tf); y = hits(f"{D}/{bed}")
    r = dict(tf=tf, n_pos=int(y.sum()), frac_pos=round(y.mean(), 4), ATAC=roc_auc_score(y, atac))
    for k, m in scores.items(): r[k] = roc_auc_score(y, m[:, j])
    r["AUPRC_ATAC"] = average_precision_score(y, atac); r["AUPRC_B_lfc"] = average_precision_score(y, z["B_lfc"][:, j])
    r["AUPRC_B_imp"] = average_precision_score(y, z["B_imp"][:, j]); r["AUPRC_B_blend_k10000"] = average_precision_score(y, z["B_blend_k10000"][:, j])
    r["AUPRC_raw_cpm"] = average_precision_score(y, z["raw_cpm"][:, j])
    rows.append(r)
A = pd.DataFrame(rows).set_index("tf")
A.loc["median"] = A.median(numeric_only=True)
A.to_csv(f"{out}/sanity_bulk_auroc.tsv", sep="\t", float_format="%.3f")
print(A.round(3).to_string())

# CTCF motif: FIMO MA0139.2 hits at p < 1e-5 (column 6 = p)
m = pd.read_csv(motif, sep="\t", header=None, usecols=[0, 1, 2, 5])
m = m[m[5] < 1e-5]
mb = f"{out}/ctcf_motif_p1e-5.bed"; m[[0, 1, 2]].to_csv(mb, sep="\t", header=False, index=False)
ym = hits(mb); j = tfs.index("CTCF")
mot = {"all elements": ym.mean()}
for k, s in [("B_lfc", z["B_lfc"][:, j]), ("B_pr", z["B_pr"][:, j]), ("B_imp", z["B_imp"][:, j]),
             ("B_blend_k10000", z["B_blend_k10000"][:, j]), ("raw_cpm", z["raw_cpm"][:, j]), ("ATAC", atac)]:
    mot[f"top1000 {k}"] = ym[np.argsort(-s, kind="stable")[:1000]].mean()
M = pd.Series(mot, name="frac_with_CTCF_motif"); M.to_csv(f"{out}/sanity_ctcf_motif.tsv", sep="\t", float_format="%.3f")
print(M.round(3).to_string())

# HOT: number of TFs calling each element, corrected vs uncorrected
raw = z["raw"]; cpm = z["raw_cpm"]
unc = np.log2((cpm + 1e-3) / (cpm.mean(0, keepdims=True) + 1e-3))
calls = {"raw CPM log2(element/TF mean) > 2 & reads >= 3": (unc > 2) & (raw >= 3),
         "B_lfc > 2 & reads >= 3": (z["B_lfc"] > 2) & (raw >= 3),
         "B_blend_k10000 z > 2": z["B_blend_k10000"] > 2,
         "B_imp z > 2": z["B_imp"] > 2}
H = []
for k, c in calls.items():
    ntf = c.sum(1)
    H.append(dict(call=k, elements_with_any=int((ntf > 0).sum()), median_tfs_if_any=float(np.median(ntf[ntf > 0])) if (ntf > 0).any() else 0,
                  p99=float(np.percentile(ntf, 99)), max=int(ntf.max()), ge10=int((ntf >= 10).sum()), ge20=int((ntf >= 20).sum()),
                  ge40=int((ntf >= 40).sum()), spearman_with_ATAC=float(pd.Series(ntf).corr(pd.Series(atac), method="spearman"))))
H = pd.DataFrame(H); H.to_csv(f"{out}/sanity_hot.tsv", sep="\t", index=False, float_format="%.3f")
print(H.to_string())
