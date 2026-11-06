# Variant check: obs/exp with ATAC + width only (no leave-one-out panel term); AUROC on the 7 bulk TFs + CTCF motif top-1000
import sys, subprocess, importlib.util
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

elements, out = sys.argv[1:3]
spec = importlib.util.spec_from_file_location("fb", f"{out}/fit_binding.py")
src = open(f"{out}/fit_binding.py").read()
ns = {"np": np}; exec(src[src.index("def nb_glm"):src.index("B_lfc = np.zeros")], ns)
z = np.load(f"{out}/binding.npz", allow_pickle=True)
tfs = list(z["tfs"]); eid = z["element_id"]; pos = {e: i for i, e in enumerate(eid)}; n = len(eid)
E = pd.read_csv(elements, sep="\t", header=None); width = (E[2] - E[1]).values.astype(float)
atac = z["atac_cpm"].astype(float); tot = pd.read_csv(f"{out}/tf_fits.tsv", sep="\t", index_col=0)
D = "/work/users/d/y/dyy12/XuLab/data"
BULK = {"CTCF": "CTCF_majority2of3.bed", "MAZ": "MAZ_HEK293_peaks.bed", "NFYA": "NFYA_K562_peaks.bed",
        "ZBTB7A": "ZBTB7A_HEK293_peaks.bed", "ZIC2": "ZIC2_HEK293_peaks.bed", "ZNF282": "ZNF282_HEK293_peaks.bed", "ZNF777": "ZNF777_HEK293_peaks.bed"}
def hits(bed):
    o = subprocess.run(["bedtools", "intersect", "-u", "-a", elements, "-b", bed], capture_output=True, text=True, check=True).stdout
    y = np.zeros(n, bool)
    for l in o.splitlines(): y[pos[l.split("\t")[3]]] = True
    return y
rows = []
ym = hits(f"{out}/ctcf_motif_p1e-5.bed")
for tf, bed in BULK.items():
    j = tfs.index(tf); y_raw = z["raw"][:, j].astype(float); y = hits(f"{D}/{bed}")
    X = np.column_stack([np.ones(n), np.log1p(atac), np.log(width)])
    b, alpha, mu = ns["nb_glm"](y_raw, X, np.full(n, np.log(tot.loc[tf, "reads"])))
    lfc = np.log2((y_raw + 0.5) / (mu + 0.5)); pr = (y_raw - mu) / np.sqrt(mu + alpha * mu ** 2)
    r = dict(tf=tf, ATAC=roc_auc_score(y, atac), raw_cpm=roc_auc_score(y, z["raw_cpm"][:, j]),
             lfc_noLOO=roc_auc_score(y, lfc), pr_noLOO=roc_auc_score(y, pr), lfc_withLOO=roc_auc_score(y, z["B_lfc"][:, j]),
             B_imp=roc_auc_score(y, z["B_imp"][:, j]), b_atac=b[1])
    if tf == "CTCF":
        r["ctcf_motif_top1000_pr_noLOO"] = ym[np.argsort(-pr)[:1000]].mean()
        r["ctcf_motif_top1000_lfc_noLOO"] = ym[np.argsort(-lfc)[:1000]].mean()
    rows.append(r)
R = pd.DataFrame(rows).set_index("tf"); R.loc["median"] = R.median(numeric_only=True)
R.to_csv(f"{out}/sanity_variant_noLOO.tsv", sep="\t", float_format="%.3f"); print(R.round(3).to_string())
