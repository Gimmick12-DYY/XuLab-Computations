# M1 v2 per-class summary: fraction of each TF's reads (and imputed calls) in each element class, enrichment vs the
# panel average (other 78 TFs), group tests (PRC2 in polycomb, HP1/SETDB1/ADNP in znf_rpts/het), bulk-ChIP AUROC per class
import json, subprocess, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

elements, cdir, out = sys.argv[1:4]
D = "/work/users/d/y/dyy12/XuLab/data"
BULK = {"CTCF": "CTCF_majority2of3.bed", "MAZ": "MAZ_HEK293_peaks.bed", "NFYA": "NFYA_K562_peaks.bed",
        "ZBTB7A": "ZBTB7A_HEK293_peaks.bed", "ZIC2": "ZIC2_HEK293_peaks.bed", "ZNF282": "ZNF282_HEK293_peaks.bed",
        "ZNF777": "ZNF777_HEK293_peaks.bed"}
GROUPS = {"PRC2 (EZH2, MTF2)": (["EZH2", "MTF2"], ["polycomb"]),
          "HP1/SETDB1/ADNP": (["HP1", "SETDB1", "ADNP"], ["znf_rpts", "het"])}
CLASSES = ["promoter", "distal_active", "bivalent", "polycomb", "znf_rpts", "het"]

E = pd.read_csv(elements, sep="\t", header=None, names=["chrom", "start", "end", "element_id", "type", "parent_id", "parent_type", "class"])
tfs = [l.strip() for l in open(f"{cdir}/../tfs.txt") if l.strip()]
cls = E["class"].values
raw, imp, meta = {}, {}, {}
for tf in tfs:
    z = np.load(f"{cdir}/{tf}.npz"); raw[tf] = z["raw"].astype(float); imp[tf] = z["imp"].astype(float)
    meta[tf] = json.load(open(f"{cdir}/{tf}.json"))

def frac_table(mat, denom):
    rows = {}
    for tf in tfs:
        r = {c: mat[tf][cls == c].sum() / denom[tf] for c in CLASSES}
        r["all_elements"] = mat[tf].sum() / denom[tf]
        rows[tf] = r
    return pd.DataFrame(rows).T[CLASSES + ["all_elements"]]
FR = frac_table(raw, {t: meta[t]["reads"] for t in tfs})
FI = frac_table(imp, {t: meta[t]["imputed_nnz"] for t in tfs})
v1 = pd.Series({t: json.load(open(f"{cdir}/../../counts/{t}.json"))["reads_in_elements"] / meta[t]["reads"] for t in tfs})
FR.insert(0, "v1_all", v1)
FR.to_csv(f"{out}/frac_reads_by_class.tsv", sep="\t", float_format="%.4f")
FI.to_csv(f"{out}/frac_imputed_by_class.tsv", sep="\t", float_format="%.4f")
bp = E.assign(w=E.end - E.start).groupby("class").w.sum() / 3.1e9
print("genome fraction by class:", (bp[CLASSES] * 100).round(2).to_dict())

def enrich(F):
    # each TF's class fraction / mean of the other 78 TFs (leave-one-out panel average)
    S = F[CLASSES].sum(0)
    return F[CLASSES].apply(lambda r: r / ((S - r) / (len(F) - 1)), axis=1)
ER, EI = enrich(FR), enrich(FI)
ER.to_csv(f"{out}/enrich_reads_by_class.tsv", sep="\t", float_format="%.3f")
EI.to_csv(f"{out}/enrich_imputed_by_class.tsv", sep="\t", float_format="%.3f")
pd.set_option("display.width", 250)
print("\nfraction of raw reads in elements by class (v1_all = old universe):")
print(FR.round(3).to_string())
print("\nraw-read enrichment vs panel average:")
print(ER.round(2).to_string())
for g, (members, cl) in GROUPS.items():
    for c in cl:
        print(f"{g} in {c}: raw " + ", ".join(f"{m} {ER.loc[m, c]:.2f}x (rank {int(ER[c].rank(ascending=False)[m])}/79)" for m in members)
              + " | imputed " + ", ".join(f"{m} {EI.loc[m, c]:.2f}x" for m in members))
for c in ["polycomb", "znf_rpts", "het"]:
    print(f"top 8 raw enrichment in {c}:", ER[c].sort_values(ascending=False).head(8).round(2).to_dict())

# ---- bulk-ChIP AUROC on the v2 universe, all elements and per class ------------------------------------------------
a = np.load(f"{cdir}/atac_counts.npz")["atac"].astype(float)
pos = {e: i for i, e in enumerate(E.element_id.values)}
rows = []
for tf, bed in BULK.items():
    o = subprocess.run(["bedtools", "intersect", "-u", "-a", elements, "-b", f"{D}/{bed}"], capture_output=True, text=True, check=True).stdout
    y = np.zeros(len(E), bool)
    for line in o.splitlines(): y[pos[line.split("\t")[3]]] = True
    for c in ["all"] + CLASSES:
        m = np.ones(len(E), bool) if c == "all" else cls == c
        if y[m].sum() < 20 or (~y[m]).sum() < 20: continue
        rows.append(dict(tf=tf, cls=c, n_pos=int(y[m].sum()), ATAC=roc_auc_score(y[m], a[m]), raw=roc_auc_score(y[m], raw[tf][m]),
                         imputed=roc_auc_score(y[m], imp[tf][m])))
A = pd.DataFrame(rows)
A.to_csv(f"{out}/bulk_auroc_by_class.tsv", sep="\t", index=False, float_format="%.3f")
print("\nbulk-ChIP AUROC, median over 7 TFs:")
print(A.groupby("cls")[["n_pos", "ATAC", "raw", "imputed"]].median().round(3).reindex(["all"] + CLASSES).to_string())
print(A.pivot(index="tf", columns="cls", values="raw").round(3).to_string())
