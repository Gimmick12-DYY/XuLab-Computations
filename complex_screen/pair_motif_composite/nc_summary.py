# Negative-control summary: per HOMER motif, its log10 p in 10 pooled-Tn5 pseudo-libraries and 10 random open-chromatin sets; how many motifs
# reach p <= 1e-5 / 1e-3 in the controls (false-positive load); and for each TF's own motif, an empirical p = fraction of the 20 control runs (per
# chromosome half) with log p as strong as the TF's own library. Adds columns to motif_tables/own_motif_all_tfs.tsv -> own_motif_all_tfs_nc.tsv
import glob, numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; T = f"{O}/motif_tables"
NC = {}
for f in glob.glob(f"{T}/nc/*.tsv"):
    mode, seed, pt = f.split("/")[-1][:-4].split("_"); k = pd.read_csv(f, sep="\t"); NC[(mode, int(seed), pt)] = dict(zip(k.iloc[:, 0], k.iloc[:, 3] / np.log(10)))
rows = []
for (mode, seed, pt), d in NC.items():
    v = np.array(list(d.values())); top = min(d, key=d.get); rows.append(dict(mode=mode, seed=seed, part=pt, n_p1e5=int((v <= -5).sum()), n_p1e3=int((v <= -3).sum()), n_p1e2=int((v <= -2).sum()), top_motif=top.split("/")[0], top_log10p=round(d[top], 1)))
R = pd.DataFrame(rows).sort_values(["mode", "part", "seed"]); R.to_csv(f"{T}/negative_control_runs.tsv", sep="\t", index=False)
print(R.groupby(["mode", "part"])[["n_p1e5", "n_p1e3", "n_p1e2"]].describe().round(1).to_string())
print("top motifs in pooled controls:", R[R["mode"] == "pooled"].top_motif.value_counts().head(8).to_dict())
D = pd.read_csv(f"{T}/own_motif_all_tfs.tsv", sep="\t"); LP = {}
for f in glob.glob(f"{O}/recall/homer/raw_lab/*/*/knownResults.txt"):
    t, pt = f.split("/")[-3:-1]; k = pd.read_csv(f, sep="\t"); LP[(t, pt)] = dict(zip(k.iloc[:, 0], k.iloc[:, 3] / np.log(10)))
for i, r in D.iterrows():
    if pd.isna(r.homer_motif): continue
    keys = [k for k in LP.get((r.tf, "disc"), {}) if k.split("/")[0] == r.homer_motif]
    if not keys: continue
    key = keys[0]
    for pt in ["disc", "val"]:
        own = LP.get((r.tf, pt), {}).get(key)
        for mode in ["pooled", "random"]:
            nul = [d.get(key) for (m, s, p), d in NC.items() if m == mode and p == pt and key in d]
            if own is not None and nul: D.at[i, f"NC_{mode}_{pt}_emp_p"] = round((1 + sum(x <= own for x in nul)) / (1 + len(nul)), 3); D.at[i, f"NC_{mode}_{pt}_median_log10p"] = round(float(np.median(nul)), 2)
D.to_csv(f"{T}/own_motif_all_tfs_nc.tsv", sep="\t", index=False)
c = [c for c in D.columns if c.startswith("NC_")]; pd.set_option("display.width", 250)
print(D.dropna(subset=["homer_motif"])[["tf", "cells", "homer_motif", "A_disc_rank", "A_disc_log10p", "A_val_rank", "A_val_log10p"] + c].to_string(index=False))
