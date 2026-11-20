# m52 side check: cause of the up-to-0.079 difference between the m51 Part C KD - NTC log1p difference and m49; is the paper score matrix affected.
import numpy as np, pandas as pd, scipy.sparse as sp
R = "/work/users/d/y/dyy12/XuLab/regulatome"; s = open(f"{R}/m48_stage1_top10k/m48_tables.py").read(); exec(s[:s.index("VERS = {")])
O = f"{R}/m51_edistance/out"  # after the exec
GS = np.load(f"{O}/ntc_gem_stats.npz", allow_pickle=True); sym = GS["genes"].astype(str); F49 = np.load(f"{R}/m49_z_fold_change/out/unit_fold_change.npz", allow_pickle=True); g49 = F49["genes"].astype(str)
det = GS["detected"].sum(0) / GS["n"].sum(); cg = np.flatnonzero((det >= .01) & (sym != ".")); print("m49 gene list reproduced:", bool((sym[cg] == g49).all()), flush=True)
full_last = pd.Series(np.arange(len(sym)), index=sym); full_last = full_last[~full_last.index.duplicated(keep="last")]
cg_last = pd.Series(cg, index=g49); cg_last = cg_last[~cg_last.index.duplicated(keep="last")]
vc = pd.Series(sym[sym != "."]).value_counts(); dup = set(vc[vc > 1].index); dU = sorted(set(UG) & dup)
diff = [g for g in UG if full_last[g] != cg_last[g]]
print("duplicated symbols in the full gene table:", len(dup), "| of them in the 11,228 universe:", len(dU), flush=True)
print("universe genes mapped to a different column (m51: last duplicate in the full table; m49/m23: last duplicate among genes with NTC detection >= 1%):", len(diff), diff, flush=True)
KD = sp.load_npz(f"{O}/kd_cells.npz").tocsr(); KO = pd.read_csv(f"{O}/kd_cells.tsv.gz", sep="\t"); ix = np.flatnonzero(KO.gene_target.values == "ADNP")
ng = np.bincount(KO.gem.values[ix], minlength=len(GS["n"])).astype(float); w = ng / ng.sum(); kdm = np.asarray(KD[ix].mean(0)).ravel(); ref = (w[:, None] * GS["mean"]).sum(0)
a = (kdm - ref)[full_last.reindex(UG).values.astype(int)]; k = list(F49["units"].astype(str)).index("ADNP|all"); b = F49["logdiff"][k][pd.Series(np.arange(len(g49)), index=g49).pipe(lambda x: x[~x.index.duplicated(keep="last")]).reindex(UG).values.astype(int)]
bad = np.flatnonzero(np.abs(a - b) > 1e-4); print("ADNP: genes with |m51 - m49| > 1e-4:", len(bad), [UG[i] for i in bad][:20], "| all in the remapped list:", set(UG[i] for i in bad) <= set(diff), flush=True)
a2 = (kdm - ref)[cg_last.reindex(UG).values.astype(int)]; print("ADNP with the m49 column mapping, max |m51 - m49|:", float(np.abs(a2 - b).max()), flush=True)
