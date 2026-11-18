# m40 specificity matrix: rows = knockdown TF (Perturb-seq genes, |z| >= c), columns = Regulome list TF, cell = share of the row TF's Perturb-seq genes found in the column TF's list.
# Diagonal = own list. Three views: raw share (%), column z (each list vs the other knockdowns' genes, off-diagonal reference), double-centred z (log2 share with row and column effects removed, off-diagonal reference).
import sys, os
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"
src = open(f"{R}/m38_link_rules/m38.py").read(); exec(src[:src.index("T25 = pd.read_csv")])  # UG, gi, rule_abc, window, cicero, V
O = f"{R}/m40_overlap_reference/out/specificity"; os.makedirs(O, exist_ok=True); sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tfs79 = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]; LISTS = {"A": {}, "G": {}}
for t in tfs79:
    f = f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
    if not os.path.exists(f) or os.path.getsize(f) == 0: continue
    P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= 5000]; c5, p5 = P.c.values.astype(str), (P.s + P.off).values
    A, _ = rule_abc(c5, p5); LISTS["A"][t] = A; LISTS["G"][t] = A | window(c5, p5, 10000) | cicero(t)[0]
T25 = pd.read_csv(f"{R}/m23_sgrna_top25/out/observed_kd_top25.tsv.gz", sep="\t", usecols=["tf", "gene", "z_top25"]); TFS = sorted(set(T25.tf) & set(LISTS["A"])); NT = len(TFS)
Z = T25.pivot(index="tf", columns="gene", values="z_top25").reindex(index=TFS, columns=UG).values.astype(float); OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
for i, t in enumerate(TFS):
    for o in OWN.get(t, [t]):
        if o in gi: Z[i, gi[o]] = np.nan
OK = np.isfinite(Z); AZ = np.abs(np.nan_to_num(Z)); EX3 = {"NRF1", "E4F1", "DNMT1"}; off = ~np.eye(NT, dtype=bool); SUM = []
def offz(M, axis):  # z of every cell vs the off-diagonal cells of its column (axis=0) or row (axis=1)
    Mo = np.where(off, M, np.nan); mu = np.nanmean(Mo, axis=axis, keepdims=True); sd = np.nanstd(Mo, axis=axis, ddof=1, keepdims=True); return (M - mu) / sd
with pd.ExcelWriter(f"{O}/specificity_matrices.xlsx") as W:
    for rule in ("A", "G"):
        B = np.vstack([LISTS[rule][t] for t in TFS]).astype(int)
        for c in (2, 3, 4, 5):
            ch = (OK & (AZ >= c)).astype(int); den = ch.sum(1).astype(float); F = ch @ B.T  # F[i, j] = row TF i Perturb-seq genes in list j
            keep = den >= 10; tf = [t for t, k in zip(TFS, keep) if k]; F_, d_ = F[np.ix_(keep, keep)], den[keep]; n = len(tf); of = ~np.eye(n, dtype=bool)
            S = 100 * F_ / d_[:, None]
            Mo = np.where(of, S, np.nan); CZ = (S - np.nanmean(Mo, 0)) / np.nanstd(Mo, 0, ddof=1)
            L = np.log2((F_ + .5) / (d_[:, None] + 1)); Lo = np.where(of, L, np.nan); rm = np.nanmean(Lo, 1, keepdims=True); cm = np.nanmean(Lo - rm, 0, keepdims=True); Rz = L - rm - cm
            Rz = (Rz - np.nanmean(np.where(of, Rz, np.nan))) / np.nanstd(np.where(of, Rz, np.nan), ddof=1)
            tag = f"{rule}_z{c}"
            for nm, M in (("share", S), ("colz", CZ), ("dcz", Rz)):
                pd.DataFrame(M, index=pd.Index(tf, name="knockdown TF (rows) / Regulome list (cols)"), columns=tf).round(3).to_csv(f"{O}/{tag}_{nm}.tsv", sep="\t"); pd.DataFrame(M, index=tf, columns=tf).round(3).to_excel(W, sheet_name=f"{tag}_{nm}")
            dg = np.diag(Rz); dc = np.diag(CZ); rk_col = np.array([1 + (S[of[:, j], j] > S[j, j]).sum() for j in range(n)]); rk_dcz_row = np.array([1 + (Rz[i, of[i]] > Rz[i, i]).sum() for i in range(n)])
            per = pd.DataFrame({"rule": rule, "cutoff": f"|z| >= {c}", "TF": tf, "Perturb-seq genes": d_.astype(int), "own share (%)": np.diag(S), "column z (own vs other knockdowns)": dc, "double-centred z": dg,
                                "rank of own knockdown in own list's column, raw (1 = best of n)": rk_col, "rank of own list in row, double-centred (1 = best of n)": rk_dcz_row, "n TFs": n})
            per.round(3).to_csv(f"{O}/{tag}_diagonal_per_tf.tsv", sep="\t", index=False)
            for en, exs in (("all TFs", set()), ("without NRF1, E4F1, DNMT1", EX3)):
                p = per[~per.TF.isin(exs)]
                SUM.append({"rule": rule, "cutoff": f"|z| >= {c}", "TFs": en, "n TFs (>= 10 Perturb-seq genes)": len(p), "diagonal column z, mean": p["column z (own vs other knockdowns)"].mean(), "diagonal column z, median": p["column z (own vs other knockdowns)"].median(),
                            "off-diagonal column z, mean": 0.0, "diagonal double-centred z, mean": p["double-centred z"].mean(), "diagonal double-centred z, median": p["double-centred z"].median(),
                            "TFs with double-centred z >= 2": int((p["double-centred z"] >= 2).sum()), "TFs with own list best in row (double-centred)": int((p["rank of own list in row, double-centred (1 = best of n)"] == 1).sum()),
                            "TFs with own list in top 5 of row (double-centred)": int((p["rank of own list in row, double-centred (1 = best of n)"] <= 5).sum()), "expected top 5 by chance": round(5 * len(p) / n, 1)})
            fig, ax = plt.subplots(1, 3, figsize=(30, 10.5))
            for a, M, ttl, cm_, lim in ((ax[0], S, "raw share found (%)", "viridis", None), (ax[1], CZ, "column z: each list vs other knockdowns' genes", "RdBu_r", 4), (ax[2], Rz, "double-centred z: row and column effects removed", "RdBu_r", 4)):
                im = a.imshow(M, cmap=cm_, vmin=-lim if lim else None, vmax=lim, aspect="equal", interpolation="nearest"); a.set_xticks(range(n)); a.set_yticks(range(n)); a.set_xticklabels(tf, rotation=90, fontsize=5); a.set_yticklabels(tf, fontsize=5)
                a.plot(range(n), range(n), "s", mfc="none", mec="k", ms=3.2, mew=.6); a.set_title(ttl, fontsize=11); a.set_xlabel("Regulome list (TF)"); a.set_ylabel("knockdown (Perturb-seq genes)"); fig.colorbar(im, ax=a, fraction=.046, pad=.02)
            fig.suptitle(f"Specificity matrix, rule {rule}, Perturb-seq genes |z| >= {c} (diagonal boxed = own list)", fontsize=13); fig.tight_layout(); fig.savefig(f"{O}/{tag}_heatmaps.png", dpi=200); fig.savefig(f"{O}/{tag}_heatmaps.pdf"); plt.close(fig)
            print(tag, n, flush=True)
    pd.DataFrame(SUM).round(3).to_excel(W, sheet_name="diagonal_summary", index=False)
pd.DataFrame(SUM).round(3).to_csv(f"{O}/diagonal_summary.tsv", sep="\t", index=False); print(pd.DataFrame(SUM).round(2).to_string(index=False)); print("done")
