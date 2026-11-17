# m36 (SPEC.md): binary matrices, 2x2 overlap per TF, cutoff table. Descriptive.
import sys
import numpy as np, pandas as pd
RG_ = "/work/users/d/y/dyy12/XuLab/regulatome"
UG = np.load(f"{RG_}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True)["genes"].astype(str); NG = len(UG); gi = {g: k for k, g in enumerate(UG)}
O = f"{RG_}/m36_binary_overlap/out"; OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
T25 = pd.read_csv(f"{RG_}/m23_sgrna_top25/out/observed_kd_top25.tsv.gz", sep="\t", usecols=["tf", "gene", "z_top25"]); TFS = sorted(set(T25.tf))
Z = T25.pivot(index="tf", columns="gene", values="z_top25").reindex(index=TFS, columns=UG).values.astype(float)
for i, t in enumerate(TFS):
    for o in OWN.get(t, [t]):
        if o in gi: Z[i, gi[o]] = np.nan
E = pd.read_csv(f"{RG_}/m34_fig5_panels_v3/out/regulome_v1.tsv.gz", sep="\t", usecols=["tf", "gene"]); B = np.zeros((len(TFS), NG), np.int8)
for i, t in enumerate(TFS):
    g = E.gene[E.tf == t]; B[i, [gi[x] for x in g if x in gi]] = 1
OK = np.isfinite(Z); A = np.abs(np.nan_to_num(Z))
def pb(cut):
    if cut == "top5pct":
        M = np.zeros_like(OK)
        for i in range(len(TFS)): idx = np.flatnonzero(OK[i]); M[i, idx[np.argsort(-A[i, idx])[:int(round(0.05 * len(idx)))]]] = True
        return M
    return OK & (A >= float(cut))
pd.DataFrame(np.where(OK, B, -1), index=TFS, columns=UG).to_csv(f"{O}/regulome_binary.tsv.gz", sep="\t")
for c in ("2", "3"): pd.DataFrame(np.where(OK, pb(c).astype(int), -1), index=TFS, columns=UG).to_csv(f"{O}/perturbseq_binary_z{c}.tsv.gz", sep="\t")
rows = []
for c in ("2", "3"):
    P = pb(c)
    for i, t in enumerate(TFS):
        ok = OK[i]; b = (B[i] == 1) & ok; p = P[i] & ok; n = int(ok.sum())
        r = dict(TF=t, cutoff=f"|z| >= {c}", genes=n, ours=int(b.sum()), perturbseq=int(p.sum()), both=int((b & p).sum()), ours_only=int((b & ~p).sum()), perturbseq_only=int((~b & p).sum()), neither=int((~b & ~p & ok).sum()))
        r["expected_both"] = r["ours"] * r["perturbseq"] / n; r["recovery_pct"] = 100 * r["both"] / r["perturbseq"] if r["perturbseq"] else np.nan; r["precision_pct"] = 100 * r["both"] / r["ours"] if r["ours"] else np.nan
        r["expected_recovery_pct"] = 100 * r["ours"] / n; r["expected_precision_pct"] = 100 * r["perturbseq"] / n; r["fold_both_over_expected"] = r["both"] / r["expected_both"] if r["expected_both"] else np.nan
        r["jaccard"] = r["both"] / max(r["both"] + r["ours_only"] + r["perturbseq_only"], 1); r["ge10_responders"] = r["perturbseq"] >= 10; rows.append(r)
T = pd.DataFrame(rows); T.insert(0, "label", "descriptive; top-25% sgRNA cells; Regulome v1 binding"); T.to_csv(f"{O}/overlap_per_tf.tsv", sep="\t", index=False, float_format="%.3f")
PO = []
for c, g in T.groupby("cutoff", sort=False):
    s = g[["genes", "ours", "perturbseq", "both", "ours_only", "perturbseq_only", "neither", "expected_both"]].sum()
    PO.append(dict(label="descriptive", cutoff=c, n_tfs=len(g), **{k: (float(v) if k == "expected_both" else int(v)) for k, v in s.items()}, recovery_pct=100 * s.both / s.perturbseq, expected_recovery_pct=100 * s.expected_both / s.perturbseq,
                   precision_pct=100 * s.both / s.ours, expected_precision_pct=100 * s.expected_both / s.ours, fold_both_over_expected=s.both / s.expected_both, jaccard=s.both / (s.both + s.ours_only + s.perturbseq_only)))
PO = pd.DataFrame(PO); PO.to_csv(f"{O}/overlap_pooled.tsv", sep="\t", index=False, float_format="%.3f")
C = []
for c in ("2", "2.5", "3", "4", "top5pct"):
    P = pb(c); resp = (P & OK).sum(1); both = ((B == 1) & P & OK).sum(1); ours = ((B == 1) & OK).sum(1); n = OK.sum(1); exp = ours * resp / n
    C.append(dict(cutoff=("top 5% of genes per TF" if c == "top5pct" else f"|z| >= {c}"), median_responders_per_tf=float(np.median(resp)), tfs_ge10_responders=int((resp >= 10).sum()), total_responders=int(resp.sum()), total_both=int(both.sum()),
                  pooled_recovery_pct=100 * both.sum() / resp.sum(), expected_recovery_pct=100 * exp.sum() / resp.sum(), pooled_precision_pct=100 * both.sum() / ours.sum(), expected_precision_pct=100 * exp.sum() / ours.sum()))
C = pd.DataFrame(C); C.to_csv(f"{O}/cutoff_table.tsv", sep="\t", index=False, float_format="%.2f")
sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
for f, d in (("overlap_per_tf", T), ("overlap_pooled", PO), ("cutoff_table", C)): d.to_excel(f"{O}/{f}.xlsx", index=False)
long = []
for i, t in enumerate(TFS):
    for j in np.flatnonzero(OK[i] & ((B[i] == 1) | (A[i] >= 2))): long.append((t, UG[j], int(B[i, j]), int(A[i, j] >= 2), int(A[i, j] >= 3), round(float(Z[i, j]), 3)))
L = pd.DataFrame(long, columns=["TF", "gene", "regulome", "perturbseq_z2", "perturbseq_z3", "z_top25"]); L.to_csv(f"{O}/binary_long_nonzero.tsv.gz", sep="\t", index=False)
pd.set_option("display.width", 250); print(PO.drop(columns="label").round(2).to_string(index=False)); print(C.round(2).to_string(index=False))
print(T[T.cutoff == "|z| >= 2"].nlargest(10, "both")[["TF", "ours", "perturbseq", "both", "ours_only", "perturbseq_only", "expected_both", "recovery_pct", "expected_recovery_pct", "precision_pct", "jaccard"]].round(2).to_string(index=False))
print("long rows", len(L))
