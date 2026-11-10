# Follow-up summary (usage: bench_summary_multi.py <dir> <reference_variant> new:old [...]) of <dir>/per_tf.tsv,
# same rules as ../bench_summary.py. Original header: Summary of bench/per_tf.tsv as m6_improve/spec_summary.py (b3b4 ceiling weights, bootstrap over TFs 10,000, seed 20261116), with and without NRF1.
# Per variant: weighted specificity + CI, unweighted, median own-KD rank, B6 effect (mean own z), mean top-5% logOR; paired weighted differences vs
# list e over old links (oldS_raw) and vs the element-count / pooled-other-TF / ATAC controls of the same link set and weighting. Plus the two
# H3K27ac contrasts: newX - oldX and newS - oldS. Writes bench/summary.tsv and bench/contrasts.tsv.
import sys
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab/regulatome"; O = sys.argv[1]; REF = sys.argv[2]; CON = [(c.split(":")[0] + w, c.split(":")[1] + w) for c in sys.argv[3:] for w in ("X", "S")]  # new:old -> newX/oldX and newS/oldS
P = pd.read_csv(f"{O}/per_tf.tsv", sep="\t"); C = pd.read_csv(f"{X}/m4_validation/b3b4/reliability_ceilings.tsv", sep="\t")
LAB = {"oldX_raw@1000": "a binding x ABC_old", "newX_raw@1000": "b binding x ABC_k27ac", "oldS_raw@1000": "c binding-only, old links (list e)", "newS_raw@1000": "d binding-only, new links"}
out, con = [], []
for excl in (False, True):
    rng = np.random.default_rng(20261116)
    for a in ["arm2", "arm1"]:
        w = C[C.arm == a].groupby("panel_tf").weight.mean(); G = P[(P.arm == a) & ~(excl & (P.tf == "NRF1"))]
        piv = lambda col: G.pivot(index="tf", columns="variant", values=col)
        W = piv("specificity").dropna(); Z = piv("z_own").loc[W.index]; R = piv("own_rank").loc[W.index]; L = piv("logor5").loc[W.index]
        wt = w.reindex(W.index).fillna(0).values; ix = rng.integers(0, len(W), (10000, len(W))); f = lambda e: (e[ix] * wt[ix]).sum(1) / np.maximum(wt[ix].sum(1), 1e-12)
        def wd(e):
            b = f(e); return round(np.average(e, weights=wt), 3), round(np.percentile(b, 2.5), 3), round(np.percentile(b, 97.5), 3)
        for v in W.columns:
            e = W[v].values; m, lo, hi = wd(e); pre = v.split("_")[0]
            r = dict(nrf1="excluded" if excl else "included", arm=a, variant=v, label=LAB.get(v, ""), n_tfs=len(e), n_weighted=int((wt > 0).sum()), wspec=m, wspec_ci=f"[{lo}, {hi}]",
                     unweighted=round(e.mean(), 3), median_own_rank=float(R[v].median()), n_kd=int(G.n_kd.iloc[0]), effect_own_z=round(Z[v].mean(), 3), logOR5=round(L[v].mean(), 3))
            for tag, cv in (("ref", REF), ("ctrl_n_elem", f"{pre}_CTRL_n_elem@1000"), ("ctrl_pool", f"{pre}_CTRL_pool@1000"), ("ctrl_atac", f"{pre}_CTRL_atac@1000")):
                if cv == v or cv not in W: r[f"d_{tag}"] = np.nan; r[f"d_{tag}_ci"] = "."; continue
                m2, l2, h2 = wd(e - W[cv].values); r[f"d_{tag}"] = m2; r[f"d_{tag}_ci"] = f"[{l2}, {h2}]"
            out.append(r)
        for x, y in CON:
            for k in ("raw", "CTRL_n_elem", "CTRL_pool", "CTRL_atac"):
                m2, l2, h2 = wd(W[f"{x}_{k}@1000"].values - W[f"{y}_{k}@1000"].values)
                con.append(dict(nrf1="excluded" if excl else "included", arm=a, contrast=f"{x}_{k} - {y}_{k}", wspec_diff=m2, ci=f"[{l2}, {h2}]",
                                n_tfs_better=int((W[f"{x}_{k}@1000"] > W[f"{y}_{k}@1000"]).sum()), n_tfs=len(W)))
S = pd.DataFrame(out).sort_values(["nrf1", "arm", "wspec"], ascending=[True, True, False]); S.to_csv(f"{O}/summary.tsv", sep="\t", index=False)
K = pd.DataFrame(con); K.to_csv(f"{O}/contrasts.tsv", sep="\t", index=False)
# list overlap old vs new
Lr = pd.read_csv(f"{O}/lists_raw.tsv.gz", sep="\t"); ov = []
for t, G in Lr.groupby("tf"):
    s = {v: set(G.gene[G.variant == v]) for v in G.variant.unique()}
    for x, y in [(f"{x}_raw@1000", f"{y}_raw@1000") for x, y in CON]: ov.append(dict(tf=t, pair=f"{x} vs {y}", overlap=len(s[x] & s[y]) / max(len(s[y]), 1)))
OV = pd.DataFrame(ov).groupby("pair").overlap.describe()[["mean", "min", "50%", "max"]]; OV.to_csv(f"{O}/list_overlap.tsv", sep="\t")
pd.set_option("display.width", 320); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 30)
print(S.to_string(index=False)); print(K.to_string(index=False)); print(OV.to_string())
