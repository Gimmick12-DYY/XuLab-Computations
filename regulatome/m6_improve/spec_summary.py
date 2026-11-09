# SPEC.md summary for an M6 result folder (per_tf.tsv; development TFs only). Per arm, with and without NRF1:
# ceiling-weighted specificity (b3b4 weights, bootstrap over TFs 10,000, seed 20261116) with CI, unweighted specificity, median own-KD rank,
# mean own z (B6 effect, 200 random sets), mean top-5% logOR if present, and weighted paired differences vs the reference and vs the
# element-count and pooled-other-TF controls of the same linking (falls back to the 50 kb window controls). Writes <dir>/spec_summary.tsv.
# usage: spec_summary.py <dir> <reference_variant>
import re, sys
import numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/regulatome/m6_improve"; dd, ref = sys.argv[1], sys.argv[2]
P = pd.read_csv(f"{O}/{dd}/per_tf.tsv", sep="\t"); C = pd.read_csv(f"{O}/../m4_validation/b3b4/reliability_ceilings.tsv", sep="\t"); out = []
def ctrl(v, cols, kind):
    pre = v.split("_")[0]; cands = [c for c in cols if kind in c and (c.startswith(pre + "_") or c.startswith(f"CTRL_{pre}"))]
    if not cands: cands = [c for c in cols if kind in c and "win50k" in c]
    cands = [c for c in cands if c.endswith("@1000")] or cands; return cands[0] if cands else None
for excl in (False, True):
    rng = np.random.default_rng(20261116)
    for a in ["arm2", "arm1"]:
        w = C[C.arm == a].groupby("panel_tf").weight.mean(); G = P[(P.arm == a) & ~(excl & (P.tf == "NRF1"))]
        piv = lambda col: G.pivot(index="tf", columns="variant", values=col)
        W = piv("specificity").dropna(); Z = piv("z_own").loc[W.index]; R = piv("own_rank").loc[W.index]; L = piv("logor5").loc[W.index] if "logor5" in G else None
        wt = w.reindex(W.index).fillna(0).values; ix = rng.integers(0, len(W), (10000, len(W))); f = lambda e: (e[ix] * wt[ix]).sum(1) / np.maximum(wt[ix].sum(1), 1e-12)
        def wd(e):
            b = f(e); return round(np.average(e, weights=wt), 3), round(np.percentile(b, 2.5), 3), round(np.percentile(b, 97.5), 3)
        for v in W.columns:
            e = W[v].values; m, lo, hi = wd(e); r = dict(nrf1="excluded" if excl else "included", arm=a, variant=v, n_tfs=len(e), n_weighted=int((wt > 0).sum()), wspec=m, wspec_ci=f"[{lo}, {hi}]",
                     unweighted=round(e.mean(), 3), median_own_rank=float(R[v].median()), n_kd=int(G.n_kd.iloc[0]), effect_own_z=round(Z[v].mean(), 3), logOR5=round(L[v].mean(), 3) if L is not None else np.nan)
            for tag, cv in (("ref", ref), ("ctrl_n_elem", ctrl(v, W.columns, "n_elem")), ("ctrl_pool", ctrl(v, W.columns, "pool"))):
                if cv is None or cv == v: r[f"d_{tag}"] = np.nan; r[f"d_{tag}_ci"] = "."; continue
                m2, l2, h2 = wd(e - W[cv].values); r[f"d_{tag}"] = m2; r[f"d_{tag}_ci"] = f"[{l2}, {h2}]"; r[f"{tag}_variant"] = cv
            out.append(r)
S = pd.DataFrame(out).sort_values(["nrf1", "arm", "wspec"], ascending=[True, True, False]); S.to_csv(f"{O}/{dd}/spec_summary.tsv", sep="\t", index=False)
pd.set_option("display.width", 320); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 30)
print(S.drop(columns=[c for c in S.columns if c.endswith("_variant")]).to_string(index=False))
