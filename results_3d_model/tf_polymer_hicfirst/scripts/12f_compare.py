"""Compare TF-only (v2), loops + TF (hybrid), loops only, Hi-C-first, Hi-C-first + TF on COMMON held-out sets.
Hi-C: O/E Spearman on (a) the 176 held-out pixels, (b) all pixels not restrained in the Hi-C-first set (pixels, loops,
red edges), both 2-181 bins apart, kept bins only.  Edges: Cicero hub/shared/step-1 pairs whose 10 kb pixel (+-1 bin) is
NOT restrained in the Hi-C-first set ("held-out edges"; for hub edges these are the blue edges), aggregate ratio
obs/exp against same-distance and TF-bound matched nulls, from each model's validation_pairs.tsv.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
R = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model"); F = R / "tf_polymer_hicfirst"
M = {"TF only (v2)": (R / "tf_polymer_v2/analysis", 16), "loops + TF": (R / "tf_polymer_hic/analysis", 16),
     "loops only": (F / "loops_only/analysis", 0), "Hi-C first": (F / "hicfirst/analysis", 0), "Hi-C first + TF": (F / "hicfirst_tf/analysis", 16)}
I = F / "hicfirst/inputs"; H = np.load(I / "hic_10kb.npz"); ice, keep = H["ice"], H["keep"]; n = ice.shape[0]
RS = pd.read_csv(I / "restraints.tsv", sep="\t"); HP = pd.read_csv(I / "heldout_pixels.tsv", sep="\t")
RES = np.zeros((n, n), bool); RES[RS.bin1, RS.bin2] = True; RES |= RES.T
NEAR = np.zeros((n, n), bool)
for p, q in zip(RS[RS.kind != "pixel"].bin1, RS[RS.kind != "pixel"].bin2): NEAR[max(0, p - 1):p + 2, max(0, q - 1):q + 2] = True
NEAR |= RES; NEAR |= NEAR.T
HO = np.zeros((n, n), bool); HO[HP.bin1, HP.bin2] = True
def oe(Mx):
    E = np.full_like(Mx, np.nan, dtype=float)
    for d in range(n):
        i = np.arange(n - d); j = i + d; ok = keep[i] & keep[j] & np.isfinite(Mx[i, j])
        if ok.any():
            mu = Mx[i[ok], j[ok]].mean()
            if mu > 0: E[i, j] = Mx[i, j] / mu
    return E
iu = np.triu_indices(n, 2); base = keep[iu[0]] & keep[iu[1]] & np.isfinite(ice[iu]); Ho = oe(ice)
CP = pd.read_csv(I / "pairs_cicero.tsv", sep="\t")
rows = []
for name, (A, eps) in M.items():
    S = np.load(A / f"sim_10kb_eps{eps:g}.npy"); So = oe(S)
    ok = base & np.isfinite(So[iu]) & np.isfinite(Ho[iu])
    r = dict(model=name, oe_heldout=stats.spearmanr(So[iu][ok & HO[iu]], Ho[iu][ok & HO[iu]])[0],
             oe_unrestrained=stats.spearmanr(So[iu][ok & ~RES[iu]], Ho[iu][ok & ~RES[iu]])[0],
             raw_unrestrained=stats.spearmanr(S[iu][ok & ~RES[iu]], ice[iu][ok & ~RES[iu]])[0])
    VP = pd.read_csv(A / "validation_pairs.tsv", sep="\t"); VP = VP[VP.eps_top == eps]
    for st in ("hub_edges", "shared_edges", "cicero_step1"):
        for nl, tag in (("random_same_distance", "dist"), ("bound_matched_class_distance", "bound")):
            d = VP[(VP.set == st) & (VP.null == nl)]
            free = ~NEAR[d.i.values // 5, d.j.values // 5]
            d = d[free & d.exp.notna()]
            r[f"{st}_{tag}"] = d.obs.mean() / d.exp.mean(); r[f"{st}_n"] = len(d)
    V = pd.read_csv(A / "validation.tsv", sep="\t"); V = V[V.eps_top == eps]
    hv = V[V.set == "hub_cores_max_in_one_cluster"].iloc[0]; hp = V[V.set == "hub_core_pairs_cocluster"].iloc[0]
    r.update(hub3=hv.frac_conf_ge3, hub3_null=hv.null_frac_conf_ge3, hubpair=hp.agg_ratio)
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(F / "compare/model_comparison.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); print(T.round(3).T.to_string())
print("held-out pixels", int((base & HO[iu]).sum()), "unrestrained pixels", int((base & ~RES[iu]).sum()))
