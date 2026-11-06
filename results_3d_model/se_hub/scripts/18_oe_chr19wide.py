"""SE-core virtual 4C with the standard expected: mean balanced contact per diagonal over the whole of chr19 at 10 kb, ignoring filtered bins (same code path as 05_hic_test.py). Checks that the 12 partner pixels reproduce hic_test/edges_oe_10kb.tsv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/clique_triangle/scripts"); from hicread import Hic
Bd = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/se_hub"); O = Bd / "figure_v5_tftracks"; r = 10000; H = Hic(sys.argv[1], r); L = int(H.g["chroms/length"][H.names.index("chr19")]); M, R, w = H.fetch("chr19", 0, L); nb = M.shape[0]; valid = np.isfinite(w); DMAX = 300; exp = np.full(DMAX + 1, np.nan)
for d in range(DMAX + 1):
    v = np.diagonal(M, d); ok = valid[:nb - d] & valid[d:] & np.isfinite(v)
    if ok.any(): exp[d] = v[ok].mean()
v0 = 125; bins = np.arange(43, 225); dd = np.abs(bins - v0); oe = np.where(valid[bins] & valid[v0], M[v0, bins] / exp[np.minimum(dd, DMAX)], np.nan); oe[dd < 2] = np.nan; reads = R[v0, bins]
pd.DataFrame(dict(bin_start=bins * r, obs_ice=M[v0, bins], exp_chr19=exp[np.minimum(dd, DMAX)], oe=oe, reads=reads)).to_csv(O / "virtual4c_10kb_chr19exp.tsv", sep="\t", index=False)
E = pd.read_csv(Bd / "hic_test/edges_oe_10kb.tsv", sep="\t"); E = E[E.set == "loop_edges_18"].copy(); E["b"] = np.where(E.i == v0, E.j, E.i); E["mine"] = [M[v0, b] / exp[abs(b - v0)] for b in E.b]; E["reads"] = [R[v0, b] for b in E.b]; print("max diff vs test", float((E.mine - E.oe).abs().max()))
S = pd.read_csv(Bd / "figure_v2_cand/spokes.tsv", sep="\t"); C = pd.read_csv(Bd / "figure_v2_cand/virtual4c_10kb.tsv", sep="\t"); rows = []
for x in S.sort_values("pos").itertuples():
    b = x.pos // r; rows.append((x.label, round(float(C.oe[C.bin_start == b * r].iloc[0]), 2), round(float(oe[bins == b][0]), 2), int(reads[bins == b][0])))
T = pd.DataFrame(rows, columns=["partner", "oe_window", "oe_chr19", "reads"]); print(T.to_string(index=False)); print("above 1: window", int((T.oe_window > 1).sum()), "chr19", int((T.oe_chr19 > 1).sum())); print("corr window vs chr19 all bins", round(pd.Series(oe).corr(C.oe.reindex(range(len(oe))).reset_index(drop=True)), 3))
