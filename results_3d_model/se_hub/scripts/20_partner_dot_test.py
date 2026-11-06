"""Dot test for each SE-candidate partner pixel, 10 kb, same measure as the clique triangle slide: centre = 3x3 O/E (expected = chr19 mean per diagonal), donut = Chebyshev distance 2-5, excluding the centre's row and column and pixels within 2 bins of the diagonal; enrichment = centre / donut. Reported at the exact pixel and with the best shift of each end within +-20 kb.
Null per partner: 1,000 random chr19 pixels at the same distance, same search. A real dot: shifted enrichment >= 1.5, >= 30 reads in the centre, empirical p < 0.05."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/clique_triangle/scripts"); from hicread import Hic
warnings.simplefilter("ignore"); Bd = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/se_hub"); O = Bd / "figure_v6_hubstyle"; r = 10000; H = Hic(sys.argv[1], r); L = int(H.g["chroms/length"][H.names.index("chr19")]); M, R, w = H.fetch("chr19", 0, L); nb = M.shape[0]; valid = np.isfinite(w); DMAX = 320; exp = np.full(DMAX + 1, np.nan); rng = np.random.default_rng(19)
for d in range(DMAX + 1):
    v = np.diagonal(M, d); ok = valid[:nb - d] & valid[d:] & np.isfinite(v)
    if ok.any(): exp[d] = v[ok].mean()
di, dj = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6), indexing="ij"); ring = (np.maximum(abs(di), abs(dj)) >= 2) & (di != 0) & (dj != 0)
def one(i, j):
    if i < 5 or j + 6 > nb or j - i < 5: return np.nan, np.nan, 0
    W = M[i - 5:i + 6, j - 5:j + 6]; D = np.clip((j + dj) - (i + di), 0, DMAX); Ew = W / exp[D]; m = ring & ((j + dj) - (i + di) >= 2); bg = np.nanmean(Ew[m]); cen = np.nanmean(Ew[4:7, 4:7]); return (cen / bg if bg > 0 else np.nan), cen, R[i - 1:i + 2, j - 1:j + 2].sum()
def search(i, j):
    best = (np.nan, np.nan, 0, 0, 0); ex = one(i, j)
    for a in (-2, -1, 0, 1, 2):
        for b in (-2, -1, 0, 1, 2):
            x = one(i + a, j + b)
            if x[2] >= 30 and not np.isnan(x[0]) and (np.isnan(best[0]) or x[0] > best[0]): best = (*x, a, b)
    return ex, best
S = pd.read_csv(Bd / "figure_v2_cand/spokes.tsv", sep="\t").sort_values("pos"); v0 = 125; rows = []
for s in S.itertuples():
    b = s.pos // r; i, j = sorted((v0, b)); d = j - i; ex, be = search(i, j); nul = []
    for _ in range(1000):
        for _t in range(30):
            a = int(rng.integers(10, nb - d - 10))
            if valid[a] and valid[a + d]: break
        x = search(a, a + d)[1][0]; nul.append(x)
    nul = np.array(nul, float); nul = nul[np.isfinite(nul)]; p = float((1 + (nul >= be[0]).sum()) / (len(nul) + 1)) if np.isfinite(be[0]) else 1.0
    rows.append(dict(label=s.label, cls=s.cls, bin=b, span_kb=s.span_kb, exact_don=ex[0], exact_oe=ex[1], exact_reads=ex[2], best_don=be[0], best_oe=be[1], best_reads=be[2], shift_kb_i=be[3] * 10, shift_kb_j=be[4] * 10, null_median=float(np.median(nul)), p=p, n_null=len(nul))); print(s.label, flush=True)
T = pd.DataFrame(rows); T["real_dot"] = (T.best_don >= 1.5) & (T.best_reads >= 30) & (T.p < 0.05); T.to_csv(O / "partner_dots.tsv", sep="\t", index=False); print(T[["label", "span_kb", "exact_don", "exact_reads", "best_don", "best_reads", "shift_kb_i", "shift_kb_j", "null_median", "p", "real_dot"]].round(3).to_string(index=False)); print("real dots", int(T.real_dot.sum()), "exact >= 1.5:", int((T.exact_don >= 1.5).sum()))
