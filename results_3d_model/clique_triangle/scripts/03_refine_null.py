"""Let each anchor move up to +-20 kb (10 kb bins; the loop rule allows +-30 kb), one shift per anchor shared by its 2 dots, and take the shift with the best minimum dot enrichment. The same search on a null: for each
triangle, one random location on the same chromosome with the same spans. Empirical count above each threshold = real vs null. Writes triangles_refined.tsv and null_refined.tsv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); from hicread import Hic
IN = sys.argv[1]; O = Path(__file__).resolve().parents[1]; PAD = 1000000; r = 10000; H = Hic(f"{IN}/wt.mcool", r); rng = np.random.default_rng(11); SH = (-2, -1, 0, 1, 2)
S = pd.read_csv(O / "triangles_scored.tsv", sep="\t"); S = S[~S.artifact & ~S.ctcf_only].reset_index(drop=True)
di, dj = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6), indexing="ij"); ring = (np.maximum(abs(di), abs(dj)) >= 2) & (di != 0) & (dj != 0)
def best(c, ms):
    L0 = max(0, min(ms) - PAD) // r * r; L1 = max(ms) + PAD; M, R, w = H.fetch(c, L0, L1); n = M.shape[0]
    if n < 30: return None
    bad = ~np.isfinite(w); M = M.copy(); M[bad, :] = np.nan; M[:, bad] = np.nan
    exp = np.array([np.nanmean(np.diagonal(M, d)) if np.isfinite(np.diagonal(M, d)).any() else np.nan for d in range(n)]); ij = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    with np.errstate(all="ignore"): E = M / exp[ij]
    b = [(m - L0) // r for m in ms]
    if b[0] < 8 or b[2] + 8 >= n: return None
    def dmap(x, y):
        o = {}
        for a in SH:
            for c_ in SH:
                i, j = b[x] + a, b[y] + c_
                if j - i < 5: o[a, c_] = (np.nan, np.nan, 0); continue
                cen = E[i - 1:i + 2, j - 1:j + 2]; w_ = E[i - 5:i + 6, j - 5:j + 6]; m = ring & ((j + dj) - (i + di) >= 2)
                with np.errstate(all="ignore"):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore"); bg = np.nanmean(w_[m]); oe = np.nanmean(cen)
                o[a, c_] = (oe / bg if bg > 0 else np.nan, oe, R[i - 1:i + 2, j - 1:j + 2].sum())
        return o
    AB, BC, AC = dmap(0, 1), dmap(1, 2), dmap(0, 2); bst = None
    for a in SH:
        for b_ in SH:
            for c_ in SH:
                v = (AB[a, b_], BC[b_, c_], AC[a, c_]); mn = min(x[0] for x in v)
                if np.isnan(mn) or min(x[2] for x in v) < 30: continue
                if bst is None or mn > bst["min_don"]:
                    bst = dict(min_don=mn, sA=a * r // 1000, sB=b_ * r // 1000, sC=c_ * r // 1000, AB_don=v[0][0], BC_don=v[1][0], AC_don=v[2][0], AB_oe=v[0][1], BC_oe=v[1][1], AC_oe=v[2][1], AB_reads=v[0][2], BC_reads=v[1][2], AC_reads=v[2][2])
    return bst
rows, nul = [], []
for k, t in enumerate(S.itertuples()):
    rows.append(best(t.chrom, (t.mA, t.mB, t.mC)) or {})
    n0 = H.g["chroms/length"][H.names.index(t.chrom)]; sp = t.mC - t.mA
    for _ in range(20):
        a = int(rng.integers(PAD + 200000, max(PAD + 200001, n0 - sp - PAD - 200000))); x = best(t.chrom, (a, a + t.mB - t.mA, a + sp))
        if x: nul.append(dict(chrom=t.chrom, mA=a, **x)); break
    if k % 300 == 0: print(k, flush=True)
Rf = pd.concat([S, pd.DataFrame(rows).add_prefix("ref_")], axis=1); Rf.to_csv(O / "triangles_refined.tsv", sep="\t", index=False); N = pd.DataFrame(nul); N.to_csv(O / "null_refined.tsv", sep="\t", index=False)
print("real", Rf.ref_min_don.notna().sum(), "null", len(N), "medians", round(Rf.ref_min_don.median(), 3), round(N.min_don.median(), 3))
for th in (1.2, 1.5, 1.75, 2, 2.5, 3): print(th, "real", int((Rf.ref_min_don >= th).sum()), "null", int((N.min_don >= th).sum()))
c = "A B C AB_kb BC_kb tfs A_tss B_tss C_tss ref_sA ref_sB ref_sC ref_AB_oe ref_BC_oe ref_AC_oe ref_AB_don ref_BC_don ref_AC_don ref_min_don".split()
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40); print(Rf.sort_values("ref_min_don", ascending=False)[c].head(14).round(2).to_string(index=False))
print("chr16p11", Rf[Rf.chr16p11].ref_min_don.describe().round(2).to_dict())
