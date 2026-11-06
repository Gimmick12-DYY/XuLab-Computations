"""Score the 3 Hi-C dots of each clique triangle at 10 kb and 5 kb. O/E = balanced contact / mean balanced contact at the same distance in the local window (+-1 Mb around the triangle). Centre = 3x3 pixels
on the anchor midpoints; donut = pixels at Chebyshev distance 2-5 from the centre, excluding its row and column and pixels within 2 bins of the diagonal; dot enrichment = centre O/E / donut O/E. Ranked by the minimum
enrichment over the 3 dots. Excluded from the showcase list: anchors within 5 kb of SERPINH1, DNAJB1, BRD4, RPS3 or TECR, chr11:75.3-75.7 Mb, CTCF-only triangles."""
import gzip, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); from hicread import Hic
IN = sys.argv[1]; O = Path(__file__).resolve().parents[1]; PAD = 1000000
T = pd.read_csv(O / "triangles_all.tsv", sep="\t"); T = T[(T.AB_kb >= 50) & (T.BC_kb >= 50) & (T.AC_kb <= 1500)].reset_index(drop=True)
Hs = {r: Hic(f"{IN}/wt.mcool", r) for r in (10000, 5000)}
def score(H, c, ms):
    r = H.res; L0 = max(0, min(ms) - PAD) // r * r; L1 = max(ms) + PAD; M, R, w = H.fetch(c, L0, L1); n = M.shape[0]; M = np.where(np.isfinite(M), M, np.nan); M[M == 0] = np.where(R[M == 0] > 0, np.nan, 0)
    bad = ~np.isfinite(w); M[bad, :] = np.nan; M[:, bad] = np.nan
    exp = np.array([np.nanmean(np.diagonal(M, d)) if np.isfinite(np.diagonal(M, d)).any() else np.nan for d in range(n)]); ij = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    with np.errstate(all="ignore"): E = M / exp[ij]
    b = [(m - L0) // r for m in ms]; out = {}
    for nm, (i, j) in dict(AB=(b[0], b[1]), BC=(b[1], b[2]), AC=(b[0], b[2])).items():
        if j + 5 >= n or i < 5: out.update({f"{nm}_oe": np.nan, f"{nm}_don": np.nan, f"{nm}_reads": 0}); continue
        cen = E[i - 1:i + 2, j - 1:j + 2]; v = []
        for di in range(-5, 6):
            for dj in range(-5, 6):
                if max(abs(di), abs(dj)) < 2 or di == 0 or dj == 0 or (j + dj) - (i + di) < 2: continue
                v.append(E[i + di, j + dj])
        bg = np.nanmean(v); oe = np.nanmean(cen); out.update({f"{nm}_oe": oe, f"{nm}_pix": E[i, j], f"{nm}_don": oe / bg if bg > 0 else np.nan, f"{nm}_reads": R[i - 1:i + 2, j - 1:j + 2].sum()})
    return out
rows = []
for k, t in enumerate(T.itertuples()):
    d = {}
    for r, H in Hs.items():
        d.update({f"{a}_{r // 1000}k": v for a, v in score(H, t.chrom, (t.mA, t.mB, t.mC)).items()})
    rows.append(d)
    if k % 400 == 0: print(k, flush=True)
S = pd.concat([T, pd.DataFrame(rows)], axis=1)
for r in ("10k", "5k"):
    S[f"min_don_{r}"] = S[[f"AB_don_{r}", f"BC_don_{r}", f"AC_don_{r}"]].min(axis=1, skipna=False); S[f"min_oe_{r}"] = S[[f"AB_oe_{r}", f"BC_oe_{r}", f"AC_oe_{r}"]].min(axis=1, skipna=False); S[f"min_reads_{r}"] = S[[f"AB_reads_{r}", f"BC_reads_{r}", f"AC_reads_{r}"]].min(axis=1)
G = pd.read_csv(gzip.open(f"{IN}/genes_exons.tsv.gz", "rt"), sep="\t", header=None, names="c t s e st g ty".split()); by = {c: d for c, d in G.groupby("c")}
ART = {"SERPINH1", "DNAJB1", "BRD4", "RPS3", "TECR"}
def pr(x): c, a = x.split(":"); s, e = map(int, a.split("-")); return c, s, e
def annot(x):
    c, s, e = pr(x); d = by.get(c); gn = d[(d.t == "gene") & (d.e >= s - 5000) & (d.s <= e + 5000)]; art = bool(set(gn.g) & ART) or (c == "chr11" and s < 75700000 and e > 75300000)
    g2 = d[(d.t == "gene") & d.ty.isin(["pc", "lnc"])]; tss = np.where(g2.st == "+", g2.s, g2.e); pm = g2[(tss >= s - 2500) & (tss <= e + 2500)].g.unique(); ex = d[(d.t == "exon") & (d.ty == "pc") & (d.e >= s) & (d.s <= e)]
    return art, "/".join(sorted(pm))[:30], len(ex) > 0
rec = {x: annot(x) for x in set(S.A) | set(S.B) | set(S.C)}
for a in "ABC":
    S[f"{a}_art"], S[f"{a}_tss"], S[f"{a}_exon"] = zip(*[rec[x] for x in S[a]])
S["artifact"] = S.A_art | S.B_art | S.C_art; S["ctcf_only"] = S.tfs == "CTCF"; S["chr16p11"] = (S.chrom == "chr16") & (S.mA < 31000000) & (S.mC > 29000000)
S.to_csv(O / "triangles_scored.tsv", sep="\t", index=False)
print("scored", len(S), "artifact", int(S.artifact.sum()), "ctcf only", int(S.ctcf_only.sum()), "chr16p11", int(S.chr16p11.sum()))
K = S[~S.artifact & ~S.ctcf_only]
for th in (1.2, 1.5, 2, 3):
    print("min enrichment >=", th, "10 kb:", int((K.min_don_10k >= th).sum()), " 5 kb:", int((K.min_don_5k >= th).sum()), " with min O/E >= 2 (10 kb):", int(((K.min_don_10k >= th) & (K.min_oe_10k >= 2)).sum()))
print(K[["min_don_10k", "min_oe_10k", "min_reads_10k"]].describe().round(2).loc[["25%", "50%", "75%", "max"]].to_string())
c = "A B C AB_kb BC_kb n_tfs tfs A_tss B_tss C_tss A_exon B_exon C_exon AB_oe_10k BC_oe_10k AC_oe_10k AB_don_10k BC_don_10k AC_don_10k min_don_10k min_don_5k min_reads_10k".split()
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
print(K[K.min_reads_10k >= 30].sort_values("min_don_10k", ascending=False)[c].head(15).round(2).to_string(index=False))
