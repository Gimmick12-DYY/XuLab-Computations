#!/usr/bin/env python
"""Round 11 step f (SPEC.md 6-7): ChIP-differential regions -> summit-group 6x6 (raw at ZNF777 depth, 3 seeds; imputed
Mdrop5) with statistics, and the heatmap source. Writes O11/{diff_overlap,summary}.tsv, m_*.tsv and
paper_figures/fig2/source_data/corr_chipdeseq2_6x6{,_imputed}.tsv, heatmap_chipdeseq2.npz."""
import itertools

import numpy as np
import pandas as pd
import pyBigWig
import scipy.sparse as sp

from r11_common import O11
from fig2_common import DM_DEPTH, DM_SEEDS, INP, OUT, ROOT, TFS, bin_index

SD = ROOT / "paper_figures/fig2/source_data"
G, NT, TOPN, CEN, FL0, FL1, E = 30, 6, 5000, 100, 1000, 2000, 1e-4
Rg = pd.read_csv(O11 / "regions.tsv", sep="\t")
CNT = pd.read_csv(O11 / "counts.tsv.gz", sep="\t")
BULK = np.column_stack([CNT[f"{t}_rep1"].values + CNT[f"{t}_rep2"].values for t in TFS])
D = {t: pd.read_csv(O11 / f"deseq2_{t}.tsv.gz", sep="\t") for t in TFS}
DIFF = {t: np.flatnonzero((np.nan_to_num(D[t].padj.values, nan=1) < 0.05) & (D[t].lfc.values > 1)) for t in TFS}
ov = pd.DataFrame([[len(np.intersect1d(DIFF[a], DIFF[b])) for b in TFS] for a in TFS], TFS, TFS)
ov.to_csv(O11 / "diff_overlap.tsv", sep="\t"); print("n differential / pairwise overlap\n", ov.to_string(), flush=True)

# selected sites: per ChIP, top 5,000 differential regions by LFC, represented by that ChIP's summit, sorted by signalValue
sel = {}
for t in TFS:
    ix = DIFF[t][np.argsort(-D[t].lfc.values[DIFF[t]], kind="stable")][:TOPN]
    ix = ix[Rg[f"summit_{t}"].values[ix] >= 0]
    sel[t] = ix[np.argsort(-Rg[f"sig_{t}"].values[ix], kind="stable")]
site_tf = np.concatenate([[t] * len(sel[t]) for t in TFS]); site_reg = np.concatenate([sel[t] for t in TFS])
chrom = Rg.chrom.values[site_reg]; summ = np.array([Rg[f"summit_{t}"].values[r] for t, r in zip(site_tf, site_reg)], np.int64)
print("sites per ChIP", {t: len(sel[t]) for t in TFS}, flush=True)

# raw single-cell counts per site for all 79 TFs (read 5' ends)
fr = np.load(INP / "frame.npz", allow_pickle=True); tf79 = [str(t) for t in fr["tf79"]]; ncell = fr["ncells_imp79"]
C79 = np.zeros((79, len(summ)), np.int64); F79 = np.zeros_like(C79); TOT = np.zeros(79)
W, NB = 1000, 40; edges = np.arange(-W, W + 1, 50)
PROF = {}   # heatmap: full-depth 5' end profiles for the 6 panel TFs and the pool
pool_prof = np.zeros((len(summ), NB))
for k, t in enumerate(tf79):
    z = np.load(INP / "five" / f"{t}.npz"); TOT[k] = sum(len(z[c]) for c in z.files)
    prof = np.zeros((len(summ), NB)) if t in TFS else None
    for c in np.unique(chrom):
        if c not in z.files:
            continue
        q = z[c]; m = np.flatnonzero(chrom == c); s = summ[m]
        cnt = lambda a, b: np.searchsorted(q, s + b, side="right") - np.searchsorted(q, s + a, side="left")
        C79[k, m] = cnt(-CEN, CEN); F79[k, m] = cnt(-FL1, -FL0 - 1) + cnt(FL0 + 1, FL1)
        lo = np.searchsorted(q, s - W); hi = np.searchsorted(q, s + W)
        for j, (r, a, b) in enumerate(zip(m, lo, hi)):
            h = np.histogram(q[a:b] - s[j], edges)[0]
            pool_prof[r] += h
            if prof is not None:
                prof[r] = h
    if prof is not None:
        PROF[t] = prof / TOT[k] * 1e6
pool_prof = pool_prof / TOT.sum() * 1e6
pi = [tf79.index(t) for t in TFS]; Cpool, Fpool = C79.sum(0), F79.sum(0)

# imputed (Mdrop5), r25 recipe
bi, off, N = bin_index()
b0 = np.array([off[c][0] for c in chrom]) + summ // 1000
nb = np.array([off[c][1] for c in chrom]); lo0 = np.array([off[c][0] for c in chrom])
FLB = np.r_[-5:-1, 2:6]; fb = np.clip(b0[:, None] + FLB[None, :], lo0[:, None], lo0[:, None] + nb[:, None] - 1)
IMP = sp.load_npz(INP / "imp79_mdrop5.npz").tocsr()
Ci = np.zeros((79, len(summ))); Fi = np.zeros((79, len(summ)))
HB = np.arange(-5, 6); hb = np.clip(b0[:, None] + HB[None, :], lo0[:, None], lo0[:, None] + nb[:, None] - 1)
IPROF, ipool = {}, np.zeros((len(summ), 11))
for k in range(79):
    v = IMP[k].toarray().ravel() / ncell[k]
    Ci[k] = v[b0]; Fi[k] = v[fb].mean(1); ipool += v[hb] / 79
    if tf79[k] in TFS:
        IPROF[tf79[k]] = v[hb]
mk = np.load(ROOT / "regulatome/m46_stage1_imputed/work/masks_extra.npz")["Mdrop5"]
reg = pd.read_csv(ROOT / "unified/work/nfya/impute/regions.tsv", header=None)[0].str.extract(r"^(chr[^:]+):(\d+)-")
kp = reg[0].isin(off).values; idx = np.array([off[c][0] if c in off else 0 for c in reg[0]]) + reg[1].astype(np.int64).values // 1000
fmask = np.zeros(N, bool); fmask[idx[kp]] = mk[kp]


def groups(half):
    cn = np.array([23 if c == "chrX" else int(c[3:]) for c in chrom])
    hm = {"all": np.ones(len(summ), bool), "odd": (cn % 2 == 1) & (cn != 23), "even_X": (cn % 2 == 0) | (cn == 23)}[half]
    out, st = [], 0
    for t in TFS:
        ix = np.arange(st, st + len(sel[t])); st += len(sel[t])
        out += [g[hm[g]] for g in np.array_split(ix, G)]
    return [g for g in out if len(g)]


def perm(M):
    k = len(M); o = ~np.eye(k, dtype=bool); s = lambda A: A.diagonal().mean() - A[o].mean()
    obs = s(M); null = np.array([s(M[list(p)]) for p in itertools.permutations(range(k))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


rows = []
SB = BULK[site_reg]
for half in ("all", "odd", "even_X"):
    gr = groups(half)
    gB = np.array([SB[g].sum(0) for g in gr]); bulk = np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)
    Ms = []
    for seed in DM_SEEDS:
        pf = np.minimum(1, DM_DEPTH / TOT[pi])[:, None]
        Cs = np.random.default_rng(seed).binomial(C79[pi], pf); Fs = np.random.default_rng(seed + 1).binomial(F79[pi], pf)
        sc = np.zeros((len(gr), NT))
        for i, k in enumerate(pi):
            for a, g in enumerate(gr):
                c, f = Cs[i, g].sum(), Fs[i, g].sum(); cp, fp = Cpool[g].sum() - C79[k, g].sum(), Fpool[g].sum() - F79[k, g].sum()
                sc[a, i] = np.log2((c + .5) / (f + .5)) - np.log2((cp + .5) / (fp + .5))
        Ms.append(np.corrcoef(np.column_stack([bulk, sc]).T)[NT:, :NT])
    Mi = np.zeros((len(gr), NT)); Cip, Fip = Ci.sum(0), Fi.sum(0)
    for i, k in enumerate(pi):
        for a, g in enumerate(gr):
            c, f = Ci[k, g].sum(), Fi[k, g].sum(); cp, fp = Cip[g].sum() - c, Fip[g].sum() - f
            Mi[a, i] = np.log2((c + E) / (f + E)) - np.log2((cp + E) / (fp + E))
    for arm, M in (("raw", np.mean(Ms, 0)), ("imputed", np.corrcoef(np.column_stack([bulk, Mi]).T)[NT:, :NT])):
        obs, p = perm(M); _, p5 = perm(M[1:, 1:])
        T = pd.DataFrame(M, [f"{t} sc" for t in TFS], [f"{t} ChIP" for t in TFS]); T.to_csv(O11 / f"m_{arm}_{half}.tsv", sep="\t", float_format="%.5f")
        if half == "all":
            T.to_csv(SD / ("corr_chipdeseq2_6x6.tsv" if arm == "raw" else "corr_chipdeseq2_6x6_imputed.tsv"), sep="\t", float_format="%.5f")
        rows.append(dict(arm=arm, half=half, n_groups=len(gr), diag_minus_off=obs, perm_p=p, noCTCF_perm_p=p5,
                         own_best_rows=int(sum(np.argmax(M[i]) == i for i in range(NT))), own_best_cols=int(sum(np.argmax(M[:, j]) == j for j in range(NT))),
                         own_is_row_max=",".join(t for i, t in enumerate(TFS) if np.argmax(M[i]) == i), **{f"own_{t}": M[i, i] for i, t in enumerate(TFS)}))
Sm = pd.DataFrame(rows); Sm.to_csv(O11 / "summary.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250); print(Sm.round(3).to_string(), flush=True)

# heatmap source (SPEC.md 7): rows averaged in groups of 25 within each block
GR = 25
def gavg(X, sizes):
    out, st = [], 0
    for n in sizes:
        B = X[st:st + n]; st += n
        out.append(np.vstack([b.mean(0) for b in np.array_split(B, max(1, int(np.ceil(n / GR))))]))
    return np.vstack(out)
sizes = [len(sel[t]) for t in TFS]
chip = np.zeros((len(summ), NB))
for t in TFS:
    bw = pyBigWig.open(str(OUT / "round6_diffpeaks/heatmaps/bw" / f"{t}.bw"))
    m = np.flatnonzero(site_tf == t)
    for r in m:
        v = bw.stats(chrom[r], int(summ[r] - W), int(summ[r] + W), nBins=NB, type="mean")
        chip[r] = np.nan_to_num(np.array(v, float))
    chip[m] /= max(np.percentile(chip[m], 99), 1e-9)
P = gavg(pool_prof, sizes)
raw = np.stack([np.log2((gavg(PROF[t], sizes) + 0.05) / (P + 0.05)) for t in TFS])
msk = fmask[hb]; IP = gavg(np.where(msk, ipool, np.nan), sizes)
imp = np.stack([np.log2((gavg(np.where(msk, IPROF[t], np.nan), sizes) + E) / (IP + E)) for t in TFS])
gsz = [max(1, int(np.ceil(n / GR))) for n in sizes]
np.savez_compressed(SD / "heatmap_chipdeseq2.npz", tfs=np.array(TFS), chip=gavg(chip, sizes).astype(np.float32), raw=raw.astype(np.float32),
                    imp=imp.astype(np.float32), block_n=np.array(sizes), block_rows=np.array(gsz))
# block summary: centre (+-200 bp) raw log2 enrichment and imputed centre bin, mean per block x TF
st, bs = 0, []
for b, t in enumerate(TFS):
    sl = slice(st, st + gsz[b]); st += gsz[b]
    bs.append(dict(block=f"{t} differential", n=sizes[b], **{f"raw_{u}": float(raw[i, sl, 16:24].mean()) for i, u in enumerate(TFS)},
                   **{f"imp_{u}": float(np.nanmean(imp[i, sl, 5])) for i, u in enumerate(TFS)}))
Bs = pd.DataFrame(bs); Bs.to_csv(O11 / "heatmap_blocks.tsv", sep="\t", index=False, float_format="%.4f"); print(Bs.round(3).to_string())
print("done")
