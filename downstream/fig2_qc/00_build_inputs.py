#!/usr/bin/env python
"""Build the shared 1 kb-bin inputs for Figure 2 QC.

Writes OUT/inputs/:
  frame.npz   universe, atac_union, atac_fc (OmniATAC pooled fold-enrichment mean), gc, cpg_oe,
              promoter (GENCODE v46 basic TSS +-1 kb), ref (7xN bool), refcount, unique (7xN bool),
              shared3, tf79, reads79, ncells_imp79
  raw_panel.npy    7xN raw sc reads per bin (5' end, not deduplicated), TFS order
  pooled_raw.npy  N, all 79 TFs summed
  imp_panel.npy    7xN imputed cells-positive per bin
  imp79_frac.npy  N, mean over 79 TFs of (cells positive / n cells)  [pooled imputed]
  imp79.npz   79xN sparse imputed cells-positive
  unique_<TF>.bed, shared3_bins.bed
"""
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pyBigWig
import scipy.sparse as sp

from fig2_common import (NT, ATAC_BW, ATAC_UNION, BIN, BLACKLIST, COUNTS, FASTA, GTF, INP, REF, ROOT, TFS,
                         bin_index, interval_mask, read_bed)

INP.mkdir(parents=True, exist_ok=True)
bi, off, N = bin_index()
chroms = list(bi.chrom)

# ---------- universe ----------
universe = np.zeros(N, bool)
for c in chroms:
    if c != "chrY":
        o, n = off[c]
        universe[o:o + n] = True
universe &= ~interval_mask(read_bed(BLACKLIST), off, N)
print("universe bins", universe.sum(), flush=True)

atac_union = interval_mask(read_bed(ATAC_UNION), off, N)

# ---------- ATAC fold-enrichment ----------
atac_fc = np.zeros(N, np.float32)
bw = pyBigWig.open(str(ATAC_BW))
bwc = bw.chroms()
for c in chroms:
    if c not in bwc:
        continue
    o, n = off[c]
    L = bwc[c]
    nfull = L // BIN
    v = bw.stats(c, 0, nfull * BIN, nBins=nfull, type="mean")
    atac_fc[o:o + nfull] = np.nan_to_num(np.array(v, dtype=float))
    if L > nfull * BIN and nfull < n:
        atac_fc[o + nfull] = np.nan_to_num(bw.stats(c, nfull * BIN, L, type="mean")[0] or 0)
print("atac fc done", flush=True)

# ---------- GC and CpG o/e ----------
fai = pd.read_csv(str(FASTA) + ".fai", sep="\t", header=None, names=["c", "len", "off", "lb", "lw"]).set_index("c")
gc = np.full(N, np.nan, np.float32)
cpg = np.full(N, np.nan, np.float32)
mm_all = np.memmap(FASTA, dtype=np.uint8, mode="r")
for c in chroms:
    o, n = off[c]
    L, fo, lb, lw = (int(x) for x in fai.loc[c, ["len", "off", "lb", "lw"]])
    q, r = divmod(L, lb)
    seq = mm_all[fo:fo + q * lw].reshape(q, lw)[:, :lb].ravel()
    if r:
        seq = np.concatenate([seq, mm_all[fo + q * lw: fo + q * lw + r]])
    seq = seq & 0xDF  # upper-case
    b = np.arange(L) // BIN
    isC, isG, isN = seq == ord("C"), seq == ord("G"), seq == ord("N")
    nC = np.bincount(b, isC, n)
    nG = np.bincount(b, isG, n)
    nN = np.bincount(b, isN, n)
    cg = np.zeros(L, bool)
    cg[:-1] = isC[:-1] & isG[1:]
    nCG = np.bincount(b, cg, n)
    Lb = np.bincount(b, minlength=n).astype(float) - nN
    with np.errstate(divide="ignore", invalid="ignore"):
        gc[o:o + n] = np.where(Lb > 500, (nC + nG) / Lb, np.nan)
        cpg[o:o + n] = np.where((Lb > 500) & (nC > 0) & (nG > 0), nCG * Lb / (nC * nG), np.nan)
    del seq, isC, isG, isN, cg, b
    print("seq", c, flush=True)

# ---------- promoters ----------
tss = []
g = pd.read_csv(GTF, sep="\t", comment="#", header=None, usecols=[0, 2, 3, 4, 6])
g = g[g[2] == "transcript"]
t = np.where(g[6] == "+", g[3] - 1, g[4] - 1)
prom = pd.DataFrame({0: g[0].values, 1: np.maximum(t - 1000, 0), 2: t + 1001})
promoter = interval_mask(prom, off, N)
print("promoter bins", promoter.sum(), flush=True)

# ---------- references ----------
ref = np.zeros((NT, N), bool)
for k, tf in enumerate(TFS):
    ref[k] = interval_mask(read_bed(REF[tf]), off, N)
refcount = ref.sum(0)
tmp = tempfile.mkdtemp(dir=INP)
for tf in TFS:
    read_bed(REF[tf])[[0, 1, 2]].sort_values([0, 1]).to_csv(f"{tmp}/{tf}.bed", sep="\t", header=False, index=False)
unique = np.zeros((NT, N), bool)
npk = {}
for k, tf in enumerate(TFS):
    others = [f"{tmp}/{o}.bed" for o in TFS if o != tf]
    res = subprocess.run(["bedtools", "intersect", "-v", "-a", f"{tmp}/{tf}.bed", "-b", *others],
                         check=True, capture_output=True, text=True).stdout
    (INP / f"unique_{tf}.bed").write_text(res)
    ub = read_bed(INP / f"unique_{tf}.bed")
    npk[tf] = (sum(1 for _ in open(f"{tmp}/{tf}.bed")), len(ub))
    unique[k] = interval_mask(ub, off, N) & (refcount == 1)
shared3 = (refcount >= 3)
for tf in TFS:
    print(tf, "peaks", npk[tf][0], "unique peaks", npk[tf][1], flush=True)

# ---------- raw sc reads ----------
stats = pd.read_csv(COUNTS / "tf_stats.tsv", sep="\t")
raw = sp.load_npz(COUNTS / "raw.npz").tocsr()
assert raw.shape == (len(stats), N), raw.shape
rs = np.asarray(raw.sum(1)).ravel()
assert np.allclose(rs, stats.reads.values), "raw.npz row order != tf_stats"
tf79 = list(stats.tf)
raw7 = np.vstack([raw[tf79.index(t)].toarray().ravel() for t in TFS]).astype(np.int32)
pooled = np.asarray(raw.sum(0)).ravel().astype(np.int64)
np.save(INP / "raw_panel.npy", raw7)
np.save(INP / "pooled_raw.npy", pooled)

# ---------- imputed ----------
def load_csr(path):
    try:
        return sp.load_npz(path).tocsr()
    except Exception:
        z = np.load(path)
        return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))


rows, ncell = [], []
for t in tf79:
    w = ROOT / "unified/work" / t.lower() / "impute"
    reg = pd.read_csv(w / "regions.tsv", header=None)[0].str.extract(r"^(chr[^:]+):(\d+)-")
    idx = np.array([off[c][0] for c in reg[0]]) + reg[1].astype(np.int64).values // BIN
    m = load_csr(w / "matrix_csr.npz")
    if m.shape[0] != len(reg):
        m = m.T.tocsr()
    assert m.shape[0] == len(reg)
    v = np.zeros(N, np.float32)
    v[idx] = np.asarray((m > 0).sum(1)).ravel()
    rows.append(sp.csr_matrix(v))
    ncell.append(m.shape[1])
    print("imputed", t, m.shape, flush=True)
imp79 = sp.vstack(rows).tocsr()
sp.save_npz(INP / "imp79.npz", imp79)
ncell = np.array(ncell, float)
np.save(INP / "imp_panel.npy", np.vstack([imp79[tf79.index(t)].toarray().ravel() for t in TFS]).astype(np.float32))
np.save(INP / "imp79_frac.npy", np.asarray((sp.diags(1 / ncell) @ imp79).mean(0)).ravel().astype(np.float32))

sh = np.flatnonzero(shared3)
cc = np.searchsorted(bi.offset.values, sh, side="right") - 1
pd.DataFrame({0: bi.chrom.values[cc], 1: (sh - bi.offset.values[cc]) * BIN,
              2: (sh - bi.offset.values[cc] + 1) * BIN}).to_csv(INP / "shared3_bins.bed", sep="\t", header=False, index=False)

np.savez_compressed(INP / "frame.npz", universe=universe, atac_union=atac_union, atac_fc=atac_fc, gc=gc,
                    cpg_oe=cpg, promoter=promoter, ref=ref, refcount=refcount.astype(np.int8), unique=unique,
                    shared3=shared3, tf79=np.array(tf79), reads79=stats.reads.values,
                    cells79=stats.cells.values, ncells_imp79=ncell,
                    npeaks=np.array([npk[t][0] for t in TFS]), nunique_peaks=np.array([npk[t][1] for t in TFS]))
print("done", flush=True)
