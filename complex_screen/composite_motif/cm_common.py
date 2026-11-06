# Shared helpers for the composite-motif test: reversed design, foreground defined by sequence, our signal averaged over many open elements.
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/composite_motif"; U = f"{O}/union"
def load_elements(): return pd.read_csv(f"{U}/union.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid"])
def glm_mu(y, off, Xd):
    b = np.zeros(Xd.shape[1]); b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(30):
        eta = off + Xd @ b; mu = np.exp(eta); z = eta - off + (y - mu) / mu
        XtW = Xd.T * mu; bn = np.linalg.solve(XtW @ Xd + 1e-6 * np.eye(len(b)), XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-7: b = bn; break
        b = bn
    return np.exp(off + Xd @ b)
def residual(y, reads, pool_counts, pool_reads):
    """Gate 1 residual on union-ATAC elements: Poisson GLM on ATAC, leave-out pooled panel, width, GC, GC^2, k50 mappability"""
    C = pd.read_csv(f"{U}/covariates.tsv", sep="\t"); atac_cpm = C.atac.values / C.atac.sum() * 1e6
    pc = pool_counts / pool_reads * 1e6
    Xd = np.column_stack([np.ones(len(y)), np.log1p(atac_cpm), np.log1p(pc), np.log(C.width.values), C.gc.values, C.gc.values ** 2, C["map"].values])
    return glm_mu(y, np.full(len(y), np.log(reads / 1e6)), Xd)
def boot_diff(a, b, n=2000, seed=1):
    """mean(a) - mean(b) with a percentile bootstrap CI over elements"""
    r = np.random.default_rng(seed); d = [r.choice(a, len(a)).mean() - r.choice(b, len(b)).mean() for _ in range(n)]
    return a.mean() - b.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)
