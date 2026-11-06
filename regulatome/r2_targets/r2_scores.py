# R2 direct ABC target scores: S(q,g) = sum_e Btilde(q,e) x ABC(e,g) x (1 + kappa Cic_q(e,g)), kappa arms 0 and 1.
# Btilde = Gate 1 v2 per-TF element residual, log2((obs+.5)/(mu+.5)), Poisson GLM: log mu = log(reads) + b0 + log1p(ATAC CPM) +
#   log1p(leave-one-out pooled panel CPM) + log(width) + GC + GC^2 + k50 mappability, all cells of the TF, active elements (promoter,
#   distal_active, bivalent). ABC = power-law links at thr 0.02 (m2_abc), as in Gate 1. Cic_q = 1 when the TF's first-step Cicero
#   pair (> 25 kb) joins the element and the gene's TSS (+/- 1 kb); only ABC-linked element-gene pairs can gain. kappa 1 = kappa 0
#   for TFs without Cicero pairs. Count-aware flag: expected reads over the gene's linked elements >= 5, the Gate 1 v2.1 rule.
#   usage: r2_scores.py prep | tf <TF>
import sys
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; G = f"{X}/regulatome/gate1"; V = f"{G}/v2"; O = f"{X}/regulatome/r2_targets"
E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"])
act = np.flatnonzero(E.cls.isin(["promoter", "distal_active", "bivalent"]).values)
Ea = E.iloc[act].reset_index(drop=True); nE = len(Ea)
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
if sys.argv[1] == "prep":
    Y = np.zeros((len(tfs), nE)); rd = np.zeros(len(tfs))
    for i, t in enumerate(tfs):
        Y[i] = np.asarray(sp.load_npz(f"{G}/cellmats/{t}.raw.npz").sum(0)).ravel(); rd[i] = np.load(f"{G}/cellmats/{t}.cells.npz")["reads"].sum()
    np.savez_compressed(f"{O}/panel_counts.npz", Y=Y, reads=rd, tfs=np.array(tfs)); print("prep done", Y.shape); sys.exit()
t = sys.argv[2]; P = np.load(f"{O}/panel_counts.npz"); Y, rd = P["Y"], P["reads"]; i = tfs.index(t)
gc = pd.read_csv(f"{V}/gc.tsv", sep="\t", header=None, index_col=0)[1].reindex(Ea.eid).values
mp = pd.read_csv(f"{V}/map.tsv", sep="\t", header=None, index_col=0)[1].reindex(Ea.eid).values
atac = np.load(f"{X}/regulatome/m1_binding/v2/counts/atac_counts.npz")["atac"][act].astype(float); atac_cpm = atac / atac.sum() * 1e6
logw = np.log(Ea.end.values - Ea.start.values)
coord = {f"{c}:{s}-{e}": k for k, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}
A = pd.read_csv(f"{X}/regulatome/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "TargetGeneTSS", "ABC.Score"])
A["e"] = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord); A = A.dropna(subset=["e"]); A["e"] = A.e.astype(int)
genes = np.array(sorted(A.TargetGene.unique())); A["g"] = A.TargetGene.map({g: k for k, g in enumerate(genes)})
W = sp.csr_matrix((A["ABC.Score"].values, (A.e.values, A.g.values)), shape=(nE, len(genes))); L = W.copy(); L.data[:] = 1.0
def cicero_W():
    f = f"{X}/cobinding/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv" if t == "RBBP4" else f"{X}/cobinding/work/ourrun_k50/{t}/{t}.peak_edges.tsv"  # RBBP4's Our Run lives apart
    try: C = pd.read_csv(f, sep="\t")
    except FileNotFoundError: return None
    def parse(s):
        p = s.str.extract(r"^(\w+):(\d+)-(\d+)$"); return p[0].values, p[1].astype(int).values, p[2].astype(int).values
    c1, s1, e1 = parse(C.peak1); c2, s2, e2 = parse(C.peak2); keep = np.abs((s1 + e1) / 2 - (s2 + e2) / 2) > 25000
    tss = A.drop_duplicates("TargetGene")[["chr", "TargetGeneTSS", "g"]]
    tss_by = {c: (d.TargetGeneTSS.values.astype(int), d.g.values) for c, d in tss.groupby("chr")}
    el_by = {c: (d.start.values, d.end.values, d.index.values) for c, d in Ea.groupby("chrom", sort=False)}
    rows, cols = [], []
    for ca, sa, ea, cb, sb, eb in zip(c1[keep], s1[keep], e1[keep], c2[keep], s2[keep], e2[keep]):
        for (xc, xs, xe), (pc, ps, pe) in [((ca, sa, ea), (cb, sb, eb)), ((cb, sb, eb), (ca, sa, ea))]:
            if pc not in tss_by or xc not in el_by: continue
            t_, g_ = tss_by[pc]; gs = g_[(t_ >= ps - 1000) & (t_ <= pe + 1000)]
            if not len(gs): continue
            s_, e_, ix = el_by[xc]
            for e in ix[(s_ < xe) & (e_ > xs)]:
                for g in gs: rows.append(e); cols.append(g)
    if not rows: return None
    C = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nE, len(genes))); C.data[:] = 1.0
    return C
def glm_mu(y, off, Xd):
    b = np.zeros(Xd.shape[1]); b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(30):
        eta = off + Xd @ b; mu = np.exp(eta); z = eta - off + (y - mu) / mu
        XtW = Xd.T * mu; bn = np.linalg.solve(XtW @ Xd + 1e-6 * np.eye(len(b)), XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-7: b = bn; break
        b = bn
    return np.exp(off + Xd @ b)
y = Y[i]; pc = (Y.sum(0) - y) / (rd.sum() - rd[i]) * 1e6
Xd = np.column_stack([np.ones(nE), np.log1p(atac_cpm), np.log1p(pc), logw, gc, gc ** 2, mp])
mu = glm_mu(y, np.full(nE, np.log(rd[i] / 1e6)), Xd); B = np.log2((y + 0.5) / (mu + 0.5))
Cm = cicero_W(); S0 = W.T @ B
if Cm is None: S1, nc = S0.copy(), 0
else:
    gain = W.multiply(Cm).tocsr(); S1 = S0 + gain.T @ B; nc = gain.nnz
cpm = y / rd[i] * 1e6
T = pd.DataFrame(dict(gene=genes, S_k0=S0, S_k1=S1, n_elements=np.asarray(L.sum(0)).ravel().astype(int), reads=L.T @ y, expected=L.T @ mu,
                      S_atac=W.T @ np.log1p(atac_cpm), S_rawcpm=W.T @ np.log1p(cpm)))
T["count_ok"] = T.expected >= 5
T["rank_k0"] = T.S_k0.rank(ascending=False, method="first").astype(int); T["rank_k1"] = T.S_k1.rank(ascending=False, method="first").astype(int)
T.sort_values("rank_k0").to_csv(f"{O}/per_tf/{t}.tsv.gz", sep="\t", index=False, float_format="%.5g")
print(t, "cicero-boosted links:", nc, "genes:", len(T), "pos:", int((S0 > 0).sum()), "count_ok:", int(T.count_ok.sum()))
