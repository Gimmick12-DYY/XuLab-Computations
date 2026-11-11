# m7_step2 step C: per-gene and per-(TF, gene) features on one ABC link set. Standardisation (per feature, per draw) uses train-chromosome genes only.
# usage: build_features.py <raw|imp> [link_set_name=powerlaw]. Writes features/<kind>_<link>/{base,real,pseudo0-4,mis0-4,refs}.npz and feature_qc.json.
import gzip, json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome"; O = f"{R}/m7_step2"; G1 = f"{R}/gate1/cellmats"
kind = sys.argv[1]; link = sys.argv[2] if len(sys.argv) > 2 else "powerlaw"; assert kind in ("raw", "imp")
ABC = {"powerlaw": f"{R}/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", "hicfixed": f"{R}/m2c_abc_canonical/results/HEK293T_hicfixed/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv"}[link]; CANON = link != "powerlaw"
Out = f"{O}/features/{kind}_{link}"; os.makedirs(Out, exist_ok=True); SEED = 20261111; ND = 5
tfs = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]; NT = len(tfs)
G = pd.read_csv(f"{O}/splits/gene_split.tsv", sep="\t"); genes = G.gene.values; NG = len(genes); gidx = pd.Series(np.arange(NG), index=genes); train = (G.set == "train").values
P = pd.read_csv(f"{O}/pair_list.tsv", sep="\t"); ta = {t: i for i, t in enumerate(tfs)}; pa = P.a.map(ta).values; pb = P.b.map(ta).values
E_ = pd.read_csv(f"{R}/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"]); act = E_.cls.isin(["promoter", "distal_active", "bivalent"]).values; Panel = E_[act].reset_index(drop=True)
CL = pd.read_csv(f"{R}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end", "ATAC.RPM", "H3K27ac.RPM"]) if CANON else None
Ea = Panel if not CANON else CL.rename(columns={"chr": "chrom"})[["chrom", "start", "end"]]; NE = len(Ea)
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}
A = pd.read_csv(ABC, sep="\t", usecols=["chr", "start", "end", "TargetGene", "TargetGeneTSS", "ABC.Score", "isSelfPromoter"]); A = A[A.TargetGene.isin(gidx.index)].copy()
A["e"] = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord); assert A.e.notna().all(); A["e"] = A.e.astype(int); A["g"] = A.TargetGene.map(gidx).values; A["self"] = A.isSelfPromoter.astype(str).str.lower().eq("true").values
gchr = G.set_index("gene").chromosome; gset = G.set_index("gene").set
qc = dict(kind=kind, link_set=link, n_genes=NG, n_elements=NE, n_links=int(len(A)), n_self_promoter_links=int(A.self.sum()), genes_without_self_promoter=int(NG - A[A.self].g.nunique()))
qc["links_element_chrom_differs_from_gene_chrom"] = int((A.chr.values != A.TargetGene.map(gchr).values).sum()); elem_set = A.chr.map(lambda c: "test" if c in {"chr3", "chr11", "chr17", "chr20"} else ("val" if c in {"chr5", "chr12"} else "train")); qc["links_element_split_differs_from_gene_split"] = int((elem_set.values != A.TargetGene.map(gset).values).sum())
Lp = sp.csr_matrix((np.ones(int(A.self.sum())), (A.g[A.self], A.e[A.self])), shape=(NG, NE)); Ld = sp.csr_matrix((np.ones(int((~A.self).sum())), (A.g[~A.self], A.e[~A.self])), shape=(NG, NE)); La = Lp + Ld
# ATAC and H3K27ac at active elements
if CANON: atac_cpm = CL["ATAC.RPM"].values.astype(float); atac = atac_cpm; k27 = CL["H3K27ac.RPM"].values.astype(float); qc["k27ac_source"] = "canonical EnhancerList H3K27ac.RPM"
else:
    atac = np.load(f"{R}/m1_binding/v2/counts/atac_counts.npz")["atac"][np.flatnonzero(act)].astype(float); assert len(atac) == NE; atac_cpm = atac / atac.sum() * 1e6
    K = pd.read_csv(f"{R}/m2b_abc_k27ac/results/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end", "H3K27ac.RPM"])
    K["e"] = (K.chr + ":" + K.start.astype(str) + "-" + K.end.astype(str)).map(coord); qc["k27ac_elements_in_candidates_mapped"] = int(K.e.notna().sum()); qc["active_elements_without_k27ac"] = int(NE - K.e.notna().sum()); K = K.dropna(subset=["e"]); k27 = np.zeros(NE); k27[K.e.astype(int).values] = K["H3K27ac.RPM"].values
# CpG islands (UCSC cpgIslandExt hg38) overlapping a promoter element
cg = pd.read_csv(f"{O}/annotation/cpgIslandExt.txt.gz", sep="\t", header=None, usecols=[1, 2, 3], names=["c", "s", "e"], compression="gzip"); ov = np.zeros(NE, bool)
for c, H in cg.groupby("c"):
    ix = np.flatnonzero(Ea.chrom.values == c)
    if not len(ix): continue
    Hs = H.sort_values("s"); hi = np.searchsorted(Hs.s.values, Ea.end.values[ix], side="left")  # CGIs starting before element end
    cummax_e = np.maximum.accumulate(Hs.e.values); ok = hi > 0; ov[ix[ok]] = cummax_e[hi[ok] - 1] > Ea.start.values[ix[ok]]
cgi = np.asarray(Lp @ ov.astype(float)).ravel() > 0; qc["cgi_genes"] = int(cgi.sum()); qc["cgi_islands"] = int(len(cg))
# binding CPM per element, 79 TFs
def panel_cpm(t):
    y = np.asarray(sp.load_npz(f"{G1}/{t}.{kind}.npz").sum(0)).ravel().astype(float); return y / max(y.sum(), 1e-12) * 1e6
if not CANON: B = np.column_stack([panel_cpm(t) for t in tfs])
elif kind == "raw":
    C = np.load(f"{O}/features/canon_counts/raw_counts.npy"); assert C.shape[0] == NE; B = C.astype(float) / np.maximum(C.sum(0), 1e-12) * 1e6
else:  # imputed tracks exist on panel elements only: overlap-bp-fraction weighted transfer to canonical elements (Amendment 4)
    pc = Panel.chrom.values; pst, pen = Panel.start.values, Panel.end.values; ro, co, va = [], [], []
    for c, H in Ea.groupby("chrom"):
        ix = np.flatnonzero(pc == c); o = np.argsort(pst[ix]); ix = ix[o]; s_ = pst[ix]; e_ = pen[ix]
        for g_, (a_, b_) in zip(H.index.values, zip(H.start.values, H.end.values)):
            lo = np.searchsorted(s_, a_, side="right") - 1; lo = max(lo, 0)
            j = lo
            while j < len(s_) and s_[j] < b_:
                ov_ = min(e_[j], b_) - max(s_[j], a_)
                if ov_ > 0: ro.append(g_); co.append(ix[j]); va.append(ov_ / (e_[j] - s_[j]))
                j += 1
    W_ov = sp.csr_matrix((va, (ro, co)), shape=(NE, len(Panel))); B = np.asarray(W_ov @ np.column_stack([panel_cpm(t) for t in tfs])); B = B / np.maximum(B.sum(0), 1e-12) * 1e6; qc["imputed_panel_elements_mapped"] = int(W_ov.nnz)
pool = B.sum(1)
def std_col(v):
    m, s = v[train].mean(), v[train].std(); return (v - m) / (s if s > 0 else 1.0)
ABCd = A[~A.self].groupby("g")["ABC.Score"].sum().reindex(np.arange(NG)).fillna(0).values; nl = A.groupby("g").size().reindex(np.arange(NG)).fillna(0).values
base = np.column_stack([std_col(np.log1p(np.asarray(Lp @ atac_cpm).ravel())), std_col(np.log1p(np.asarray(Lp @ k27).ravel())), std_col(np.log1p(ABCd)), std_col(np.log1p(nl)), cgi.astype(float), std_col(np.log1p(np.asarray(Lp @ pool).ravel())), std_col(np.log1p(np.asarray(Ld @ pool).ravel()))]).astype(np.float32)
np.savez(f"{Out}/base.npz", base=base, names=np.array(["promoter_atac", "promoter_h3k27ac", "abc_distal_sum", "n_links", "cgi_flag", "pool_promoter", "pool_distal"]))
def feats(Bm):
    prom = np.asarray(Lp @ Bm); dist = np.asarray(Ld @ Bm); pair = np.column_stack([np.asarray(La @ np.minimum(Bm[:, a], Bm[:, b])).ravel() for a, b in zip(pa, pb)])
    f = lambda M: np.column_stack([std_col(np.log1p(M[:, j])) for j in range(M.shape[1])]).astype(np.float32)
    return dict(prom=f(prom).T, dist=f(dist).T, pair=f(pair).T)  # TF x gene; pair x gene
np.savez(f"{Out}/real.npz", **feats(B)); print("real done", flush=True)
# pseudo-TF: each TF's element values permuted within ATAC-decile strata (matched accessibility)
dec = pd.qcut(pd.Series(atac).rank(method="first"), 10, labels=False).values  # ATAC deciles of this element set; DEC = [np.flatnonzero(dec == k) for k in range(10)]
for d in range(ND):
    Bp = B.copy()
    for i in range(NT):
        rng = np.random.default_rng(SEED + 1000 * (d + 1) + i)
        for ix in DEC: Bp[ix, i] = B[ix[rng.permutation(len(ix))], i]
    np.savez(f"{Out}/pseudo{d}.npz", **feats(Bp)); print("pseudo", d, flush=True)
# mis-paired: each TF's binding replaced by another TF's from the same depth tertile (derangement within tertile)
F = pd.read_csv(f"{R}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf"); ter = F.tertile.reindex(tfs).values; assert pd.notna(ter).all(); MAP = {}
for d in range(ND):
    rng = np.random.default_rng(SEED + 900 + d); sig = np.arange(NT)
    for tr in ("low", "mid", "high"):
        ix = np.flatnonzero(ter == tr)
        while True:
            pm = rng.permutation(ix)
            if (pm != ix).all(): break
        sig[ix] = pm
    MAP[f"mis{d}"] = [tfs[j] for j in sig]; np.savez(f"{Out}/mis{d}.npz", **feats(B[:, sig])); print("mis", d, flush=True)
qc["mis_paired_map"] = MAP
if not CANON:  # references are defined on the powerlaw/panel build and shared (Amendment 3)
    # reference predictors: plain TSS +/- 100 kb raw-CPM sum over elements (m6_improve win100k) and frozen v8 top-1000 membership
    mid = ((Ea.start + Ea.end) // 2).values; Gt = A.drop_duplicates("g").set_index("g").TargetGeneTSS.reindex(np.arange(NG)); ro, co = [], []
    for ch, Gs in G.groupby("chromosome"):
        ix = np.flatnonzero(Ea.chrom.values == ch); o = np.argsort(mid[ix]); ix = ix[o]; m = mid[ix]; gi = gidx[Gs.gene.values].values; t_ = Gt.values[gi]
        lo = np.searchsorted(m, t_ - 100000); hi = np.searchsorted(m, t_ + 100000, side="right")
        for g_, a_, b_ in zip(gi, lo, hi): ro += [g_] * int(b_ - a_); co += list(ix[int(a_):int(b_)])
    W = sp.csr_matrix((np.ones(len(ro)), (ro, co)), shape=(NG, NE)); win = np.asarray(W @ B).T.astype(np.float32)
    v8 = np.zeros((NT, NG), bool)
    for i, t in enumerate(tfs):
        S = pd.read_csv(f"{R}/r6_v8/lists/{t}.tsv", sep="\t").gene; v8[i, gidx.reindex(S).dropna().astype(int).values] = True
    qc["v8_genes_in_universe_mean"] = float(v8.sum(1).mean())
    np.savez(f"{Out}/refs.npz", win100k=win, v8=v8)
json.dump(qc, open(f"{Out}/feature_qc.json", "w"), indent=1, default=str); print(json.dumps(qc, default=str)[:1500])
