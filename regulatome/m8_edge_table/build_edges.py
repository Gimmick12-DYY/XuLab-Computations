# m8_edge_table: Regulome edge table, one row per TF and gene (SPEC.md). Descriptive; nothing is fitted to or compared with Perturb-seq.
# usage: build_edges.py <hicfixed|powerlaw>  -> edges/edge_table_<link>.tsv.gz, edges/qc_<link>.json
# Null (section 3): for each TF, 200 draws; each draw permutes the TF's per-element fragment counts among candidate elements within the same
# ATAC decile x H3K27ac decile stratum (100 strata), which keeps the TF's fragment total and its per-element count distribution within each stratum
# (the m7_step2 pseudo-TF permutation, with the H3K27ac decile added). Gene-level promoter / distal / total binding is recomputed per draw.
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import rankdata
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome"; M7 = f"{R}/m7_step2"; O = f"{R}/m8_edge_table"; os.makedirs(f"{O}/edges", exist_ok=True)
link = sys.argv[1]; ND = 200; SEED = 20261111
tfs = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]; NT = len(tfs)
G = pd.read_csv(f"{M7}/splits/gene_split_wide.tsv", sep="\t", usecols=["gene", "chromosome"]); genes = G.gene.values; NG = len(genes); gidx = pd.Series(np.arange(NG), index=genes)
ABC = {"powerlaw": f"{R}/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", "hicfixed": f"{R}/m2c_abc_canonical/results/HEK293T_hicfixed/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv"}[link]
# candidate elements, ATAC, H3K27ac and raw fragment counts per element (as m7_step2 build_features.py, Amendment 4)
if link == "hicfixed":
    CL = pd.read_csv(f"{R}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end", "ATAC.RPM", "H3K27ac.RPM"]).rename(columns={"chr": "chrom"})
    Ea = CL[["chrom", "start", "end"]]; atac = CL["ATAC.RPM"].values.astype(float); k27 = CL["H3K27ac.RPM"].values.astype(float)
    C = np.load(f"{M7}/features/canon_counts/raw_counts.npy").astype(np.float64)
else:
    E_ = pd.read_csv(f"{R}/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"]); act = E_.cls.isin(["promoter", "distal_active", "bivalent"]).values; Ea = E_[act].reset_index(drop=True)[["chrom", "start", "end"]]
    atac = np.load(f"{R}/m1_binding/v2/counts/atac_counts.npz")["atac"][np.flatnonzero(act)].astype(float)
    coord0 = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}
    K = pd.read_csv(f"{R}/m2b_abc_k27ac/results/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end", "H3K27ac.RPM"]); K["e"] = (K.chr + ":" + K.start.astype(str) + "-" + K.end.astype(str)).map(coord0); K = K.dropna(subset=["e"]); k27 = np.zeros(len(Ea)); k27[K.e.astype(int).values] = K["H3K27ac.RPM"].values
    C = np.column_stack([np.asarray(sp.load_npz(f"{R}/gate1/cellmats/{t}.raw.npz").sum(0)).ravel() for t in tfs]).astype(np.float64)
NE = len(Ea); assert C.shape == (NE, NT); tot = C.sum(0); CPM = C / np.maximum(tot, 1e-12) * 1e6
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}
# links
A = pd.read_csv(ABC, sep="\t", usecols=["chr", "start", "end", "TargetGene", "TargetGeneTSS", "isSelfPromoter"]); A = A[A.TargetGene.isin(gidx.index)].copy()
A["e"] = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord); qc = dict(link=link, n_genes=NG, n_tfs=NT, n_elements=NE, n_links=int(len(A)), links_unmapped=int(A.e.isna().sum())); A = A.dropna(subset=["e"])
A["e"] = A.e.astype(int); A["g"] = A.TargetGene.map(gidx).values; A["self"] = A.isSelfPromoter.astype(str).str.lower().eq("true").values
Lp = sp.csr_matrix((np.ones(int(A.self.sum())), (A.g[A.self], A.e[A.self])), shape=(NG, NE)); Ld = sp.csr_matrix((np.ones(int((~A.self).sum())), (A.g[~A.self], A.e[~A.self])), shape=(NG, NE)); La = ((Lp + Ld) > 0).astype(np.float64)
n_prom = np.asarray(Lp.sum(1)).ravel().astype(int); n_dist = np.asarray(Ld.sum(1)).ravel().astype(int)
qc.update(genes_without_any_link=int(((n_prom + n_dist) == 0).sum()), genes_without_self_promoter=int((n_prom == 0).sum()))
# loop_supported: a distal link whose element midpoint lies in one anchor and the gene's TSS in the other anchor of the same Peakachu 0.7 loop (10 kb anchors)
L7 = pd.read_csv(f"{X}/hic/work/loops/peakachu_10000.loops.0.7.bedpe", sep="\t", header=None, usecols=[0, 1, 3, 4]); L7.columns = ["c1", "s1", "c2", "s2"]
loops = set(zip(L7.c1, L7.s1 // 10000, L7.s2 // 10000)) | set(zip(L7.c1, L7.s2 // 10000, L7.s1 // 10000))
Dd = A[~A.self]; mid = ((Ea.start.values + Ea.end.values) // 2)[Dd.e.values]
hit = np.array([(c, int(m) // 10000, int(t) // 10000) in loops for c, m, t in zip(Dd.chr.values, mid, Dd.TargetGeneTSS.values)])
loop_sup = np.zeros(NG, bool); loop_sup[Dd.g.values[hit]] = True; qc.update(peakachu_loops=int(len(L7)), distal_links_loop_supported=int(hit.sum()), genes_loop_supported=int(loop_sup.sum()))
# strata: ATAC decile x H3K27ac decile
da = pd.qcut(pd.Series(atac).rank(method="first"), 10, labels=False).values; dk = pd.qcut(pd.Series(k27).rank(method="first"), 10, labels=False).values; st = da * 10 + dk
STR = [np.flatnonzero(st == k) for k in range(100)]; qc["stratum_size_min"] = int(min(len(s) for s in STR)); qc["stratum_size_max"] = int(max(len(s) for s in STR))
def bh(p):
    n = len(p); o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1); q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(q, 1); return out
FT = pd.read_csv(f"{R}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf"); SG = pd.read_csv(f"{R}/m6_improve/signed_v1/tf_sign_annotation.tsv", sep="\t").set_index("tf")
v8 = {t: set(pd.read_csv(f"{R}/r6_v8/lists/{t}.tsv", sep="\t").gene) for t in tfs}
cols = {}; called = np.zeros((NT, NG), bool)
for i, t in enumerate(tfs):
    obs_p = np.asarray(Lp @ CPM[:, i]).ravel(); obs_d = np.asarray(Ld @ CPM[:, i]).ravel(); obs_t = obs_p + obs_d
    rng = np.random.default_rng(SEED + i); P = np.empty((NE, ND), np.float32); c = CPM[:, i]
    for d in range(ND):
        x = c.copy()
        for ix in STR: x[ix] = c[ix[rng.permutation(len(ix))]]
        P[:, d] = x
    np_ = np.asarray(Lp @ P); nd_ = np.asarray(Ld @ P); nt_ = np_ + nd_; eps = 1e-6
    pp = (1 + (np_ >= obs_p[:, None] - eps).sum(1)) / (ND + 1); pd_ = (1 + (nd_ >= obs_d[:, None] - eps).sum(1)) / (ND + 1); pt = (1 + (nt_ >= obs_t[:, None] - eps).sum(1)) / (ND + 1)
    qt = bh(pt); called[i] = qt <= 0.05
    cols[t] = dict(promoter=np.log1p(obs_p), distal=np.log1p(obs_d), total=np.log1p(obs_t), p_total=pt, q_total=qt, p_promoter=pp, q_promoter=bh(pp), p_distal=pd_, q_distal=bh(pd_), rank_in_tf=rankdata(obs_t) / NG * 100)
    print(t, int(called[i].sum()), flush=True)
deep = np.array([FT.tertile.get(t) == "high" for t in tfs]); shared = called.sum(0); shared_deep = called[deep].sum(0); qc["high_tertile_tfs"] = int(deep.sum())
# partner (section 7)
PL = pd.read_csv(f"{M7}/pair_list.tsv", sep="\t"); partner = {t: [[] for _ in range(NG)] for t in tfs}; ti = {t: i for i, t in enumerate(tfs)}
for a, b in zip(PL.a, PL.b):
    common = np.asarray(La @ ((C[:, ti[a]] > 0) & (C[:, ti[b]] > 0)).astype(np.float64)).ravel() > 0; f = np.flatnonzero(called[ti[a]] & called[ti[b]] & common)
    for g in f: partner[a][g].append(b); partner[b][g].append(a)
    qc[f"pair_{a}_{b}_genes"] = int(len(f))
def tier(t):
    d = FT.d.get(t, np.nan); return "claim" if d >= .04 else ("above_null" if bool(FT.clears.get(t, False)) else "not_supported")
out = f"{O}/edges/edge_table_{link}.tsv.gz"; first = True
for i, t in enumerate(tfs):
    c_ = cols[t]; D = pd.DataFrame(dict(tf=t, gene=genes, chromosome=G.chromosome.values, **{k: c_[k] for k in ("promoter", "distal", "total")}, n_promoter_elements=n_prom, n_distal_elements=n_dist, loop_supported=loop_sup,
        **{k: c_[k] for k in ("p_total", "q_total", "p_promoter", "q_promoter", "p_distal", "q_distal")}, called=called[i], rank_in_tf=c_["rank_in_tf"], in_v8=pd.Series(genes).isin(v8[t]).values, shared_by=shared, shared_by_deep=shared_deep,
        cells=int(FT.cells.get(t, -1)), tertile=FT.tertile.get(t, ""), d=FT.d.get(t, np.nan), tier=tier(t), annotated_sign=SG.predicted_sign.get(t, ""), sign_confidence=SG.confidence.get(t, ""), partner=[";".join(sorted(set(x))) for x in partner[t]]))
    D.to_csv(f"{O}/edges/_part_{link}_{i:02d}.tsv", sep="\t", index=False, float_format="%.5g", header=first); first = False
import gzip, shutil, glob
with gzip.GzipFile(out, "wb", mtime=0) as fo:
    for f in sorted(glob.glob(f"{O}/edges/_part_{link}_*.tsv")):
        with open(f, "rb") as fi: shutil.copyfileobj(fi, fo)
        os.remove(f)
json.dump(qc, open(f"{O}/edges/qc_{link}.json", "w"), indent=1, default=str); print(json.dumps(qc, default=str))
