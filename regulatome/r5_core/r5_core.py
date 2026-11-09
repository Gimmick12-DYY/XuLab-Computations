# R5 Regulome core (spec: SPEC.md). S(t,g) = sum over m2_abc power-law links (thr 0.02, promoters included) of CPM_te x ABC(e,g); raw read 5' ends
# per element per TF (Gate 1 cellmats, counted with the m1_binding/v2 rule), CPM = reads at element / all reads of the TF's cells x 1e6.
# No gate, residual, EB, weights, propagation or imputation; all 79 TFs identical. Companions: S_atac (union ATAC CPM x ABC) and S_pool
# (leave-TF-out pooled panel CPM x ABC). Writes lists/, tf_summary.tsv, checks/, edges/, then manifest_v7.sha256 (refuses if it exists).
# usage: r5_core.py [--test TF1,TF2]
import glob, hashlib, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr
X = "/work/users/d/y/dyy12/XuLab"; G = f"{X}/regulatome/gate1"; R = f"{X}/regulatome/r5_core"
TEST = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[1] == "--test" else None; O = f"{R}/tmp/test" if TEST else R
assert not os.path.exists(f"{R}/manifest_v7.sha256"), "manifest v7 exists; R5 core outputs are frozen"
for d in ("lists", "checks", "edges"): os.makedirs(f"{O}/{d}", exist_ok=True)
t0 = time.time(); log = lambda *m: print(*m, f"{time.time() - t0:.0f}s", flush=True)
E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"])
act = np.flatnonzero(E.cls.isin(["promoter", "distal_active", "bivalent"]).values); Ea = E.iloc[act].reset_index(drop=True); nE = len(Ea)
atac = np.load(f"{X}/regulatome/m1_binding/v2/counts/atac_counts.npz")["atac"][act].astype(float); atac_cpm = atac / atac.sum() * 1e6
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
Y, NR = {}, {}
for t in tfs: Y[t] = np.asarray(sp.load_npz(f"{G}/cellmats/{t}.raw.npz").sum(0)).ravel(); NR[t] = np.load(f"{G}/cellmats/{t}.cells.npz")["reads"].astype(float).sum()
YT = sum(Y.values()); NT = sum(NR.values())
A = pd.read_csv(f"{X}/regulatome/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "class", "TargetGene", "TargetGeneTSS", "ABC.Score"])
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}; A["e"] = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord)
assert A.e.notna().all(), "every m2_abc link must map to an active element"; A["e"] = A.e.astype(int)
genes = np.array(sorted(A.TargetGene.unique())); ng = len(genes); gix = {g: i for i, g in enumerate(genes)}; A["g"] = A.TargetGene.map(gix)
le, lg, lw = A.e.values, A.g.values, A["ABC.Score"].values; lprom = (A["class"] == "promoter").values
mid = ((Ea.start.values + Ea.end.values) // 2).astype(np.int64); ldist = np.abs(mid[le] - A.TargetGeneTSS.values.astype(np.int64))
maxabc = np.zeros(ng); np.maximum.at(maxabc, lg, lw)
# loop support: distal (non-promoter-class) linked element inside a Peakachu 5 kb 0.7 anchor whose partner overlaps the gene's TSS +/- 2 kb
src = f"{X}/regulatome/r5_two_layer/edges/loop_anchor_promoter.tsv.gz"; shutil.copy(src, f"{O}/edges/loop_anchor_promoter.tsv.gz"); LP = pd.read_csv(src, sep="\t")
eb = {c: np.flatnonzero(Ea.chrom.values == c) for c in Ea.chrom.unique()}; loopset = set()
for c, d in LP.groupby("chrom"):
    if c not in eb: continue
    ei = eb[c][np.argsort(Ea.start.values[eb[c]])]; es, ee = Ea.start.values[ei], Ea.end.values[ei]
    for r in d.itertuples(index=False):
        lo, hi = np.searchsorted(ee, r.partner_start, side="right"), np.searchsorted(es, r.partner_end, side="left")
        if r.gene in gix:
            for e in ei[lo:hi]: loopset.add((int(e), gix[r.gene]))
lloop = np.array([(int(e), int(g)) in loopset for e, g in zip(le, lg)]) & ~lprom
log(f"elements {nE}; links {len(A)} (promoter class {int(lprom.sum())}); genes {ng}; distal links in a promoter-paired 5 kb anchor {int(lloop.sum())}")
def score(cpm):  # S, promoter part, distal part, N_e, top-contributing element distance, loop elements with >= 1 read
    c = cpm[le] * lw; S = np.bincount(lg, c, ng); Sp = np.bincount(lg, np.where(lprom, c, 0), ng); return S, Sp, S - Sp, c
def extras(y, c):
    Ne = np.bincount(lg, (y[le] > 0).astype(float), ng).astype(int); nl = np.bincount(lg, (lloop & (y[le] > 0)).astype(float), ng).astype(int)
    o = np.lexsort((-c, lg)); first = np.r_[True, lg[o][1:] != lg[o][:-1]]; td = np.full(ng, np.nan); td[lg[o][first]] = np.where(c[o][first] > 0, ldist[o][first], np.nan); return Ne, nl, td
def top(S, ok, n=1000):
    s = np.where(ok & (S > 0), S, -np.inf); o = np.argsort(-s, kind="stable"); return o[np.isfinite(s[o])][:n]
Sa, Sap, Sad, _ = score(atac_cpm); atac_top = set(top(Sa, np.ones(ng, bool)))
TRUTH = {"CTCF": "CTCF_majority2of3.bed", "MAZ": "MAZ_HEK293_peaks.bed", "NFYA": "NFYA_K562_peaks.bed", "ZBTB7A": "ZBTB7A_HEK293_peaks.bed", "ZIC2": "ZIC2_HEK293_peaks.bed", "ZNF282": "ZNF282_HEK293_peaks.bed", "ZNF777": "ZNF777_HEK293_peaks.bed"}
hp = subprocess.run(f"unzip -p {X}/data/hpa/rna_celline.tsv.zip | awk -F'\\t' '$3==\"HEK293\"'", shell=True, capture_output=True, text=True).stdout
H_ = pd.DataFrame([l.split("\t") for l in hp.strip().split("\n")], columns=["ens", "gene", "line", "TPM", "pTPM", "nTPM"]); nt = H_.drop_duplicates("gene").set_index("gene").nTPM.astype(float).reindex(genes).values; hk = ~np.isnan(nt)
v3 = lambda t: set(pd.read_csv(f"{X}/regulatome/r2_targets/top1000/{t}.tsv", sep="\t").gene) if os.path.exists(f"{X}/regulatome/r2_targets/top1000/{t}.tsv") else set()
def po(t):
    f = f"{X}/regulatome/m4_validation/arms/promoter_only/{t}.tsv"; return set(pd.read_csv(f, sep="\t").sort_values("S", ascending=False).head(1000).gene) if os.path.exists(f) else set()
jac = lambda a, b: len(a & b) / len(a | b) if a | b else np.nan
rows, CH, HP, JP = [], [], [], []
for t in (TEST or tfs):
    y = Y[t]; cpm = y / NR[t] * 1e6; pool = (YT - y) / (NT - NR[t]) * 1e6
    S, Sp, Sd, c = score(cpm); Ne, nl, td = extras(y, c); Spool = score(pool)[0]
    ok = Ne >= 2; tp = top(S, ok); tp_any = top(S, np.ones(ng, bool)); tps = set(tp)
    T = pd.DataFrame(dict(gene=genes, S=S, S_atac=Sa, S_pool=Spool, N_e=Ne, promoter_share=np.divide(Sp, S, out=np.full(ng, np.nan), where=S > 0), distal_share=np.divide(Sd, S, out=np.full(ng, np.nan), where=S > 0),
                          n_loop_elements=nl, max_ABC=maxabc, top_element_distance=td))
    T["top_1000"] = np.isin(np.arange(ng), tp); T = T[T.S > 0].sort_values(["S", "N_e"], ascending=[False, False]); T["rank"] = np.arange(1, len(T) + 1); T.to_csv(f"{O}/lists/{t}.tsv.gz", sep="\t", index=False, float_format="%.5g")
    Rm = sp.load_npz(f"{G}/cellmats/{t}.raw.npz").tocsr(); rd = np.load(f"{G}/cellmats/{t}.cells.npz")["reads"].astype(float); hh = np.random.default_rng(20261029 + tfs.index(t)).permutation(Rm.shape[0]) < Rm.shape[0] // 2; half = []
    for m in (hh, ~hh):
        yh = np.asarray(Rm[m].sum(0)).ravel(); Sh, _, _, ch = score(yh / rd[m].sum() * 1e6); Neh = np.bincount(lg, (yh[le] > 0).astype(float), ng); half.append(set(top(Sh, Neh >= 2)))
    sel = S > 0; G1 = set(genes[tp])
    tpp, tpd = set(genes[top(Sp, ok)]), set(genes[top(Sd, ok)])
    rows.append(dict(tf=t, reads=NR[t], genes_S_pos=int(sel.sum()), n_top1000=len(tp), top1000_excluded_by_Ne=len(set(tp_any) - tps), median_promoter_share_top1000=float(np.nanmedian(T.promoter_share[T.top_1000])) if len(tp) else np.nan,
                     frac_top1000_loop_supported=float((nl[tp] > 0).mean()) if len(tp) else np.nan, spearman_S_Satac=spearmanr(S[sel], Sa[sel])[0], spearman_S_Spool=spearmanr(S[sel], Spool[sel])[0],
                     jaccard_split_half_top1000=jac(half[0], half[1]), ovl_v3_direct=len(G1 & v3(t)), ovl_promoter_only_B0=len(G1 & po(t)), ovl_Satac_top1000=len(G1 & set(genes[list(atac_top)]))))
    JP.append(dict(tf=t, jaccard_promoter_vs_distal=jac(tpp, tpd), jaccard_promoter_vs_full=jac(tpp, G1), jaccard_distal_vs_full=jac(tpd, G1)))
    HP.append(dict(tf=t, n_genes=int((sel & hk).sum()), spearman_S_nTPM=spearmanr(S[sel & hk], nt[sel & hk])[0], spearman_Satac_nTPM=spearmanr(Sa[sel & hk], nt[sel & hk])[0]))
    if t in TRUTH:
        pk = pd.read_csv(f"{X}/data/{TRUTH[t]}", sep="\t", header=None, usecols=[0, 1, 2], comment="#"); pos = np.zeros(nE, bool)
        for cc, d in pk.groupby(0):
            if cc not in eb: continue
            ei = eb[cc]; ss, se = np.sort(d[1].values), np.sort(d[2].values); pos[ei] = (np.searchsorted(ss, Ea.end.values[ei], side="left") - np.searchsorted(se, Ea.start.values[ei], side="right")) > 0
        for nm, ix, contrib in (("S top 1000", tp, y[le] > 0), ("S_atac top 1000", np.array(sorted(atac_top)), atac[le] > 0)):
            for part, msk in (("any", np.ones(len(le), bool)), ("promoter", lprom), ("distal", ~lprom)):
                hit = np.zeros(ng, bool); np.logical_or.at(hit, lg[contrib & msk], pos[le[contrib & msk]]); CH.append(dict(tf=t, list=nm, elements=part, n=len(ix), frac_genes_with_chip_peak=float(hit[ix].mean()) if len(ix) else np.nan))
    log(t, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()})
pd.DataFrame(rows).to_csv(f"{O}/tf_summary.tsv", sep="\t", index=False, float_format="%.5g"); pd.DataFrame(CH).to_csv(f"{O}/checks/chip_contributing_elements.tsv", sep="\t", index=False, float_format="%.4g")
pd.DataFrame(HP).to_csv(f"{O}/checks/hpa_spearman.tsv", sep="\t", index=False, float_format="%.4g"); pd.DataFrame(JP).to_csv(f"{O}/checks/promoter_distal_jaccard.tsv", sep="\t", index=False, float_format="%.4g"); log("written")
if not TEST:
    shutil.rmtree(f"{R}/tmp", ignore_errors=True)
    fs = sorted(p for p in glob.glob(f"{R}/**/*", recursive=True) if os.path.isfile(p) and not p.endswith("manifest_v7.sha256") and "/logs/" not in p)
    open(f"{R}/manifest_v7.sha256", "w").write("".join(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.relpath(p, X)}\n" for p in fs)); log("manifest v7", hashlib.sha256(open(f"{R}/manifest_v7.sha256", "rb").read()).hexdigest(), len(fs))
