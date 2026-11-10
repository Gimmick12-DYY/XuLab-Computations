# Threshold-free motif profile, step 3: per-TF motif scores, pair motif tables and profile concordance. No cutoff decides anything;
# every pair gets ranked values plus their place in random-pair distributions (context only).
# Motif set = 2,528 known (db/known.meme) + 393 de novo family representatives (per_tf_raw/denovo_cmp/reps.meme; novel = no Tomtom q < 0.05).
# (a) peak-first, N = 500 and 2,000 (profile_pools.py): FIMO p <= 1e-4 presence in summit windows vs matched OCR background, two-sided
#     Fisher; score = sign(log fold) x -log10 p. FIMO + Fisher stands in for HOMER known so that every one of the 2,921 motifs gets a
#     signed score (HOMER known reports only its own library and only enrichment).
# (b) sequence-first: reversed-design z with raw reads subsampled to 5,903 cells (N20) and 3,000 (N3k), 3 seeds averaged; here pooled
#     over all chromosomes (rev_depth_z.py machinery: CPM minus 79-TF mean, rank-normal, residual within GC x CpG x promoter strata).
# Pair, per method and motif: min(score_A, score_B), geometric mean of within-TF ranks (1 = top); top 10 non-generic motifs by that mean
# (generic = enriched in > 20/79 TFs; enriched = score >= 3), with cross-panel percentile of min score over all panel pairs and
# depth-matched percentile over pairs A' x B' (A', B' among the 10 TFs nearest in cell count to A, B).
# Concordance = Spearman of the two TFs' score vectors over all motifs; percentile among 200 random pairs matched on depth tertiles and
# correlation-pct bin (same scheme as rev_enrich.py).
# Output profile/: scores_{peak500,peak2000,seqN20,seqN3k}.tsv, concordance_matrix_{peak,peak2000,seq,seqN3k}.tsv, pair_top_motifs_long.tsv,
#   novel_pairs_motif_profile.tsv, all_pairs_motif_profile.tsv, controls.tsv
import os, gzip, glob, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import norm, rankdata, fisher_exact, spearmanr
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = f"{O}/profile"; R = f"{O}/reversed"; DP = f"{R}/depth"
rng = np.random.default_rng(20261110); GEN_TF = 20; ENR = 3.0
cells = pd.read_csv(f"{X}/complex_screen/pair_motif_lookup/tf_cells.tsv", sep="\t").set_index("tf").n_cells; tfs = list(cells.index)
alt = {}
for f in [f"{O}/db/known.meme", f"{O}/per_tf_raw/denovo_cmp/reps.meme"]:
    for l in open(f):
        if l.startswith("MOTIF "): x = l.split(); alt[x[1]] = x[2] if len(x) > 2 else x[1]
motifs = list(alt); mi = {m: k for k, m in enumerate(motifs)}; M = len(motifs)
fam = pd.read_csv(f"{O}/per_tf_raw/denovo_families.tsv", sep="\t").set_index("family")
for k in fam.index: alt[f"fam{k}"] = f"denovo:{fam.cons[k]}" + ("(novel)" if fam.novel[k] else f"(~{fam.guess[k]})")
isfam = np.array([m.startswith("fam") for m in motifs])
def label(m): return f"{m}|{alt[m]}" if not m.startswith("fam") else alt[m]
# ---------- (a) peak-first
win = {}
for f in sorted(glob.glob(f"{G}/fimo/win_c*.tsv.gz")):
    for l in gzip.open(f, "rt"): m, s = l.rstrip("\n").split("\t"); win.setdefault(s, set()).add(mi[m])
binh = {}
for f in sorted(glob.glob(f"{R}/fimo/c*.tsv.gz")) + sorted(glob.glob(f"{G}/fimo/bins_c*.tsv.gz")):
    for l in gzip.open(f, "rt"):
        m, s = l.rstrip("\n").split("\t")
        if m in mi: binh.setdefault(s, set()).add(mi[m])
print(len(win), "windows with hits;", len(binh), "bins with hits", flush=True)
def counts(ids, H):
    c = np.zeros(M, int)
    for s in ids:
        for j in H.get(s, ()): c[j] += 1
    return c
SC = {}
for N in [500, 2000]:
    S = np.full((M, len(tfs)), np.nan); FO = S.copy()
    for j, t in enumerate(tfs):
        p = f"{G}/pools/{t}.tsv"
        if not os.path.exists(p): continue
        w = pd.read_csv(p, sep="\t"); fg = list(w.n[w["rank"] < N]); bg = [x for x in open(f"{G}/pools/{t}_bg{N}.txt").read().split()]
        if len(fg) < 100 or len(bg) < 100: continue
        a = counts(fg, win); b = counts(bg, binh); nf, nb = len(fg), len(bg)
        fo = ((a + .5) / nf) / ((b + .5) / nb); pv = np.array([fisher_exact([[a[k], nf - a[k]], [b[k], nb - b[k]]])[1] if a[k] + b[k] else 1.0 for k in range(M)])
        S[:, j] = np.sign(np.log(fo)) * -np.log10(np.maximum(pv, 1e-300)); FO[:, j] = fo
    SC[f"peak{N}"] = pd.DataFrame(S, index=motifs, columns=tfs); SC[f"peak{N}_fold"] = pd.DataFrame(FO, index=motifs, columns=tfs); print("peak", N, "done", flush=True)
# ---------- (b) sequence-first
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t"); num = Q.chrom.str.replace("chr", "")
ok = Q.n_ok.values & ~Q.blacklist.values & (Q.exonic.values <= 0.5) & num.isin([str(i) for i in range(1, 23)] + ["X"]).values; ei = np.flatnonzero(ok)
prom = np.zeros(len(Q), bool); prom[[int(x) for x in open(f"{O}/ref/promoter_bins.txt").read().split()]] = True
cq = np.quantile(Q.cpg_oe.values[ei], [0.2, 0.4, 0.6, 0.8]); st = ((np.floor(Q.gc.values / 0.025).astype(int) * 10 + np.searchsorted(cq, Q.cpg_oe.values)) * 2 + prom)[ei]
pos = {f"b{b}": i for i, b in enumerate(Q.bin.values[ei])}; rr, cc = [], []
for s, js in binh.items():
    if s in pos: rr += [pos[s]] * len(js); cc += list(js)
H = sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(len(ei), M)); H.data[:] = 1; nH = np.asarray(H.sum(0)).ravel()
def zall(D):
    Y = np.apply_along_axis(lambda v: norm.ppf((rankdata(v) - 0.5) / len(v)), 1, D[:, ei]).T
    for s in np.unique(st): q = st == s; Y[q] -= Y[q].mean(0)
    z = (H.T @ Y) / np.maximum(nH, 1)[:, None] * np.sqrt(np.maximum(nH, 1))[:, None] / Y.std(0)[None, :]; z[nH < 50] = np.nan; return z
for lv in ["N20", "N3k"]:
    acc = []
    for sd in range(3):
        C = np.stack([np.load(f"{DP}/counts/{t}.npz")[lv][sd] for t in tfs]); C = C / C.sum(1, keepdims=True) * 1e6; acc.append(zall(C - C.mean(0)))
    SC[f"seq{lv}"] = pd.DataFrame(np.mean(acc, 0), index=motifs, columns=tfs); print("seq", lv, "done", flush=True)
for k, v in SC.items(): v.round(3).to_csv(f"{G}/scores_{k}.tsv", sep="\t")
METH = ["peak500", "peak2000", "seqN20", "seqN3k"]; CM = {"peak500": "peak", "peak2000": "peak2000", "seqN20": "seq", "seqN3k": "seqN3k"}
RK, GENF, CONC = {}, {}, {}
for m in METH:
    S = SC[m]; RK[m] = S.rank(ascending=False, method="average")  # 1 = top within TF
    GENF[m] = (S >= ENR).sum(1) > GEN_TF
    C = S.rank().corr(method="pearson")  # Spearman over motifs (NaN-aware pairwise)
    CONC[m] = C; C.round(4).to_csv(f"{G}/concordance_matrix_{CM[m]}.tsv", sep="\t")
# ---------- pairs
R1 = pd.read_csv(f"{X}/regulatome/r1_tftf/v2/r1v2_pairs_provisional.tsv", sep="\t"); R1 = R1[R1.a.isin(tfs) & R1.b.isin(tfs)].copy()
R1["pct"] = R1[["pearson_domain_pct", "spqn_noquies_pct"]].max(1); R1["pair"] = R1.a + "-" + R1.b; R1["novel"] = R1.string_physical.isna() & ~R1.capselex.astype(str).str.startswith("called")
tert = np.quantile(cells.values, [1 / 3, 2 / 3]); cb = {t: int(np.searchsorted(tert, cells[t])) for t in tfs}
R1["dbin"] = [tuple(sorted((cb[a], cb[b]))) for a, b in zip(R1.a, R1.b)]; R1["pbin"] = pd.cut(R1.pct, [-1, .6, .8, .9, 1.01], labels=False)
af3 = {}
for f in glob.glob(f"{X}/complex_screen/af3_corr060_20261104/scores_s1/*.tsv"):
    x = open(f).read().split("\t"); a_, b_ = x[0].split("__"); af3[tuple(sorted((a_, b_)))] = (x[4], float(x[1]), float(x[2]))
for f in [f"{X}/complex_screen/af3_novel_20261103/af3_scores_weak.tsv", f"{X}/complex_screen/af3_novel_20261103/af3_scores.tsv"]:
    for l in open(f):
        x = l.rstrip("\n").split("\t")
        if x[0] != "pair": a_, b_ = [y.replace("CBX3", "HP1").replace("CBX1", "HP1").replace("ZUP1", "ZUFSP") for y in x[0].split("__")]; af3[tuple(sorted((a_, b_)))] = (x[9], float(x[1]), float(x[7]))
R1["af3_tier"], R1["af3_iptm"], R1["af3_ipsae"] = zip(*[af3.get(tuple(sorted((a, b))), ("not_run", np.nan, np.nan)) for a, b in zip(R1.a, R1.b)])
tri = pd.read_csv(f"{O}/triple_support/pair_summary.tsv", sep="\t"); tri = {tuple(sorted(p.split("-"))): t for p, t in zip(tri.pair, tri.tier)}
R1["triple_tier"] = [tri.get(tuple(sorted((a, b))), "") for a, b in zip(R1.a, R1.b)]
for m in METH:
    C = CONC[m]; R1[f"conc_{m}"] = [C.loc[a, b] for a, b in zip(R1.a, R1.b)]
    def pctl(r):
        pool = R1[(R1.dbin == r.dbin) & (R1.pbin == r.pbin) & (R1.pair != r.pair)]
        if len(pool) < 20: pool = R1[(R1.dbin == r.dbin) & (R1.pair != r.pair)]
        v = rng.choice(pool[f"conc_{m}"].dropna().values, 200); return np.nan if np.isnan(r[f"conc_{m}"]) else round(float((v < r[f"conc_{m}"]).mean() + .5 * (v == r[f"conc_{m}"]).mean()), 3)
    R1[f"conc_{m}_matched_pct"] = [pctl(r) for _, r in R1.iterrows()]
    R1[f"conc_{m}_panel_pct"] = R1[f"conc_{m}"].rank(pct=True).round(3)
# per-motif pair min scores for all panel pairs (for percentiles)
A_ = [tfs.index(a) for a in R1.a]; B_ = [tfs.index(b) for b in R1.b]; near = {t: list((cells.drop(t) - cells[t]).abs().sort_values().index[:10]) for t in tfs}
key = {tuple(sorted((a, b))): i for i, (a, b) in enumerate(zip(R1.a, R1.b))}
long = []; TOPC = {m: [] for m in METH}; FAMC = {m: [] for m in METH}
for m in METH:
    S = SC[m].values; Rk = RK[m].values; gen = GENF[m].values; MIN = np.minimum(S[:, A_], S[:, B_])  # motifs x pairs
    nen = (SC[m] >= ENR).sum(1).values
    for i, (a, b) in enumerate(zip(R1.a, R1.b)):
        ja, jb = tfs.index(a), tfs.index(b); gm = np.sqrt(Rk[:, ja] * Rk[:, jb]); gm[np.isnan(gm)] = np.inf
        def pick(mask, n):
            o = [k for k in np.lexsort((-np.nan_to_num(MIN[:, i], nan=-1e9), gm)) if mask[k] and np.isfinite(gm[k])][:n]; out = []
            for k in o:
                row = MIN[k]; xp = float(np.nanmean(row < row[i]) + .5 * np.nanmean(row == row[i]))
                dm = [key[tuple(sorted((u, v)))] for u in near[a] for v in near[b] if u != v and tuple(sorted((u, v))) in key and tuple(sorted((u, v))) != tuple(sorted((a, b)))]
                dp = float(np.mean(row[dm] < row[i]) + .5 * np.mean(row[dm] == row[i])) if dm else np.nan
                d = dict(pair=R1.pair.iloc[i], method=m, motif=label(motifs[k]), denovo=bool(isfam[k]), generic=bool(gen[k]), n_tf_enriched=int(nen[k]),
                         score_A=round(S[k, ja], 2), score_B=round(S[k, jb], 2), rank_A=int(Rk[k, ja]), rank_B=int(Rk[k, jb]), geo_rank=round(gm[k], 1), min_score=round(MIN[k, i], 2),
                         panel_pct=round(xp, 3), depth_matched_pct=round(dp, 3), n_depth_matched=len(dm))
                if m.startswith("peak"): F = SC[f"{m}_fold"].values; d["fold_A"] = round(F[k, ja], 2); d["fold_B"] = round(F[k, jb], 2)
                out.append(d)
            return out
        top = pick(~gen & ~isfam, 10); fm = pick(~gen & isfam, 5); long += top + fm
        fmt = lambda L: "; ".join(f"{d['motif']} [{d['score_A']}/{d['score_B']}" + (f", fold {d['fold_A']}/{d['fold_B']}" if 'fold_A' in d else "") + f", panel {d['panel_pct']}, depth-matched {d['depth_matched_pct']}]" for d in L)
        TOPC[m].append(fmt(top)); FAMC[m].append(fmt(fm))
    print("pairs", m, "done", flush=True)
for m in METH: R1[f"top10_{m}"] = TOPC[m]; R1[f"top5_denovo_{m}"] = FAMC[m]
pd.DataFrame(long).to_csv(f"{G}/pair_top_motifs_long.tsv", sep="\t", index=False)
cols = ["pair", "novel", "triple_tier", "pct", "pearson_domain_pct", "spqn_noquies_pct", "af3_tier", "af3_iptm", "af3_ipsae"]
for m in METH: cols += [f"conc_{m}", f"conc_{m}_panel_pct", f"conc_{m}_matched_pct"]
cols += [f"top10_{m}" for m in METH] + [f"top5_denovo_{m}" for m in METH]
rk = {"strong": 0, "weak": 1}; R1["_a"] = R1.af3_tier.map(rk).fillna(2); R1 = R1.sort_values(["_a", "pct"], ascending=[True, False])
R1[cols].to_csv(f"{G}/all_pairs_motif_profile.tsv", sep="\t", index=False); R1[R1.novel][cols].to_csv(f"{G}/novel_pairs_motif_profile.tsv", sep="\t", index=False)
CON = ["CTCF-STAG2", "MAZ-ZNF143", "NFYA-ZNF644", "EZH2-MTF2", "RBBP7-SRCAP"]; Cn = R1[R1.pair.isin(CON) | (R1.b + "-" + R1.a).isin(CON)][cols]; Cn.to_csv(f"{G}/controls.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
cc = ["pair", "af3_tier"] + [c for m in METH for c in (f"conc_{m}", f"conc_{m}_matched_pct")]
print(Cn[cc].to_string(index=False)); nv = R1[R1.novel]
for tier in ["strong", "weak"]:
    s = nv[nv.af3_tier == tier]; print(tier, len(s), {m: (round(s[f"conc_{m}_matched_pct"].median(), 3), round((s[f"conc_{m}_matched_pct"] >= .95).mean(), 3)) for m in METH})
print("all novel", len(nv), {m: (round(nv[f"conc_{m}_matched_pct"].median(), 3), round((nv[f"conc_{m}_matched_pct"] >= .95).mean(), 3)) for m in METH})
own = {"CTCF": "CTCF.H12CORE.0.P.B", "NFYA": "NFYA.H12CORE.0.P.B", "NRF1": "NRF1.H12CORE.0.PS.A", "MAZ": "MAZ.H12CORE.0.P.B", "ZNF143": "MA0088.2", "HES1": "HES1.H12CORE.0.S.C", "E2F3": "MA0469.4", "MLX": "MA0663.1", "ZBTB7A": "ZBTB7A.H12CORE.0.P.B" if "ZBTB7A.H12CORE.0.P.B" in mi else "MA0750.3"}
print("own-motif within-TF rank (of", M, "):"); print(pd.DataFrame({t: {m: (int(RK[m].loc[o, t]) if o in mi and not np.isnan(RK[m].loc[o, t]) else -1) for m in METH} for t, o in own.items()}).T.to_string())
