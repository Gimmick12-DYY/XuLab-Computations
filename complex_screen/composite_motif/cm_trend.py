# EXPLORATORY trend test, no fixed p-value cutoffs. Scores every union-ATAC element by best log-odds (both strands): a motif's own best site, or for a pair the best sum of two non-overlapping half-sites
# within 50 bp. Elements are stratified by 20 expected-read bins x 5 width bins (longer elements have higher best scores by chance, so width is held fixed; this is added to the requested design).
# Within each stratum the score becomes a percentile and the outcome is divided by the stratum mean. Deciles are taken on that percentile. Trend: Spearman and Jonckheere-Terpstra (tie-corrected), one-sided;
# effect = top over bottom decile mean ratio with a bootstrap CI over elements. Outcomes: co-binding residual (product, min) and each partner's (y + .5) / (mu + .5). Column-shuffled motifs are controls.
import os, re
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.ndimage import maximum_filter1d
from scipy.stats import norm, rankdata, spearmanr
from cm_common import *
OUT = f"{O}/trend_exploratory"; os.makedirs(OUT, exist_ok=True); E = load_elements(); n = len(E)
seqs = {}; name = None
for l in open(f"{U}/union.fa"):
    l = l.strip()
    if l.startswith(">"): name = l[1:].split("::")[0]; seqs[name] = []
    else: seqs[name].append(l.upper())
tr = np.full(256, 4, dtype=np.int8)
for i, c in enumerate(b"ACGT"): tr[c] = i
S = [tr[np.frombuffer("".join(seqs[e]).encode(), dtype=np.uint8)] for e in E.eid]
def meme(path, mid):
    b = open(path).read().split(f"MOTIF {mid}")[1].split("MOTIF")[0].strip().split("\n"); i = [k for k, l in enumerate(b) if l.startswith("letter-probability")][0]
    return np.array([[float(x) for x in l.split()] for l in b[i + 1:] if l.strip() and not l.startswith("URL")])
def pssm(P): return np.log2((P + 1e-3) / 0.25)
def pos_scores(s, M):
    w = len(M)
    if len(s) < w: return None
    Mx = np.vstack([M.T, np.full(w, -10.0)]); Rc = M[::-1, ::-1]; Rx = np.vstack([Rc.T, np.full(w, -10.0)]); ix = swv(s, w); k = np.arange(w)
    return np.maximum(Mx[ix, k].sum(1), Rx[ix, k].sum(1))
def single(M):
    out = np.full(n, np.nan)
    for i, s in enumerate(S):
        p = pos_scores(s, M)
        if p is not None: out[i] = p.max()
    return out
def composite(A, B, G=50):
    wa, wb = len(A), len(B); out = np.full(n, np.nan); W = G + 1
    for i, s in enumerate(S):
        pa, pb = pos_scores(s, A), pos_scores(s, B)
        if pa is None or pb is None: continue
        # max of pb over j in [x + wa, x + wa + G] (B after A) and [x - wb - G, x - wb] (B before A)
        mf = maximum_filter1d(pb, W, origin=-(W // 2), mode="constant", cval=-1e9); best = -1e9
        if len(pa) > 0:
            aft = np.arange(len(pa)) + wa; ok = aft < len(pb)
            if ok.any(): best = max(best, (pa[ok] + mf[aft[ok]]).max())
            st = np.arange(len(pa)) - wb - G; bk = np.arange(len(pa)) - wb; ok = bk >= 0
            if ok.any():
                mg = maximum_filter1d(pb, W, origin=W // 2, mode="constant", cval=-1e9); best = max(best, (pa[ok] + mg[bk[ok]]).max())
        if best > -1e8: out[i] = best
    return out
J = f"{X}/motif_analysis/cache/motifdb/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"; P = f"{O}/pairs/all.meme"; C = f"{O}/calibration/ctcf_real_shuf.meme"
mot = lambda m: pssm(meme(P, m)); ctcf = pssm(meme(J, "MA0139.2"))
SC = {"CTCF": single(ctcf)}
for s in (1, 2, 3): SC[f"CTCF_shuf{s}"] = single(pssm(meme(C, f"shuffle{s}")))
SC["MLX+SRE"] = composite(mot("MA0663.1"), mot("MA0596.1"))
for s in (1, 2, 3): SC[f"MLX+SRE_shuf{s}"] = composite(mot(f"MA0663.1.shuf{s}"), mot(f"MA0596.1.shuf{s}"))
SC["MLX_FOXA1"] = single(mot("MLX_FOXA1")); SC["MLX_FOXD2"] = single(mot("MLX_FOXD2"))
pd.DataFrame(SC, index=E.eid).to_csv(f"{OUT}/scores.tsv.gz", sep="\t", float_format="%.3f")
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]; Yz = {t: np.load(f"{U}/counts/{t}.npz") for t in tfs}; tot = sum(Yz[t]["y"] for t in tfs); NN = sum(int(Yz[t]["total"]) for t in tfs)
def resid(t):
    y = Yz[t]["y"].astype(float); R = int(Yz[t]["total"]); return y, residual(y, R, tot - y, NN - R)
RES = {t: resid(t) for t in ["CTCF", "MLX", "SREBF2", "FOXK1", "RBBP7", "SRCAP"]}; W_ = (E.end - E.start).values
def strata(*mus):
    rk = np.mean([pd.Series(m).rank(pct=True).values for m in mus], axis=0); return np.minimum((rk * 20).astype(int), 19) * 5 + np.minimum((pd.Series(W_).rank(pct=True).values * 5).astype(int), 4)
def norm_within(v, st): d = pd.Series(v).groupby(st).transform("mean").values; return v / d
def pct_within(v, st): return pd.Series(v).groupby(st).rank(pct=True).values
def jt(x, g):
    N = len(x); ni = np.bincount(g, minlength=10).astype(float); _, t = np.unique(x, return_counts=True); t = t.astype(float); r = rankdata(x); J_ = 0.0
    for a in range(10):
        for b in range(a + 1, 10):
            rr = rankdata(np.concatenate([x[g == a], x[g == b]])); na, nb = (g == a).sum(), (g == b).sum(); J_ += rr[na:].sum() - nb * (nb + 1) / 2
    m = (N ** 2 - (ni ** 2).sum()) / 4; f = lambda v: (v * (v - 1) * (2 * v + 5)).sum(); f2 = lambda v: (v * (v - 1) * (v - 2)).sum(); f3 = lambda v: (v * (v - 1)).sum()
    V = (N * (N - 1) * (2 * N + 5) - f(ni) - f(t)) / 72 + f2(ni) * f2(t) / (36 * N * (N - 1) * (N - 2)) + f3(ni) * f3(t) / (8 * N * (N - 1)); z = (J_ - m) / np.sqrt(V); return z, norm.sf(z)
def trend(score, stat, st, label):
    ok = ~np.isnan(score); sc = pct_within(score[ok], st[ok]); v = norm_within(stat[ok], st[ok]); g = np.minimum((sc * 10).astype(int), 9); rho, _ = spearmanr(sc, v); rp = spearmanr(sc, v, alternative="greater").pvalue; z, jp = jt(v, g)
    top, bot = g == 9, g == 0; eff = v[top].mean() / v[bot].mean(); r = np.random.default_rng(11); N = len(v); bs = []
    for _ in range(1000):
        ix = r.integers(0, N, N); bs.append(v[ix][g[ix] == 9].mean() / v[ix][g[ix] == 0].mean())
    return dict(label, n=int(ok.sum()), spearman=round(rho, 4), spearman_p=rp, jt_z=round(z, 2), jt_p=jp, top_vs_bottom=round(eff, 3), lo=round(np.percentile(bs, 2.5), 3), hi=round(np.percentile(bs, 97.5), 3), decile_means=",".join(f"{v[g == k].mean():.3f}" for k in range(10)))
rows = []
def own(score_name, t, role):
    y, mu = RES[t]; rows.append(trend(SC[score_name], (y + .5) / (mu + .5), strata(mu), dict(test=role, score=score_name, outcome=f"own_{t}")))
def cob(score_name, a, b, role):
    ya, ma = RES[a]; yb, mb = RES[b]; ra, rb = (ya + .5) / (ma + .5), (yb + .5) / (mb + .5); st = strata(ma, mb)
    rows.append(trend(SC[score_name], ra * rb, st, dict(test=role, score=score_name, outcome="cobind_product"))); rows.append(trend(SC[score_name], np.minimum(ra, rb), st, dict(test=role, score=score_name, outcome="cobind_min")))
own("CTCF", "CTCF", "calibration")
for s in (1, 2, 3): own(f"CTCF_shuf{s}", "CTCF", "calibration shuffle")
cob("MLX+SRE", "MLX", "SREBF2", "MLX-SREBF2"); own("MLX+SRE", "MLX", "MLX-SREBF2"); own("MLX+SRE", "SREBF2", "MLX-SREBF2")
for s in (1, 2, 3): cob(f"MLX+SRE_shuf{s}", "MLX", "SREBF2", "MLX-SREBF2 shuffle")
for c in ("MLX_FOXA1", "MLX_FOXD2"): cob(c, "FOXK1", "MLX", "FOXK1-MLX (paralog composite)"); own(c, "FOXK1", "FOXK1-MLX (paralog composite)"); own(c, "MLX", "FOXK1-MLX (paralog composite)")
cob("MLX+SRE", "RBBP7", "SRCAP", "negative control"); own("MLX+SRE", "RBBP7", "negative control"); own("MLX+SRE", "SRCAP", "negative control")
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/trend_results.tsv", sep="\t", index=False); pd.set_option("display.width", 200); pd.set_option("display.max_columns", 15); pd.set_option("display.max_colwidth", 32); print(T.drop(columns="decile_means").to_string(index=False)); print(T[T.score.isin(["CTCF", "MLX+SRE"])][["score", "outcome", "decile_means"]].to_string(index=False))
