# Composite-motif pair tests (design fixed before results were seen; p <= 1e-5 + expected-read bins is post-hoc after the CTCF calibration, justified by its dose response).
# Elements: composite = hits of both half-sites, non-overlapping, gap <= 50 bp; single = one half-site only, the other absent anywhere in the element; none = neither; far = both, gap > 50 bp (left out).
# Primary contrast: composite vs single (pooled); secondary: composite vs none; both within 20 bins of expected reads. Ratio = binned mean over binned mean, weights min(n1, n0), bootstrap over elements.
# Outcomes: co-binding residual as product and as min of the partners' (y + .5) / (mu + .5); each partner's own obs/exp. Side column: pre-specified p <= 1e-4 raw means.
# Pass (declared here): product AND min both have CI above 1 at p <= 1e-5 on the primary contrast, and both ratios > 1 at p <= 1e-6. Dose: tertiles of the summed half-site log-odds, one-sided bootstrap T3 > T1.
# FOXK1-MLX: the CAP-SELEX spacing prior (E-box, 8 N, forkhead core on the forkhead strand, from MLX_FOXA1 / MLX_FOXD2) is the single pre-registered test, run before its discovery scan.
import numpy as np, pandas as pd
from scipy.stats import poisson
from cm_common import *
D = f"{O}/pairs"; E = load_elements(); n = len(E); idx = {e: i for i, e in enumerate(E.eid)}; W = (E.end - E.start).values
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]; Yz = {t: np.load(f"{U}/counts/{t}.npz") for t in tfs}; tot = sum(Yz[t]["y"] for t in tfs); NN = sum(int(Yz[t]["total"]) for t in tfs)
def resid(t):
    y = Yz[t]["y"].astype(float); R = int(Yz[t]["total"]); return y, residual(y, R, tot - y, NN - R), R
F = pd.read_csv(f"{D}/fimo_all.tsv", sep="\t", comment="#"); F = F[F["p-value"].notna()]; F["i"] = F.sequence_name.map(idx); FD = pd.read_csv(f"{D}/fimo_dinuc.tsv", sep="\t", comment="#"); FD = FD[FD["p-value"].notna()]; FD["i"] = FD.sequence_name.str.replace("_s$", "", regex=True).map(idx)
def hits(Fx, ms, p): return Fx[Fx.motif_id.isin(ms) & (Fx["p-value"] <= p)]
def classes(A, B, keep=None):
    """composite mask and best summed log-odds per element; keep(rowA, rowB) restricts to a fixed configuration"""
    hA = np.zeros(n, bool); hB = np.zeros(n, bool); hA[A.i.values] = True; hB[B.i.values] = True; comp = np.zeros(n, bool); sc = np.zeros(n)
    Bg = {k: d for k, d in B.groupby("i")}
    for r in A.itertuples():
        d = Bg.get(r.i)
        if d is None: continue
        g = np.maximum(r.start, d.start.values) - np.minimum(r.stop, d.stop.values) - 1; ok = (g >= 0) & (g <= 50)
        if keep is not None: ok &= keep(r, d)
        if ok.any(): comp[r.i] = True; sc[r.i] = max(sc[r.i], r.score + d.score.values[ok].max())
    return comp, hA & ~hB, hB & ~hA, ~hA & ~hB, sc
def binned(stat, g1, g0, b, ix=None):
    if ix is not None: stat, g1, g0, b = stat[ix], g1[ix], g0[ix], b[ix]
    n1 = np.bincount(b[g1], minlength=20).astype(float); n0 = np.bincount(b[g0], minlength=20).astype(float); s1 = np.bincount(b[g1], stat[g1], minlength=20); s0 = np.bincount(b[g0], stat[g0], minlength=20)
    v = (n1 >= 1) & (n0 >= 1); w = np.minimum(n1, n0)[v]
    if w.sum() == 0: return np.nan
    a = (s1[v] / n1[v] * w).sum(); c = (s0[v] / n0[v] * w).sum(); return a / c if c > 0 else np.nan
def ci(stat, g1, g0, b, B=500, seed=3):
    r = np.random.default_rng(seed); x = [binned(stat, g1, g0, b, r.integers(0, n, n)) for _ in range(B)]; return binned(stat, g1, g0, b), np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5)
def bins_of(*mus):
    rk = np.mean([pd.Series(m).rank(pct=True).values for m in mus], axis=0); return np.minimum((rk * 20).astype(int), 19)
def outcomes(a, b):
    ya, ma, _ = RES[a]; yb, mb, _ = RES[b]; ra = (ya + .5) / (ma + .5); rb = (yb + .5) / (mb + .5); bj = bins_of(ma, mb)
    return {"cobind_product": (ra * rb, bj), "cobind_min": (np.minimum(ra, rb), bj), f"own_{a}": ((ya + .5) / (ma + .5), bins_of(ma)), f"own_{b}": ((yb + .5) / (mb + .5), bins_of(mb))}, ya, yb
RES = {t: resid(t) for t in ["MLX", "SREBF2", "FOXK1", "DR1", "THAP6", "RBBP7", "SRCAP"]}; rows = []
def run(pair, a, b, design, A, B, ths, keep=None, shuffle_of=None):
    out, ya, yb = outcomes(a, b)
    for p in ths:
        comp, ao, bo, no, sc = classes(hits(A, A.motif_id.unique(), p), hits(B, B.motif_id.unique(), p), keep); single = ao | bo
        base = dict(pair=pair, partners=f"{a},{b}", design=design, fimo_p=p, n_composite=int(comp.sum()), n_A_only=int(ao.sum()), n_B_only=int(bo.sum()), n_none=int(no.sum()))
        cob = (ya > 0) & (yb > 0); base["frac_cobound_composite"] = round(cob[comp].mean(), 4) if comp.sum() else np.nan; base["frac_cobound_single"] = round(cob[single].mean(), 4) if single.sum() else np.nan
        for ctr, g0 in (("primary_vs_single", single), ("secondary_vs_none", no)):
            for k, (st, bn) in out.items():
                if comp.sum() < 5: rows.append(dict(base, contrast=ctr, outcome=k, ratio=np.nan, lo=np.nan, hi=np.nan)); continue
                x, lo, hi = ci(st, comp, g0, bn); rw = st[comp].mean() / st[g0].mean(); rows.append(dict(base, contrast=ctr, outcome=k, ratio=round(x, 3), lo=round(lo, 3), hi=round(hi, 3), raw_mean_ratio=round(rw, 3)))
        if design == "real_half_sites" and p == 1e-5 and comp.sum() >= 30:
            q = np.quantile(sc[comp], [1 / 3, 2 / 3]); T = [comp & (sc <= q[0]), comp & (sc > q[0]) & (sc <= q[1]), comp & (sc > q[1])]
            st, bn = out["cobind_product"]; r = np.random.default_rng(5); d = []
            for _ in range(500):
                ix = r.integers(0, n, n); d.append(binned(st, T[2], single, bn, ix) - binned(st, T[0], single, bn, ix))
            DOSE.append(dict(pair=pair, n_tertile=int(T[0].sum()), T1=round(binned(st, T[0], single, bn), 3), T2=round(binned(st, T[1], single, bn), 3), T3=round(binned(st, T[2], single, bn), 3), one_sided_p_T3_gt_T1=round(np.nanmean(np.array(d) <= 0), 4)))
DOSE = []; TH = (1e-4, 1e-5, 1e-6); sub = lambda ms: F[F.motif_id.isin(ms)]
# MLX-SREBF2 and its negative control read out at the same elements
MA, SB = ["MA0663.1"], ["MA0596.1", "MA0828.3"]
for pair, (a, b) in {"MLX-SREBF2": ("MLX", "SREBF2"), "RBBP7-SRCAP at MLX-SREBF2 sites (negative control)": ("RBBP7", "SRCAP")}.items():
    run(pair, a, b, "real_half_sites", sub(MA), sub(SB), TH)
    for s in (1, 2, 3): run(pair, a, b, f"shuffle{s}", sub([f"{m}.shuf{s}" for m in MA]), sub([f"{m}.shuf{s}" for m in SB]), (1e-4, 1e-5))
# FOXK1-MLX pre-registered: E-box core, 8 N, forkhead core, read on the forkhead strand
def core(meme_id, cons):
    b = open(f"{D}/real.meme").read().split(f"MOTIF {meme_id}")[1].split("MOTIF")[0].strip().split("\n"); i = [k for k, l in enumerate(b) if l.startswith("letter-probability")][0]; P = np.array([[float(x) for x in l.split()] for l in b[i + 1:] if l.strip() and not l.startswith("URL")]); L = "ACGT"
    amb = {"A": "A", "C": "C", "G": "G", "T": "T", "R": "AG", "Y": "CT"}; S = [sum(P[j + q, L.index(c)] for c in amb[ch]) for j in range(len(P) - len(cons) + 1) for q, ch in enumerate(cons)]; s = np.array(S).reshape(-1, len(cons)).sum(1); return int(s.argmax()), len(P)
eo, ew = core("MA0663.1", "CACGTG"); fo, fw = core("MA0852.3", "RTAAAYA")
def foxk_keep(tol):
    def keep(r, d):
        # r = FOXK1 hit, d = MLX hits; forkhead-strand coordinates: E-box core upstream of the forkhead core
        fs = r.start + fo if r.strand == "+" else r.stop - fo - 7 + 1; fe = fs + 6
        ep = np.where(d.strand.values == "+", d.start.values + eo, d.stop.values - eo - 6 + 1); een = ep + 5
        g = fs - een - 1 if r.strand == "+" else ep - fe - 1
        return np.abs(g - 8) <= tol
    return keep
for lab, tol in (("prereg_config_exact", 0), ("config_plus_minus_1", 1)):
    run("FOXK1-MLX", "FOXK1", "MLX", lab, sub(["MA0852.3"]), sub(MA), TH, keep=foxk_keep(tol))
    for s in (1, 2, 3): run("FOXK1-MLX", "FOXK1", "MLX", f"{lab}_shuffle{s}", sub([f"MA0852.3.shuf{s}"]), sub([f"MA0663.1.shuf{s}"]), (1e-4,), keep=foxk_keep(tol))
for cp in ("MLX_FOXA1", "MLX_FOXD2"):
    # paralog composite scanned whole: composite present vs absent, with half-site-only elements as the comparator
    out, ya, yb = outcomes("FOXK1", "MLX")
    for p in TH:
        h = np.zeros(n, bool); h[hits(F, [cp], p).i.values] = True; hF = np.zeros(n, bool); hF[hits(F, ["MA0852.3"], p).i.values] = True; hM = np.zeros(n, bool); hM[hits(F, MA, p).i.values] = True; g0 = (hF | hM) & ~h
        base = dict(pair="FOXK1-MLX", partners="FOXK1,MLX", design=f"capselex_composite_{cp}", fimo_p=p, n_composite=int(h.sum()), n_A_only=int((hF & ~hM & ~h).sum()), n_B_only=int((hM & ~hF & ~h).sum()), n_none=int((~hF & ~hM & ~h).sum()))
        for k, (st, bn) in out.items():
            if h.sum() < 5: rows.append(dict(base, contrast="primary_vs_single", outcome=k)); continue
            x, lo, hi = ci(st, h, g0, bn); rows.append(dict(base, contrast="primary_vs_single", outcome=k, ratio=round(x, 3), lo=round(lo, 3), hi=round(hi, 3), raw_mean_ratio=round(st[h].mean() / st[g0].mean(), 3)))
# DR1-THAP6: one motif, THAP1 as a family proxy; DR1 and THAP6 signal at motif vs no motif
for t in ("DR1", "THAP6"):
    y, mu, _ = RES[t]; st = (y + .5) / (mu + .5); bn = bins_of(mu)
    for lab, ms in [("THAP1_proxy", ["MA0597.3"])] + [(f"THAP1_proxy_shuffle{s}", [f"MA0597.3.shuf{s}"]) for s in (1, 2, 3)]:
        for p in TH:
            h = np.zeros(n, bool); h[hits(F, ms, p).i.values] = True
            if h.sum() < 5: rows.append(dict(pair="DR1-THAP6", partners=t, design=lab, fimo_p=p, n_composite=int(h.sum()), contrast="motif_vs_no_motif", outcome=f"own_{t}")); continue
            x, lo, hi = ci(st, h, ~h, bn); rows.append(dict(pair="DR1-THAP6", partners=t, design=lab, fimo_p=p, n_composite=int(h.sum()), n_none=int((~h).sum()), contrast="motif_vs_no_motif", outcome=f"own_{t}", ratio=round(x, 3), lo=round(lo, 3), hi=round(hi, 3), raw_mean_ratio=round(st[h].mean() / st[~h].mean(), 3)))
T = pd.DataFrame(rows); T.to_csv(f"{D}/pair_tests.tsv", sep="\t", index=False); Dd = pd.DataFrame(DOSE); Dd.to_csv(f"{D}/dose_tertiles.tsv", sep="\t", index=False)
# spacing discovery (run after the FOXK1-MLX pre-registered test): spacing x orientation against independent placement and dinucleotide-shuffled sequence, BH FDR 10%
Disc = []
def spacing(Fx, ms_a, ms_b, p):
    A = hits(Fx, ms_a, p); Bh = hits(Fx, ms_b, p); Bg = {k: d for k, d in Bh.groupby("i")}; c = {}; ex = {}
    for r in A.itertuples():
        d = Bg.get(r.i)
        if d is None: continue
        L = W[r.i]; wa = r.stop - r.start + 1
        for s, st, sp in zip(d.strand.values, d.start.values, d.stop.values):
            wb = sp - st + 1; g = max(r.start, st) - min(r.stop, sp) - 1; o = "AB" if r.start < st else "BA"; key = (r.strand + s, o)
            if 0 <= g <= 50: c[key + (g,)] = c.get(key + (g,), 0) + 1
            for gg in range(51):
                pr = max(L - wa - wb - gg + 1, 0) / max((L - wa + 1) * (L - wb + 1), 1); ex[(r.strand + s, "AB", gg)] = ex.get((r.strand + s, "AB", gg), 0) + pr; ex[(r.strand + s, "BA", gg)] = ex.get((r.strand + s, "BA", gg), 0) + pr
    return c, ex
for pair, a, b in (("MLX-SREBF2", MA, SB), ("FOXK1-MLX", ["MA0852.3"], MA)):
    c, ex = spacing(F, a, b, 1e-5); cd, _ = spacing(FD, a, b, 1e-5); R_ = []
    for k, e in ex.items():
        o = c.get(k, 0); bg = max(e, cd.get(k, 0), 0.5); R_.append(dict(pair=pair, orient=k[0], order=k[1], gap=k[2], obs=o, exp_indep=round(e, 2), dinuc=cd.get(k, 0), p=float(poisson.sf(o - 1, bg)) if o else 1.0))
    print(pair, "half-site pairs at p <= 1e-5 in the same element:", sum(c.values()), "within 50 bp;", "elements with both:", len(set(hits(F, a, 1e-5).i) & set(hits(F, b, 1e-5).i)))
    if not R_: continue
    Q = pd.DataFrame(R_).sort_values("p").reset_index(drop=True); Q["q"] = (Q.p * len(Q) / (Q.index + 1))[::-1].cummin()[::-1]; Q["q"] = Q.q.clip(upper=1); Disc.append(Q)
Dq = pd.concat(Disc) if Disc else pd.DataFrame(columns=["pair", "orient", "order", "gap", "obs", "exp_indep", "dinuc", "p", "q"]); Dq.to_csv(f"{D}/spacing_discovery.tsv", sep="\t", index=False, float_format="%.4g")
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 30)
print(T[T.outcome.astype(str).str.startswith("cobind") | T.design.astype(str).str.contains("THAP1")][["pair", "design", "fimo_p", "n_composite", "n_A_only", "n_B_only", "n_none", "frac_cobound_composite", "frac_cobound_single", "contrast", "outcome", "ratio", "lo", "hi", "raw_mean_ratio"]].to_string(index=False))
print(T[T.outcome.astype(str).str.startswith("own") & T.design.isin(["real_half_sites", "prereg_config_exact"]) & (T.contrast == "primary_vs_single")][["pair", "fimo_p", "outcome", "n_composite", "ratio", "lo", "hi"]].to_string(index=False))
print(Dd.to_string(index=False)); print(Dq[Dq.q <= 0.1].head(30).to_string(index=False)); print({p: int(((Dq.pair == p) & (Dq.q <= 0.1)).sum()) for p in Dq.pair.unique()}); print("reads:", {k: v[2] for k, v in RES.items()}); print("cores", eo, ew, fo, fw)
