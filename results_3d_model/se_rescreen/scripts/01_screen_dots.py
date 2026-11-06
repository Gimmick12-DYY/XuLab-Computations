"""Re-screen of super-enhancer candidates by partner Hi-C dots. Candidates = loop-supported anchor-hub cores (>= 5 TFs, Cicero edges on a WT Peakachu 0.7 loop, +-30 kb, > 25 kb) lying in one of the 342 ROSE-style ENCODE HEK293 SE candidates, core distal (no TSS within 2.5 kb), as for SE_0027.
Dot test per partner, 10 kb, as on slide 20: centre = 3x3 O/E (expected = that chromosome's mean balanced contact per diagonal over valid bin pairs), donut = Chebyshev distance 2-5 excluding the centre's row and column and pixels within 2 bins of the diagonal; enrichment = centre / donut; each end may shift +-20 kb, requiring >= 30 raw reads in the centre. Null: 2,000 random pixels on the same chromosome at the same distance, same search. BH-FDR across all tested pixels. Clear dot = enrichment >= 1.5, q < 0.05. Partners whose best pixels are within 1 bin count as one distinct dot."""
import gzip, re, sys
from collections import defaultdict
from pathlib import Path
import h5py, numpy as np, pandas as pd
IN = sys.argv[1]; R0 = Path(__file__).resolve().parents[1]; Bd = R0.parent / "se_hub"; I = R0 / "inputs"; r = 10000; DM = 320; rng = np.random.default_rng(23); NN = 2000; SH = np.array([-2, -1, 0, 1, 2])
f = h5py.File(IN + "/wt.mcool", "r"); g = f[f"resolutions/{r}"]; names = [x.decode() for x in g["chroms/name"][:]]; co = g["indexes/chrom_offset"][:]; bo = g["indexes/bin1_offset"]
Band = {}
def load(c):
    if c in Band: return Band[c]
    k = names.index(c); c0, c1 = int(co[k]), int(co[k + 1]); lo, hi = int(bo[c0]), int(bo[c1]); a = g["pixels/bin1_id"][lo:hi] - c0; b = g["pixels/bin2_id"][lo:hi] - c0; v = g["pixels/count"][lo:hi]; n = c1 - c0; d = b - a; m = (d <= DM) & (b < n); a, d, b, v = a[m], d[m], b[m], v[m]; w = g["bins/weight"][c0:c1]; valid = np.isfinite(w)
    Raw = np.zeros((n, DM + 1), np.float32); Raw[a, d] = v; Ice = np.zeros((n, DM + 1), np.float32); ww = np.nan_to_num(w); Ice[a, d] = v * ww[a] * ww[b]; exp = np.full(DM + 1, np.nan)
    for dd in range(DM + 1):
        ok = valid[:n - dd] & valid[dd:]
        if ok.any(): exp[dd] = Ice[:n - dd, dd][ok].mean()
    Band.clear(); Band[c] = (Raw, Ice, valid, exp, n); return Band[c]
di, dj = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6), indexing="ij"); ringm = (np.maximum(abs(di), abs(dj)) >= 2) & (di != 0) & (dj != 0); cenm = np.maximum(abs(di), abs(dj)) <= 1
def score(c, I_, J_):
    """I_, J_ arrays of pixels (i<j); best of 25 shifts per pixel; returns best enrichment, O/E, reads, shift indices, and exact-pixel enrichment."""
    Raw, Ice, valid, exp, n = load(c); I_ = np.asarray(I_); J_ = np.asarray(J_); out = np.full(len(I_), np.nan); oe_ = out.copy(); rd = np.zeros(len(I_)); sa = np.zeros(len(I_), int); sb = sa.copy(); ex = out.copy()
    for a in SH:
        for b in SH:
            ii = I_ + a; jj = J_ + b; ok = (ii >= 5) & (jj + 5 < n) & (jj - ii >= 5) & (jj - ii + 10 <= DM); Ei = np.full((len(I_), 121), np.nan); Rr = np.zeros(len(I_))
            iw = ii[:, None] + di.ravel(); jw = jj[:, None] + dj.ravel(); dw = jw - iw; good = ok[:, None] & (iw >= 0) & (jw < n) & (dw >= 0) & (dw <= DM); ix, ky = np.where(good); vi = iw[ix, ky]; vj = jw[ix, ky]; vd = dw[ix, ky]; val = valid[vi] & valid[vj]; e = np.full(len(ix), np.nan); e[val] = Ice[vi[val], vd[val]] / exp[vd[val]]; Ei[ix, ky] = e
            mr = np.zeros((len(I_), 121), bool); mr[ix, ky] = True; rmask = ringm.ravel()[None, :] & ((dw) >= 2); C = cenm.ravel()
            with np.errstate(all="ignore"):
                cen = np.nanmean(Ei[:, C], axis=1); E2 = Ei.copy(); E2[~rmask] = np.nan; bg = np.nanmean(E2, axis=1); en = cen / bg
            ci = np.where(C)[0]; RRv = np.zeros((len(I_), 121), np.float32); RRv[ix, ky] = Raw[vi, vd]; Rr = RRv[:, C].sum(1); use = ok & np.isfinite(en) & (bg > 0)
            if a == 0 and b == 0: ex = np.where(use, en, np.nan)
            better = use & (Rr >= 30) & (~np.isfinite(out) | (en > out)); out[better] = en[better]; oe_[better] = cen[better]; rd[better] = Rr[better]; sa[better] = a; sb[better] = b
    return out, oe_, rd, sa, sb, ex
import warnings; warnings.simplefilter("ignore")
SE = pd.read_csv(Bd / "step1/HEK293_SE_rose_hg38.tsv", sep="\t"); T2 = pd.read_csv(I / "HEK293T_SE_candidates_consensus_hg38.tsv", sep="\t"); HP = pd.read_csv(I / "anchor_hub_partners.tsv", sep="\t"); HN = pd.read_csv(I / "anchor_hubs_named.tsv", sep="\t")
G = pd.read_csv(gzip.open(IN + "/genes_exons.tsv.gz", "rt"), sep="\t", header=None, names="c t s e st g ty".split()); Gg = G[G.t == "gene"]; Gt = Gg[Gg.ty.isin(["pc", "lnc"])].copy(); Gt["tss"] = np.where(Gt.st == "+", Gt.s, Gt.e); bygt = {c: d for c, d in Gt.groupby("c")}; bygg = {c: d for c, d in Gg.groupby("c")}; byex = {c: d for c, d in G[(G.t == "exon") & (G.ty == "pc")].groupby("c")}
def pr(x): c, a = x.split(":"); s, e = map(int, a.split("-")); return c, s, e
def tss_g(x):
    c, s, e = pr(x); d = bygt.get(c); return sorted(set(d.g[(d.tss >= s - 2500) & (d.tss <= e + 2500)])) if d is not None else []
ART = {"SERPINH1", "DNAJB1", "BRD4", "RPS3", "TECR"}
def artifact(x):
    c, s, e = pr(x); d = bygg.get(c); gn = set(d.g[(d.e >= s - 5000) & (d.s <= e + 5000)]) if d is not None else set(); hist = any(re.match(r"^(H1-|H2A|H2B|H3C|H4C|H3-|H4-)", q) for q in gn)
    return bool(gn & ART) or hist or (c == "chr11" and s < 75700000 and e > 75300000) or (c == "chr6" and s < 28000000 and e > 25500000)
def exonic(x):
    c, s, e = pr(x); d = byex.get(c); return d is not None and bool(((d.e >= s) & (d.s <= e)).any())
rows = []
for h in HN.itertuples():
    c, s, e = pr(h.core); q = SE[(SE.chrom == c) & (SE.end > s) & (SE.start < e)]
    if q.empty: continue
    se = q.iloc[0]; t = T2[(T2.chrom == c) & (T2.end > s) & (T2.start < e)]; rows.append(dict(core=h.core, chrom=c, se_id=se.se_id, se=f"{c}:{se.start}-{se['end']}", se_rank=se["rank"], core_n_tfs=h.n_tfs, core_tfs=h.tfs, core_tss=",".join(tss_g(h.core)), hek293t_rank=int(t["rank"].iloc[0]) if len(t) else -1))
C = pd.DataFrame(rows); print("cores in an SE", len(C), "distal", (C.core_tss == "").sum(), "SEs", C.se_id.nunique()); C = C[C.core_tss == ""].reset_index(drop=True); Pr = HP[HP.core.isin(C.core) & (HP.distance_bp > 25000)].copy(); print("partners", len(Pr))
Pr["chrom"] = [pr(x)[0] for x in Pr.partner]; res = []; nulc = {}
for c, sub in Pr.groupby("chrom"):
    Raw, Ice, valid, exp, n = load(c); nulc = {}
    bc = np.array([sum(pr(x)[1:]) // 2 // r for x in sub.core]); bp = np.array([sum(pr(x)[1:]) // 2 // r for x in sub.partner]); I_, J_ = np.minimum(bc, bp), np.maximum(bc, bp); be, oe, rd, sa, sb, ex = score(c, I_, J_); ps = []
    for b_, d_ in zip(be, J_ - I_):
        if not np.isfinite(b_): ps.append(np.nan); continue
        if d_ not in nulc:
            a = rng.integers(10, n - d_ - 10, NN * 2); a = a[valid[a] & valid[a + d_]][:NN]; nulc[d_] = score(c, a, a + d_)[0]; nulc[d_] = nulc[d_][np.isfinite(nulc[d_])]
        nn = nulc[d_]; ps.append(float((1 + (nn >= b_).sum()) / (len(nn) + 1)))
    sub = sub.assign(i=I_, j=J_, best_don=be, best_oe=oe, best_reads=rd, si=sa, sj=sb, exact_don=ex, p=ps); res.append(sub); print(c, len(sub), len(nulc), flush=True)
P = pd.concat(res).reset_index(drop=True); t_ = P.p.notna(); m = int(t_.sum()); o = P.p[t_].sort_values(); q = (o * m / np.arange(1, m + 1))[::-1].cummin()[::-1]; P["q"] = np.nan; P.loc[o.index, "q"] = q; P["clear"] = (P.best_don >= 1.5) & (P.q < 0.05); P["clear_raw"] = (P.best_don >= 1.5) & (P.p < 0.05); P["untestable"] = ~t_
P["bi"] = P.i + P.si; P["bj"] = P.j + P.sj; P["p_tss"] = [",".join(tss_g(x)) for x in P.partner]; P["p_art"] = [artifact(x) for x in P.partner]; P["p_exon"] = [exonic(x) for x in P.partner]; P.to_csv(R0 / "partner_dots_all.tsv", sep="\t", index=False); print("tested", m, "of", len(P), "clear (q)", int(P.clear.sum()), "clear (raw p)", int(P.clear_raw.sum()))
def distinct(d):
    pts = d[["bi", "bj"]].values; lab = list(range(len(pts)))
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            if max(abs(pts[a] - pts[b])) <= 1:
                la, lb = lab[a], lab[b]; lab = [la if x == lb else x for x in lab]
    return len(set(lab))
out = []
for k in C.itertuples():
    d = P[P.core == k.core]
    if d.empty: continue
    cl = d[d.clear]; clr = d[d.clear_raw]; out.append(dict(se_id=k.se_id, core=k.core, se=k.se, se_rank=k.se_rank, hek293t_rank=k.hek293t_rank, core_n_tfs=k.core_n_tfs, n_partners=len(d), n_tested=int(d.p.notna().sum()), clear=len(cl), distinct_clear=distinct(cl) if len(cl) else 0, distinct_raw=distinct(clr) if len(clr) else 0, frac_clear=len(cl) / len(d), mean_don=float(cl.best_don.mean()) if len(cl) else np.nan, mean_partner_tfs=float(d.partner_n_tfs_total.mean()), genes=",".join(sorted({g_ for x in d.p_tss for g_ in x.split(",") if g_}))[:70], artifact=artifact(k.core) or bool(d.p_art.mean() > 0.3), frac_exon=float(d.p_exon.mean()), ctcf_only=k.core_tfs == "CTCF"))
Q = pd.DataFrame(out).sort_values(["distinct_clear", "frac_clear", "mean_don"], ascending=False); Q.to_csv(R0 / "se_candidates_ranked.tsv", sep="\t", index=False); pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); cols = "se_id core se_rank hek293t_rank core_n_tfs n_partners n_tested clear distinct_clear distinct_raw frac_clear mean_don mean_partner_tfs artifact frac_exon genes".split()
print("candidates", len(Q), "; with hek293T", (Q.hek293t_rank > 0).sum(), "; artifact", Q.artifact.sum()); print(Q[cols].head(15).round(2).to_string(index=False)); print(Q.reset_index(drop=True).assign(rk=lambda x: x.index + 1)[lambda x: x.se_id == "HEK293_SE_0027"][["rk"] + cols].round(2).to_string(index=False))
