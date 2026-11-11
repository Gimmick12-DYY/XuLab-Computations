# m16_strong_phenotype part B (SPEC.md): simple model (q005 peaks x ABC; and promoter-only) vs Arm 2 raw z, strong-phenotype TFs vs the rest. Diagnostic, benchmark already read.
# usage: m16_compare.py -> out/simple_per_tf.tsv, out/simple_pooled.tsv, out/simple_deciles.tsv, out/simple_summary.png/.svg
import json, os
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import glob as _g, matplotlib.font_manager as _fm
for _f in _g.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): _fm.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Arial"; plt.rcParams["svg.fonttype"] = "none"
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m16_strong_phenotype"; OUT = f"{O}/out"; LAB = "diagnostic, benchmark already read"; SEED = 20261111; rng = np.random.default_rng(SEED)
OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}; tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]
MIS = dict(zip(tfs79, json.load(open(f"{RG}/m7_step2/features/raw_hicfixed/feature_qc.json"))["mis_paired_map"]["mis0"]))
SP = pd.read_csv(f"{OUT}/strong_phenotype_per_tf.tsv", sep="\t").set_index("panel_tf"); STRONG = set(SP.index[SP.status == "strong phenotype"])
Rz = np.load(f"{RG}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True); TT = list(Rz["tfs"]); ug = pd.Index(Rz["genes"].astype(str)); n = len(ug); Z = Rz["r1"].astype(float).copy()
for i, t in enumerate(TT):
    for o in OWN.get(t, [t]):
        if o in ug: Z[i, ug.get_loc(o)] = np.nan
W = {k: pd.read_csv(f"{RG}/m11_peak_regulome/out/wide_{k}_q0.05.tsv.gz", sep="\t", index_col=0).reindex(columns=ug).fillna(0) for k in ("score", "promoter")}
GLf = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t").drop_duplicates("name", keep="last").set_index("name"); acol = [c for c in GLf.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]
atac = GLf[acol].reindex(ug).fillna(0).values; GW = pd.read_csv(f"{RG}/m7_step2/splits/gene_split_wide.tsv", sep="\t"); BW = np.load(f"{RG}/m7_step2/features/raw_hicfixed_wide/base.npz"); cgi = pd.Series(BW["base"][:, 4], index=GW.gene).reindex(ug).fillna(0).values
strat = pd.qcut(pd.Series(atac).rank(method="first"), 10, labels=False).values * 2 + (cgi > 0); SI = {s: np.flatnonzero(strat == s) for s in np.unique(strat)}
def toplist(p, ok):  # top 1,000 genes by P among P > 0 (all P > 0 genes if fewer), within the TF's usable genes
    p = np.where(ok, p, 0); pos = np.flatnonzero(p > 0); o = pos[np.lexsort((rng.random(len(pos)), -p[pos]))][:1000]; m = np.zeros(n, bool); m[o] = True; return m
def or_(tg, rs): a = (tg & rs).sum(-1); b = (tg & ~rs).sum(-1); c = (~tg & rs).sum(-1); d = (~tg & ~rs).sum(-1); return (a + .5) * (d + .5) / ((b + .5) * (c + .5))
def sp(a, b): m = np.isfinite(a) & np.isfinite(b); return np.nan if m.sum() < 10 or np.std(a[m]) == 0 else float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])
ROWS = []; DEC = []; TAB = {}
for ver in ("score", "promoter"):
    for i, t in enumerate(TT):
        if t not in W[ver].index: continue
        y = Z[i]; ok = np.isfinite(y); a = np.abs(np.nan_to_num(y)); rs = np.zeros(n, bool); idx = np.flatnonzero(ok); rs[idx[np.argsort(-a[idx], kind="stable")[:int(round(ok.sum() * .05))]]] = True
        p = W[ver].loc[t].values; tg = toplist(p, ok); tg_ok = tg & ok
        A_, B_ = tg_ok[ok], rs[ok]; o = float(or_(A_, B_)); bi = rng.integers(0, ok.sum(), (1000, ok.sum())); bo = or_(A_[bi], B_[bi])
        cnt = {s: int((tg_ok & (strat == s)).sum()) for s in SI}; nul = []
        for _ in range(200):
            m = np.zeros(n, bool)
            for s, c in cnt.items():
                if c: pool = SI[s][ok[SI[s]]]; m[rng.choice(pool, min(c, len(pool)), replace=False)] = True
            nul.append(or_(m[ok], B_))
        sw = MIS.get(t); osw = float(or_(toplist(W[ver].loc[sw].values, ok)[ok], B_)) if sw in W[ver].index else np.nan
        spr = sp(np.where(ok, p, np.nan), np.where(ok, a, np.nan)); bsp = [sp(np.where(ok, p, np.nan)[j], np.where(ok, a, np.nan)[j]) for j in (rng.integers(0, n, n) for _ in range(300))]
        TAB[(ver, t)] = (int((A_ & B_).sum()), int((A_ & ~B_).sum()), int((~A_ & B_).sum()), int((~A_ & ~B_).sum()))
        ROWS.append(dict(label=LAB, version="peak_score_x_abc" if ver == "score" else "promoter_only", tf=t, strong_phenotype=t in STRONG, kd_status=SP.status.get(t, ""), n_genes=int(ok.sum()), n_list=int(tg_ok.sum()),
                         OR=o, OR_lo=float(np.percentile(bo, 2.5)), OR_hi=float(np.percentile(bo, 97.5)), OR_matched=float(np.median(nul)), swap_tf=sw, OR_swapped=osw, spearman_P_absz=spr, sp_lo=float(np.nanpercentile(bsp, 2.5)), sp_hi=float(np.nanpercentile(bsp, 97.5))))
        pos_ = (p > 0) & ok; groups = [("P=0", ~(p > 0) & ok)]
        if pos_.sum() >= 10:
            dq = pd.qcut(pd.Series(p[pos_]).rank(method="first"), 10, labels=False).values; ix = np.flatnonzero(pos_)
            for k in range(10): m = np.zeros(n, bool); m[ix[dq == k]] = True; groups.append((f"D{k + 1}", m))
        for gname, m in groups:
            v = a[m]; bv = v[rng.integers(0, len(v), (500, len(v)))].mean(1) if len(v) else np.array([np.nan])
            DEC.append(dict(label=LAB, version=ver, tf=t, group=gname, n=int(m.sum()), mean_absz=float(v.mean()) if len(v) else np.nan, lo=float(np.nanpercentile(bv, 2.5)), hi=float(np.nanpercentile(bv, 97.5))))
R = pd.DataFrame(ROWS); R.loc[R.n_list == 0, ["OR", "OR_lo", "OR_hi", "OR_matched", "spearman_P_absz", "sp_lo", "sp_hi"]] = np.nan  # empty list (no q005 peak linked): NA, not the +0.5-corrected 19.0
for (ver, t) in list(TAB):
    if TAB[(ver, t)][0] + TAB[(ver, t)][1] == 0: del TAB[(ver, t)]
EMPTY = {(v, t) for v in ("score", "promoter") for t in W[v].index if not (W[v].loc[t].values > 0).any()}
R.loc[[(("score" if r.version == "peak_score_x_abc" else "promoter"), r.swap_tf) in EMPTY for r in R.itertuples()], "OR_swapped"] = np.nan
R.to_csv(f"{OUT}/simple_per_tf.tsv", sep="\t", index=False, float_format="%.4f"); pd.DataFrame(DEC).to_csv(f"{OUT}/simple_deciles.tsv", sep="\t", index=False, float_format="%.4f")
def mh(keys):  # Mantel-Haenszel OR with Robins-Breslow-Greenland 95% interval
    num = den = P_ = Q_ = R_ = S_ = 0.0
    for k in keys:
        a, b, c, d = TAB[k]; N = a + b + c + d
        if N == 0: continue
        r, s = a * d / N, b * c / N; num += r; den += s; p_, q_ = (a + d) / N, (b + c) / N; P_ += p_ * r; Q_ += (p_ * s + q_ * r); S_ += q_ * s
    orm = num / den; v = P_ / (2 * num ** 2) + Q_ / (2 * num * den) + S_ / (2 * den ** 2); se = np.sqrt(v); return orm, orm * np.exp(-1.96 * se), orm * np.exp(1.96 * se)
PO = []
for ver, G in R.groupby("version"):
    for grp, H in (("strong_phenotype", G[G.strong_phenotype]), ("not_strong", G[~G.strong_phenotype])):
        if not len(H): continue
        H = H[H.OR.notna()]; dlt = (H.OR - H.OR_matched).values; b = np.median(dlt[rng.integers(0, len(dlt), (2000, len(dlt)))], axis=1); m, lo, hi = mh([k for k in [(("score" if ver == "peak_score_x_abc" else "promoter"), t) for t in H.tf] if k in TAB])
        PO.append(dict(label=LAB, version=ver, group=grp, n_tfs=len(H), median_OR=H.OR.median(), median_OR_matched=H.OR_matched.median(), median_OR_minus_matched=float(np.median(dlt)), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                       frac_OR_gt_matched=float((H.OR > H.OR_matched).mean()), frac_OR_gt_swapped=float((H.OR > H.OR_swapped)[H.OR_swapped.notna()].mean()), mantel_haenszel_OR=m, mh_lo=lo, mh_hi=hi, mean_spearman=H.spearman_P_absz.mean(), tfs=",".join(H.tf)))
PO = pd.DataFrame(PO); PO.to_csv(f"{OUT}/simple_pooled.tsv", sep="\t", index=False, float_format="%.4f"); print(PO.drop(columns=["tfs", "label"]).to_string(index=False))
# figure
fig = plt.figure(figsize=(11, 5.5)); ax = fig.add_axes([0.07, 0.30, 0.62, 0.62]); col = {"peak_score_x_abc": "#E08214", "promoter_only": "#2C6FB7"}
G = R[(R.version == "peak_score_x_abc") & R.OR.notna()].sort_values("OR").reset_index(drop=True); xs = np.arange(len(G)); Pm = R[R.version == "promoter_only"].set_index("tf")
for k, r in G.iterrows():
    c = col["peak_score_x_abc"] if r.strong_phenotype else "#BBBBBB"; ax.plot([k, k], [r.OR_lo, r.OR_hi], color=c, lw=1); ax.plot(k, r.OR, "o", color=c, ms=4); ax.plot([k - .3, k + .3], [r.OR_matched] * 2, color="#555555", lw=1)
    if r.tf in Pm.index: ax.plot(k + .25, Pm.loc[r.tf, "OR"], "s", color=col["promoter_only"] if r.strong_phenotype else "#DDDDDD", ms=3)
ax.axhline(1, color="k", lw=.5); ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(G.tf, rotation=90, fontsize=6); ax.set_ylabel("OR: top-1,000 peak-score genes among top-5% |z| responders")
ax.set_title(f"Simple model vs HEK293T Perturb-seq (Arm 2). Coloured = strong-phenotype knockdown (authors' criteria, reproduced; n = {int(G.strong_phenotype.sum())}); grey tick = matched genes", fontsize=8)
ax.plot([], [], "o", color=col["peak_score_x_abc"], label="promoter + enhancer (peak x ABC)"); ax.plot([], [], "s", color=col["promoter_only"], label="promoter only"); ax.plot([], [], color="#555555", label="matched genes"); ax.legend(fontsize=7, loc="upper left")
ax2 = fig.add_axes([0.76, 0.40, 0.21, 0.42]); D = pd.DataFrame(DEC)
for ver in ("score", "promoter"):
    Q = D[(D.tf == "NRF1") & (D.version == ver)]
    if len(Q): x2 = np.arange(len(Q)); ax2.errorbar(x2, Q.mean_absz, yerr=[Q.mean_absz - Q.lo, Q.hi - Q.mean_absz], marker="o", ms=3, color=col["peak_score_x_abc" if ver == "score" else "promoter_only"], capsize=1.5, lw=1); ax2.set_xticks(x2); ax2.set_xticklabels(Q.group, rotation=90, fontsize=6)
ax2.set_title("NRF1: mean |z| by peak-score decile", fontsize=8); ax2.set_ylabel("mean |z|", fontsize=7); fig.text(0.07, 0.02, LAB, fontsize=6)
fig.savefig(f"{OUT}/simple_summary.png", dpi=200); fig.savefig(f"{OUT}/simple_summary.svg"); print("done")
