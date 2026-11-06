"""Robustness of the Hi-C-first structure (no TF term) shown on 4E-H8b (conformation 1506).
a) split matrix, 10 kb O/E (WT Hi-C upper, model ensemble lower; same expected method: per-distance mean over kept bins of
   the locus), 18 loop-edges red / 6 Cicero-only blue circles, held-out pixels hatched, hub promoters as ticks;
b) model vs Hi-C O/E per pixel by class (restrained / held-out / unrestrained; pairs >= 20 kb, kept bins) with Spearman;
c) ensemble: loop-edges closed per structure (< 2 sigma), per-edge closure frequency, pairwise-distance CV among the 20
   edge-end promoters, radius of gyration of the 20 promoters (1506 marked);
d) hub compactness vs matched random promoter sets: Rg of the 20 promoters vs Rg of 20 TF-bound beads drawn within +-25 beads
   (50 kb) of each promoter (200 sets), per structure on every 10th structure."""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); a = ap.parse_args()
Bd = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/tf_polymer_hicfirst"); A = Bd / "hicfirst"; O = Bd / "figure_robustness"; O.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "axes.spines.top": False, "axes.spines.right": False})
RED, BLUE, INK, GREY = "#C0392B", "#4A6FA5", "#222222", "#9E9E9E"
L0, N, CONF = 29.73, 910, 1506
H = np.load(A / "inputs/hic_10kb.npz"); ice, keep = H["ice"], H["keep"]; n = ice.shape[0]
S = np.load(A / "analysis/sim_10kb_eps0.npy")
RS = pd.read_csv(A / "inputs/restraints.tsv", sep="\t"); HP = pd.read_csv(A / "inputs/heldout_pixels.tsv", sep="\t")
T = pd.read_csv(Bd / "figure_loopedges_flower/loop_edges.tsv", sep="\t"); HE = pd.read_csv(A / "figure_hub_flow/hub_edges_beads.tsv", sep="\t")
BL = HE[HE["on_loop_0.7"] == 0]
B = pd.read_csv(A / "inputs/beads.tsv", sep="\t"); EL = pd.read_csv(A / "figure_hub_flow/hub_elements.tsv", sep="\t")
def oe(M):
    E = np.full_like(M, np.nan, dtype=float)
    for d in range(n):
        i = np.arange(n - d); j = i + d; ok = keep[i] & keep[j] & np.isfinite(M[i, j])
        if ok.any():
            mu = M[i[ok], j[ok]].mean()
            if mu > 0: E[i, j] = E[j, i] = M[i, j] / mu
    return E
Ho, So = oe(ice), oe(S)
RES = np.zeros((n, n), bool); RES[RS.bin1, RS.bin2] = True; RES |= RES.T
HO = np.zeros((n, n), bool); HO[HP.bin1, HP.bin2] = True; HO |= HO.T
iu = np.triu_indices(n, 2); ok = keep[iu[0]] & keep[iu[1]] & np.isfinite(Ho[iu]) & np.isfinite(So[iu])
cls_ = np.where(HO[iu], "held-out", np.where(RES[iu], "restrained", "unrestrained"))
res = {}
for c in ("restrained", "held-out", "unrestrained"):
    m = ok & (cls_ == c); res[f"spearman_{c}"] = float(stats.spearmanr(So[iu][m], Ho[iu][m])[0]); res[f"n_{c}"] = int(m.sum())
m = ok & (cls_ != "restrained"); res["spearman_all_not_restrained"] = float(stats.spearmanr(So[iu][m], Ho[iu][m])[0])
# ensemble
files = sorted(glob.glob(str(Path(a.sim) / "traj*.npz")))
ends = np.array(sorted(set(T.b1) | set(T.b2))); closed, dmats, rg, conf = [], [], [], []
bound = np.where(B.bind.values)[0] if "bind" in B else np.where(B.n_tfs.values > 0)[0]
rng = np.random.default_rng(3); null_sets = []
for e in ends:
    c = bound[(np.abs(bound - e) <= 25) & ~np.isin(bound, ends)]; null_sets.append(c)
nullsets = np.array([[rng.choice(c) for c in null_sets] for _ in range(200)])
rg_hub_s, rg_null_s = [], []
for t, f in enumerate(files):
    X = np.load(f)["X"][:, :N]
    d = np.linalg.norm(X[:, T.b1] - X[:, T.b2], axis=2); closed.append(d < 2.0)
    P = X[:, ends]; dm = np.linalg.norm(P[:, :, None] - P[:, None], axis=3); dmats.append(dm)
    rg.append(np.sqrt(((P - P.mean(1, keepdims=True)) ** 2).sum(2).mean(1))); conf += [t * 50 + k + 1 for k in range(len(X))]
    for k in range(0, len(X), 10):
        rg_hub_s.append(rg[-1][k]); Q = X[k][nullsets]; rg_null_s.append(np.sqrt(((Q - Q.mean(1, keepdims=True)) ** 2).sum(2).mean(1)).mean())
closed = np.concatenate(closed); dmats = np.concatenate(dmats); rg = np.concatenate(rg); conf = np.array(conf)
nclosed = closed.sum(1); k1506 = int(np.where(conf == CONF)[0][0])
iu20 = np.triu_indices(len(ends), 1); cv = dmats[:, iu20[0], iu20[1]].std(0) / dmats[:, iu20[0], iu20[1]].mean(0)
rg_hub_s, rg_null_s = np.array(rg_hub_s), np.array(rg_null_s)
res.update(n_structures=int(len(conf)), mean_closed=float(nclosed.mean()), median_closed=float(np.median(nclosed)),
           frac_all18=float((nclosed == 18).mean()), conf1506_closed=int(nclosed[k1506]),
           per_edge_closure_min=float(closed.mean(0).min()), per_edge_closure_max=float(closed.mean(0).max()),
           pairdist_cv_median=float(np.median(cv)), pairdist_cv_range=[float(cv.min()), float(cv.max())],
           rg_hub_median=float(np.median(rg)), rg_1506=float(rg[k1506]), rg_1506_percentile=float((rg < rg[k1506]).mean() * 100),
           rg_null_median=float(np.median(rg_null_s)), rg_hub_vs_null_ratio=float(np.median(rg_hub_s) / np.median(rg_null_s)),
           frac_structures_hub_more_compact_than_null=float((rg_hub_s < rg_null_s).mean()))
pd.DataFrame(dict(gene1=T.gene1, gene2=T.gene2, span_kb=T.span_kb, closure_freq=closed.mean(0), closed_in_1506=closed[k1506])).to_csv(O / "edge_closure.tsv", sep="\t", index=False)
json.dump(res, open(O / "robustness.json", "w"), indent=1)
print(json.dumps(res, indent=1))
# ---------------- figure
fig = plt.figure(figsize=(180 / 25.4, 92 / 25.4))
w0, w1 = int((29.78 - L0) / 0.01), int((31.00 - L0) / 0.01)
ax = fig.add_axes([0.05, 0.14, 0.34, 0.34 * 180 / 92]); Z = np.full((w1 - w0, w1 - w0), np.nan)
sub = slice(w0, w1)
with np.errstate(divide="ignore", invalid="ignore"): Hs, Ss = np.log2(Ho[sub, sub]), np.log2(So[sub, sub])
kk = keep[sub]; bad = ~(kk[:, None] & kk[None, :]); Hs[bad | ~np.isfinite(Hs)] = np.nan; Ss[bad | ~np.isfinite(Ss)] = np.nan
ii, jj = np.triu_indices(w1 - w0, 1)
Z[ii, jj] = Hs[ii, jj]; Z[jj, ii] = Ss[jj, ii]
ext = (29.78, 31.00, 31.00, 29.78)
cm_ = plt.get_cmap("RdBu_r").copy(); cm_.set_bad("#EEEEEE")
ax.imshow(np.ma.masked_invalid(Z), cmap=cm_, vmin=-1.5, vmax=1.5, extent=ext, interpolation="nearest")
ax.plot([29.78, 31.0], [29.78, 31.0], color="white", lw=0.6)
for r in HP.itertuples():
    if w0 <= r.bin1 < w1 and w0 <= r.bin2 < w1 and keep[r.bin1] and keep[r.bin2]:
        x, y = L0 + r.bin2 * 0.01, L0 + r.bin1 * 0.01
        for (px, py) in ((x, y), (y, x)): ax.add_patch(Rectangle((px, py), 0.01, 0.01, fill=False, ec="#555555", lw=0.25, hatch="////"))
for E_, c in ((T, RED), (BL, BLUE)):
    for r in E_.itertuples():
        p1, p2 = L0 + 0.002 * r.b1, L0 + 0.002 * r.b2
        for (px, py) in ((p2, p1), (p1, p2)): ax.add_patch(Circle((px, py), 0.012, fill=False, ec=c, lw=0.7))
hubpos = L0 + 0.002 * EL.bead.values + 0.001
ax.set_xticks(hubpos, minor=True); ax.set_yticks(hubpos, minor=True); ax.tick_params(which="minor", length=2.5, color=INK, width=0.6)
ax.set_xticks([29.8, 30.2, 30.6, 31.0]); ax.set_yticks([29.8, 30.2, 30.6, 31.0]); ax.set_xlabel("chr16 (Mb)")
for s_ in ("top", "right"): ax.spines[s_].set_visible(True)
ax.text(30.98, 29.8, "WT Hi-C", ha="right", va="top", fontsize=6.5, bbox=dict(fc="white", ec="none", pad=0.5)); ax.text(29.8, 30.98, "model ensemble", ha="left", va="bottom", fontsize=6.5, bbox=dict(fc="white", ec="none", pad=0.5))
ax.set_xlim(29.78, 31.0); ax.set_ylim(31.0, 29.78); ax.set_title("a  Hi-C vs Hi-C-constrained model (10 kb)", loc="left", fontsize=7)
fig.text(0.05, 0.015, "circles: red = loop-edge, blue = Cicero-only edge; hatched = held-out pixel; ticks = hub promoters; colour = log2 O/E (±1.5)", fontsize=5.3)
# b
ax = fig.add_axes([0.48, 0.6, 0.19, 0.32])
for c, col in (("unrestrained", "#BDBDBD"), ("restrained", RED), ("held-out", INK)):
    mm = ok & (cls_ == c); x, y = np.log2(Ho[iu][mm]), np.log2(So[iu][mm])
    ax.scatter(x, y, s=1.2 if c == "unrestrained" else 2.5, color=col, lw=0, alpha=0.35 if c == "unrestrained" else 0.7, rasterized=True,
               label=f"{c} (n={mm.sum():,}; ρ={res['spearman_' + c]:.2f})")
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.plot([-3, 3], [-3, 3], color="#DDDDDD", lw=0.5)
ax.set_xlabel("Hi-C log2 O/E"); ax.set_ylabel("model log2 O/E"); ax.legend(frameon=False, fontsize=4.8, loc="lower right", markerscale=3, handletextpad=0.1, bbox_to_anchor=(1.05, -0.02))
ax.set_title("b  Pixel agreement (≥ 20 kb)", loc="left", fontsize=7)
# c1 closed per structure
ax = fig.add_axes([0.76, 0.6, 0.22, 0.32])
ax.hist(nclosed, bins=np.arange(-0.5, 19), color=GREY); ax.axvline(nclosed[k1506], color=RED, lw=1); ax.axvline(nclosed.mean(), color=INK, lw=0.8, ls=(0, (2, 1)))
ax.text(nclosed[k1506] - 0.3, ax.get_ylim()[1] * 0.92, "#1506", color=RED, ha="right", fontsize=5.5)
ax.set_xlabel("loop-edges closed (of 18)"); ax.set_ylabel("structures"); ax.set_title(f"c  Ensemble: mean {nclosed.mean():.1f}/18", loc="left", fontsize=7)
# c2 per-edge closure
ax = fig.add_axes([0.48, 0.24, 0.29, 0.2]); fq = closed.mean(0); o_ = np.argsort(T.span_kb.values)
ax.bar(range(18), fq[o_], color=RED, width=0.7); ax.set_xticks(range(18)); ax.set_xticklabels([f"{T.gene1[i]}–{T.gene2[i]}" for i in o_], rotation=90, fontsize=4.0)
ax.set_ylim(0, 1); ax.set_ylabel("fraction closed"); ax.set_title("per loop-edge (shortest → longest)", loc="left", fontsize=6)
# d compactness
ax = fig.add_axes([0.85, 0.24, 0.13, 0.2])
vp = ax.violinplot([rg_null_s, rg_hub_s], positions=[0, 1], showextrema=False, widths=0.8)
for b_, c in zip(vp["bodies"], (GREY, RED)): b_.set_facecolor(c); b_.set_alpha(0.55); b_.set_edgecolor("none")
ax.scatter([1], [rg[k1506]], s=8, color=INK, zorder=3); ax.set_xticks([0, 1]); ax.set_xticklabels(["matched\nTF-bound sets", "hub\npromoters"], fontsize=5.5)
ax.set_ylabel("Rg, 20 sites (σ)"); ax.set_title("d  Hub compactness", loc="left", fontsize=6.5)
for e in ("svg", "pdf"): fig.savefig(O / f"robustness.{e}", transparent=True)
fig.savefig(O / "robustness.png", dpi=600, facecolor="white"); fig.savefig(O / "robustness_preview.png", dpi=250, facecolor="white")
