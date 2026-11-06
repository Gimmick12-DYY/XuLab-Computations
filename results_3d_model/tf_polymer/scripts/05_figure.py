"""Step 5 (Mac): Illustrator-ready panels for the TF-binding polymer (v3 style).

Arial 7 pt, live SVG text (svg.fonttype none), PDF fonttype 42. Each panel is its own SVG/PDF; raster parts are separate
600 dpi PNGs, and *_linked.svg versions link to them (keep panels/ and raster/ side by side).
  1_conformation   PyMOL ray trace (transparent) + vector hub-core labels, leader lines and legend
  2_split_matrix   Hi-C (upper) vs simulation (lower), 10 kb, each log10 and scaled to its own 1-99th percentile; + P(s)
  3_contacts       contact frequency / matched expectation: loops (positive control), Cicero hub / shared / step-1 edges
  4_cocluster      hub-core co-clustering in TF clusters vs class- and distance-matched bound beads
  5_gallery        6 random conformations (optional)
"""
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="local copy of tf_polymer (inputs/, analysis/, render/)")
a = ap.parse_args()
Bd = Path(a.base); I = Bd / "inputs"; A = Bd / "analysis"; RN = Bd / "render"
F = Bd / "figure"; P = F / "panels"; RD = F / "raster"; P.mkdir(parents=True, exist_ok=True); RD.mkdir(exist_ok=True)
FS, FSS = 7, 6
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": FS, "axes.titlesize": FS,
                     "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
                     "ytick.major.size": 2, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "savefig.dpi": 600, "mathtext.default": "regular", "axes.spines.top": False, "axes.spines.right": False})
MM = 1 / 25.4; PT = 72 / 25.4
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]
TFC, CORE, TUBE = "#7FA7D9", "#262626", "#9A9A9A"

SM = pd.read_csv(A / "sweep_summary.tsv", sep="\t"); best = float(SM.loc[SM.chosen, "eps_top"].iloc[0])
V = pd.read_csv(A / "validation.tsv", sep="\t"); V = V[V.eps_top == best]
VP = pd.read_csv(A / "validation_pairs.tsv", sep="\t"); VP = VP[VP.eps_top == best]
B = pd.read_csv(I / "beads.tsv", sep="\t").fillna({"hub_label": "", "tfs": ""})
H = np.load(I / "hic_10kb.npz"); ice = H["ice"]; keep = H["keep"]; nbin = ice.shape[0]
S = np.load(A / f"sim_10kb_eps{best:g}.npy")
rep = json.load(open(A / "representative.json"))
row = SM[SM.chosen].iloc[0]

def fig_mm(w, h): return plt.figure(figsize=(w * MM, h * MM))
def ax_mm(fig, x, y, w, h):
    W, Hh = fig.get_size_inches() / MM; return fig.add_axes([x / W, y / Hh, w / W, h / Hh])
def save(fig, stem, transparent=False):
    for ext in ("svg", "pdf"): fig.savefig(P / f"{stem}.{ext}", transparent=transparent)
    fig.savefig(F / f"{stem}_preview.png", dpi=200, transparent=transparent); plt.close(fig)
def relink(stem, href, x, y, w, h, Hfig):
    """copy of stem.svg with its first embedded image replaced by a link (x, y, w, h in mm from the bottom-left)."""
    t = open(P / f"{stem}.svg").read(); m = next(re.finditer(r"<image\b.*?/>", t, re.S))
    new = (f'<image xlink:href="{href}" x="{x * PT:.4f}" y="{(Hfig - y - h) * PT:.4f}" width="{w * PT:.4f}" '
           f'height="{h * PT:.4f}" preserveAspectRatio="none"/>')
    open(P / f"{stem}_linked.svg", "w").write(t[:m.start()] + new + t[m.end():])

# ---------------- 1: representative conformation ----------------
img = np.asarray(Image.open(RN / "representative.png").convert("RGBA")).astype(float) / 255
ih, iw = img.shape[:2]
al = img[..., 3] > 0.02; ys, xs = np.where(al)
pad = 40; x0, x1 = max(0, xs.min() - pad), min(iw, xs.max() + pad); y0, y1 = max(0, ys.min() - pad), min(ih, ys.max() + pad)
crop = img[y0:y1, x0:x1]; ch, cw = crop.shape[:2]
W1 = 108.0; IMG_W = 108.0 * 0.64; IMG_H = IMG_W * ch / cw; IMG_X = (W1 - IMG_W) / 2; LEG_H = 14.0; H1 = IMG_H + LEG_H + 2
dpi1 = cw / (IMG_W * MM)
Image.fromarray((crop * 255).astype(np.uint8), "RGBA").save(RD / "1_conformation_render.png", dpi=(round(dpi1), round(dpi1)))
hp = pd.read_csv(RN / "representative_hub_projection.tsv", sep="\t")
hp["x"] = IMG_X + (hp.px - x0) / cw * IMG_W; hp["y"] = LEG_H + IMG_H - (hp.py - y0) / ch * IMG_H
CP = pd.read_csv(I / "pairs_cicero.tsv", sep="\t"); HE = CP[CP.set == "hub_edges"]

def panel1(mode):
    fig = fig_mm(W1, H1)
    ax = ax_mm(fig, 0, 0, W1, H1); ax.set_xlim(0, W1); ax.set_ylim(0, H1); ax.set_axis_off()
    if mode in ("full", "render"):
        ax.imshow(crop, extent=(IMG_X, IMG_X + IMG_W, LEG_H, LEG_H + IMG_H), interpolation="none", zorder=0)
    if mode in ("full", "labels"):
        # labels in two side columns, de-overlapped vertically (>= 3.2 mm apart), straight leader lines
        cx = IMG_X + IMG_W / 2
        for side, sub in ((-1, hp[hp.x < cx]), (1, hp[hp.x >= cx])):
            sub = sub.sort_values("y"); ys_ = list(sub.y.values)
            for k in range(1, len(ys_)): ys_[k] = max(ys_[k], ys_[k - 1] + 3.2)
            over = ys_[-1] - (LEG_H + IMG_H - 1) if ys_ else 0
            if over > 0: ys_ = [y - over for y in ys_]
            tx = IMG_X - 1.5 if side < 0 else IMG_X + IMG_W + 1.5
            for r, ty in zip(sub.itertuples(), ys_):
                ax.plot([r.x, tx], [r.y, ty], color="#444444", lw=0.4, zorder=3)
                ax.text(tx + 0.6 * side, ty, r.label, ha="left" if side > 0 else "right", va="center", fontsize=FS,
                        fontstyle="italic", zorder=4)
        ly = 9.0
        ax.plot([2, 7], [ly, ly], color=TUBE, lw=2.2, solid_capstyle="round"); ax.text(8, ly, "chromatin (2 kb beads)", va="center")
        ax.add_patch(Circle((4.5, ly - 4.2), 1.3, color=TFC, alpha=0.5, lw=0)); ax.text(8, ly - 4.2, "clustered TFs (DBSCAN)", va="center")
        ax.add_patch(Circle((44.5, ly), 1.3, color=CORE, lw=0)); ax.text(47, ly, "hub core", va="center")
        for k, c in enumerate(OKABE): ax.plot([42 + k * 1.6, 43.2 + k * 1.6], [ly - 4.2] * 2, color=c, lw=1.2)
        ax.text(49, ly - 4.2, "Cicero hub edge (clique colour)", va="center")
        ax.text(W1 - 1, H1 - 1, f"conformation {rep['conf'] + 1} of {int(row.n_conf)} ({row.eps_top:g} kT)\n"
                f"{rep['max_hub_in_strict_cluster']}/9 hub cores in one TF cluster", ha="right", va="top", fontsize=FSS)
    return fig
for mode, stem, tr in (("full", "1_conformation", True), ("render", "1_conformation_render_only", True), ("labels", "1_conformation_labels_only", True)):
    save(panel1(mode), stem, tr)
for stem in ("1_conformation", "1_conformation_render_only"):
    relink(stem, "../raster/1_conformation_render.png", IMG_X, LEG_H, IMG_W, IMG_H, H1)

# ---------------- 2: split matrix + P(s) ----------------
def scale(M):
    L = np.log10(np.where(M > 0, M, np.nan)); lo, hi = np.nanpercentile(L, [1, 99]); return np.clip((L - lo) / (hi - lo), 0, 1)
Hs, Ss = scale(np.where(np.outer(keep, keep), ice, np.nan)), scale(S)
SPL = np.full((nbin, nbin), np.nan); iu = np.triu_indices(nbin, 1); il = np.tril_indices(nbin, -1)
SPL[iu] = Hs[iu]; SPL[il] = Ss[il]
cmap = plt.get_cmap("Reds").copy(); cmap.set_bad("#E6E6E6")
W2, H2, MX, MY, MS = 62.0, 98.0, 11.0, 46.0, 47.0
npx = int(round(MS / 25.4 * 600))
Image.fromarray((cmap(np.ma.masked_invalid(SPL)) * 255).astype(np.uint8), "RGBA").convert("RGB").resize((npx, npx), Image.NEAREST).save(RD / "2_split_matrix_600dpi.png", dpi=(600, 600))
ps = pd.read_csv(A / "ps.tsv", sep="\t"); ps = ps[ps.eps_top == best]
def panel2():
    fig = fig_mm(W2, H2); ax = ax_mm(fig, MX, MY, MS, MS)
    ax.imshow(SPL, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", extent=(29.73, 31.55, 31.55, 29.73), rasterized=True)
    ax.plot([29.73, 31.55], [29.73, 31.55], color="white", lw=0.6)
    for s in ("top", "right"): ax.spines[s].set_visible(True)
    ax.set_xticks([30.0, 30.5, 31.0, 31.5]); ax.set_yticks([30.0, 30.5, 31.0, 31.5])
    ax.set_xlabel("chr16 (Mb)"); ax.set_ylabel("chr16 (Mb)")
    ax.text(31.5, 29.8, "Hi-C (ICE)", ha="right", va="top"); ax.text(29.78, 31.5, "simulation", ha="left", va="bottom")
    ax.set_title(f"10 kb; Spearman raw {row.spearman_raw:.2f}, O/E {row.spearman_oe:.2f}", fontsize=FS, pad=3)
    cb = ax_mm(fig, MX + MS + 1.5, MY, 1.5, MS)
    cb.imshow(np.linspace(1, 0, 256)[:, None], cmap=cmap, aspect="auto", extent=(0, 1, 0, 1), rasterized=True)
    cb.set_xticks([]); cb.yaxis.tick_right(); cb.set_yticks([0, 1]); cb.set_yticklabels(["low", "high"])
    for s in cb.spines.values(): s.set_visible(False)
    a2 = ax_mm(fig, MX, 12, MS, 24)
    s_kb = ps.s_kb.values; ok = s_kb <= 1500
    for col, c, lab in (("p_hic", "#B2182B", "Hi-C (ICE)"), ("p_sim", "#333333", "simulation")):
        v = ps[col].values; v = v / v[1]
        a2.loglog(s_kb[ok], v[ok], color=c, lw=0.8, label=lab)
    a2.set_xlabel("genomic distance s (kb)"); a2.set_ylabel("P(s) / P(20 kb)")
    a2.legend(frameon=False, loc="lower left", handlelength=1.2)
    return fig
save(panel2(), "2_split_matrix")
relink("2_split_matrix", "../raster/2_split_matrix_600dpi.png", MX, MY, MS, MS, H2)

# ---------------- 3: contact frequency vs nulls ----------------
sets = [("loops_peakachu07", "Peakachu 0.7\nloops"), ("hub_edges", "hub-component\nedges"), ("shared_edges", "shared clique\nedges"), ("cicero_step1", "Cicero\nstep-1 pairs")]
nulls = [("random_same_distance", "#9E9E9E", "vs random, same distance"), ("bound_matched_class_distance", "#0072B2", "vs TF-bound, matched distance + class")]
def panel3():
    fig = fig_mm(100, 62); ax = ax_mm(fig, 13, 15, 84, 40)
    for k, (st, _) in enumerate(sets):
        for q, (nl, c, _) in enumerate(nulls):
            d = VP[(VP.set == st) & (VP.null == nl)]
            if not len(d): continue
            pc = 1.0 / float(row.n_conf)
            y = np.clip(np.log2((d.obs.values + pc) / (d.exp.values + pc)), -6, 6)
            x = k + (q - 0.5) * 0.48 if st != "loops_peakachu07" else k
            jit = np.random.default_rng(k * 10 + q).uniform(-0.08, 0.08, len(y))
            ax.scatter(x + jit, y, s=2.5 if len(y) < 300 else 0.6, color=c, alpha=0.6 if len(y) < 300 else 0.25, lw=0, rasterized=len(y) >= 300)
            ax.plot([x - 0.14, x + 0.14], [np.median(y)] * 2, color="black", lw=1.0)
            sm = V[(V.set == st) & (V.null == nl)].iloc[0]
            pz = "p<0.001" if sm.p < 0.001 else f"p={sm.p:.3f}"
            ax.text(x, 6.3, f"{sm.agg_ratio:.2f}x\n{pz}", ha="center", va="bottom", fontsize=FSS)
    ax.axhline(0, color="#777777", lw=0.4, ls=(0, (2, 2)))
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([s[1] for s in sets], fontsize=FSS)
    ax.set_ylim(-6.3, 6.2); ax.set_ylabel("log2 (contact frequency /\nmatched expectation)")
    n = {st: int(V[(V.set == st)].n.iloc[0]) for st, _ in sets}
    for k, (st, _) in enumerate(sets): ax.text(k, -6.1, f"n={n[st]}", ha="center", va="bottom", fontsize=FSS)
    hs = [Line2D([], [], marker="o", ls="", color=c, ms=2.5) for _, c, _ in nulls]
    fig.legend(hs, [l for _, _, l in nulls], loc="lower center", ncol=2, frameon=False, fontsize=FSS, handletextpad=0.2, bbox_to_anchor=(0.55, 0.0))
    return fig
save(panel3(), "3_contacts")

# ---------------- 4: co-clustering ----------------
R1 = np.load(A / f"rosette_eps{best:g}.npz", allow_pickle=True); R2 = np.load(A / f"rosette_strict_eps{best:g}.npz", allow_pickle=True)
def panel4():
    fig = fig_mm(96, 58)
    a1 = ax_mm(fig, 11, 13, 32, 32)
    for g, (tag, nm) in enumerate((("", "DBSCAN 1.6\u03c3"), ("_strict", "strict 1.25\u03c3"))):
        d = VP[VP.set == f"hub_core_pairs_cocluster{tag}"]; x0_ = g * 2.4
        for r in d.itertuples(): a1.plot([x0_, x0_ + 1], [r.exp, r.obs], color="#BBBBBB", lw=0.4)
        a1.scatter(np.full(len(d), x0_), d.exp, s=4, color="#0072B2", lw=0, zorder=3)
        a1.scatter(np.full(len(d), x0_ + 1), d.obs, s=4, color=CORE, lw=0, zorder=3)
        sm = V[V.set == f"hub_core_pairs_cocluster{tag}"].iloc[0]
        a1.text(x0_ + 0.5, 1.04, f"{nm}\n{sm.agg_ratio:.2f}x, " + ("p<0.005" if sm.p < 0.005 else f"p={sm.p:.2f}"),
                ha="center", va="bottom", fontsize=FSS)
    a1.set_xticks([0, 1, 2.4, 3.4]); a1.set_xticklabels(["matched", "hub", "matched", "hub"], fontsize=FSS)
    a1.set_xlim(-0.5, 3.9); a1.set_ylim(0, 1.0); a1.set_ylabel("P(pair in same TF cluster)")
    a1.text(1.7, -0.17, "36 hub-core pairs vs matched bound pairs", ha="center", va="top", fontsize=FSS)
    a2 = ax_mm(fig, 61, 13, 32, 32)
    mh, nm_ = R2["mh"], R2["null_mean"]; k = np.arange(10)
    a2.bar(k, np.bincount(mh, minlength=10)[:10] / len(mh), color=CORE, width=0.7)
    a2.axvline(nm_.mean(), color="#0072B2", lw=0.8, ls=(0, (2, 1.5))); a2.axvline(mh.mean(), color=CORE, lw=0.8)
    a2.set_xticks(k); a2.set_xlabel("hub cores in one strict\nTF cluster (max)"); a2.set_ylabel("fraction of conformations")
    sm2 = V[V.set == "hub_cores_max_in_one_cluster_strict"].iloc[0]; sm1 = V[V.set == "hub_cores_max_in_one_cluster"].iloc[0]
    a2.text(4.5, 1.03, f"mean {mh.mean():.2f} vs {nm_.mean():.2f} random sets\n"
            f">=3 in one cluster: {sm2.frac_conf_ge3 * 100:.0f}% (random {sm2.null_frac_conf_ge3 * 100:.0f}%)\n"
            f"DBSCAN 1.6\u03c3: {sm1.mean_obs:.1f} vs {sm1.mean_exp:.1f} of 9", ha="center", va="bottom", fontsize=FSS, transform=a2.get_xaxis_transform())
    return fig
save(panel4(), "4_cocluster")

# ---------------- 5: gallery ----------------
gl = sorted(RN.glob("gallery_*.png"))
if gl:
    Z = np.load(A / "representative.npz"); W5 = 108.0; cell = W5 / 3
    fig = fig_mm(W5, cell * 2 + 4); ax = ax_mm(fig, 0, 0, W5, cell * 2 + 4); ax.set_xlim(0, W5); ax.set_ylim(0, cell * 2 + 4); ax.set_axis_off()
    for g, f in enumerate(gl):
        im = np.asarray(Image.open(f).convert("RGBA")); r_, c_ = divmod(g, 3)
        yy, xx = np.where(im[..., 3] > 5); hw_ = max(np.ptp(yy), np.ptp(xx)) // 2 + 10; cy_, cx_ = (yy.min() + yy.max()) // 2, (xx.min() + xx.max()) // 2
        im = im[max(0, cy_ - hw_):cy_ + hw_, max(0, cx_ - hw_):cx_ + hw_]
        Image.fromarray(im).save(RD / f"5_gallery_{g + 1}.png", dpi=(round(im.shape[1] / (cell * MM)),) * 2)
        x, y = c_ * cell, (1 - r_) * (cell + 2)
        ax.imshow(im, extent=(x, x + cell, y + 2, y + 2 + cell - 2), interpolation="none")
        ax.text(x + cell / 2, y + 1.0, f"conf {int(Z['gallery'][g]) + 1}: {int(Z['mh_g'][g])}/9 in one cluster", ha="center", va="center", fontsize=FSS)
    save(fig, "5_gallery", True)
pd.DataFrame([("1_conformation (3 versions)", W1, round(H1, 1), f"render {cw}x{ch} px = {IMG_W:.1f} mm @ {dpi1:.0f} dpi"),
              ("2_split_matrix", W2, H2, f"heatmap {npx} px = {MS} mm @ 600 dpi"), ("3_contacts", 100, 62, ""), ("4_cocluster", 96, 58, ""),
              ("5_gallery", 108, round(108 / 3 * 2 + 4, 1), "")], columns=["file", "width_mm", "height_mm", "raster"]).to_csv(F / "sizes_mm.tsv", sep="\t", index=False)
print(f"render {cw}x{ch} px over {IMG_W:.1f} mm = {dpi1:.0f} dpi")
