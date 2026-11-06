"""SE hub, data-driven figure (no polymer model). a) virtual 4C from the SE core bin, WT Hi-C (5 and 10 kb), O/E = ICE / mean ICE at
the same distance over the window, with aligned genes, TF-count bars and Cicero loop-edge arcs; b) star schematic: SE hexagon in the
centre, the 12 linked elements around it (TF pies, v2 colours), red spokes with width ~ Hi-C O/E (10 kb, chr19-wide expected from
05_hic_test.py), span labels, no element-element links; c) direct Hi-C test numbers vs nulls."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge, RegularPolygon, Rectangle
Bd = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/se_hub"); O0 = Bd / "figure_v2"; O = Bd / "figure_v2_cand"; I = Bd / "model/inputs"; T_ = Bd / "hic_test"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "axes.spines.top": False, "axes.spines.right": False})
TFS = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"]
COL = dict(zip(TFS, ["#C48311", "#32A190", "#D9B637", "#C681F1", "#739B64", "#8C486E", "#4DC7E5", "#8AC975", "#65278F", "#712F55"]))
LRED = "#E6A79F"; OTHER, INK, RED, GREY, SEC = "#BDBDBD", "#222222", "#C0392B", "#9E9E9E", "#4D4D4D"
L0, L1 = 430000, 2250000
B = pd.read_csv(I / "beads.tsv", sep="\t").fillna({"tfs": ""}); EL = pd.read_csv(I / "hub_elements.tsv", sep="\t")
SEr = pd.read_csv(Bd / "step1/HEK293_SE_rose_hg38.tsv", sep="\t"); se = SEr[SEr.se_id == "HEK293_SE_0027"].iloc[0]
G = pd.read_csv(O0 / "genes.tsv", sep="\t"); R10 = json.load(open(T_ / "results_10kb.json")); R5 = json.load(open(T_ / "results_5kb.json"))
EO = pd.read_csv(T_ / "edges_oe_10kb.tsv", sep="\t"); EO = EO[EO.set == "loop_edges_18"]
core = EL[EL.is_core].iloc[0]; part = EL[~EL.is_core].reset_index(drop=True)
mid = lambda r: (int(r.split(":")[1].split("-")[0]) + int(r.split(":")[1].split("-")[1])) // 2
cpos = mid(core.element); part["pos"] = [mid(r) for r in part.element]; part["span_kb"] = (part.pos - cpos).abs() / 1000
def oe_at(p):
    i, j = sorted(((cpos) // 10000, p // 10000)); m = EO[(EO.i == i) & (EO.j == j)]; return float(m.oe.iloc[0]) if len(m) else np.nan
part["oe"] = [oe_at(p) for p in part.pos]; part["cls"] = ["element" if re.search(r" [+-]\d+ kb$", l) else "promoter" for l in part.label]  # promoter = TSS within 2.5 kb (03_prep_model.py labels)
part.to_csv(O / "spokes.tsv", sep="\t", index=False)
def tfset(b): return set(x for x in B.tfs[b].split(",") if x)
def pie(ax, x, y, b, r, diamond=False):
    clip = None
    if diamond: clip = RegularPolygon((x, y), 4, radius=r * 1.35, fc="none", ec="none", zorder=3); ax.add_patch(clip); r = r * 1.4
    tf = tfset(b); tl = [t for t in TFS if t in tf]; no = len(tf - set(TFS)); tot = max(1, len(tl) + no); st = 90.0
    for t in tl + (["other"] if no else []):
        w = 360.0 * (no if t == "other" else 1) / tot; wd = Wedge((x, y), r, st, st + w, color=COL.get(t, OTHER), lw=0, zorder=4); ax.add_patch(wd); wd.set_clip_path(clip) if clip is not None else None; st += w
def v4c(res):
    z = np.load(O / f"hic_{res // 1000}kb.npz"); M = z["ice"]; n = M.shape[0]
    exp = np.array([np.nanmean(np.diagonal(M, d)) for d in range(n)])
    v = (cpos - L0) // res; prof = M[v]; d = np.abs(np.arange(n) - v)
    with np.errstate(invalid="ignore", divide="ignore"): o = prof / exp[d]
    o[d < 2] = np.nan; return L0 + (np.arange(n) + 0.5) * res, prof, exp[d], o
fig = plt.figure(figsize=(330 / 25.4, 104 / 25.4))
# ---- a: virtual 4C (10 kb; 5 kb is too sparse at >100 kb: most bins 0-2 reads)
z = np.load(O0 / "hic_10kb.npz"); M, Rw = z["ice"], z["raw"]; nn = M.shape[0]; cov = Rw.sum(1) - np.diag(Rw)
ok = np.isfinite(np.diag(M) * 0 + M.sum(1)) & (cov >= 0.2 * np.median(cov[cov > 0]))
Mk = np.where(ok[:, None] & ok[None, :], M, np.nan); exp = np.array([np.nanmean(np.diagonal(Mk, d)) for d in range(nn)])
v = (cpos - L0) // 10000; dd = np.abs(np.arange(nn) - v); o = Mk[v] / exp[dd]; o[dd < 2] = np.nan
sm = pd.Series(o).rolling(3, center=True, min_periods=2).mean().values; xb = (L0 + (np.arange(nn) + 0.5) * 10000) / 1e6
pd.DataFrame(dict(bin_start=L0 + np.arange(nn) * 10000, obs_ice=Mk[v], exp_ice=exp[dd], oe=o, oe_30kb=sm)).to_csv(O / "virtual4c_10kb.tsv", sep="\t", index=False)
ax = fig.add_axes([0.05, 0.58, 0.43, 0.3])
ax.bar(xb, np.nan_to_num(o), width=0.0095, color="#CFCFCF", lw=0, label="O/E, 10 kb bin")
ax.plot(xb, sm, color=INK, lw=0.8, label="30 kb running mean")
ax.axhline(1, color="#777777", lw=0.5, ls=(0, (2, 1.5))); ax.axvspan(se.start / 1e6, se.end / 1e6, color="#F1E3E1", zorder=0, lw=0)
ax.text((se.start + se.end) / 2e6, 5.55, "SE cand.", ha="center", fontsize=6.5, fontweight="bold", color=SEC)
for r in part.itertuples():
    ax.plot([r.pos / 1e6] * 2, [-0.45, -0.1], color=RED, lw=1.2, clip_on=False)
    ax.plot(r.pos / 1e6, min(r.oe, 5.8), "o" if r.cls == "promoter" else "D", ms=2.6 if r.cls == "promoter" else 2.3, mfc=RED, mec="white", mew=0.3, zorder=5)
ax.plot([], [], "o", ms=2.6, mfc=RED, mec="white", mew=0.3, label="loop-edge pixel: promoter"); ax.plot([], [], "D", ms=2.3, mfc=RED, mec="white", mew=0.3, label="element")
ax.set_xlim(L0 / 1e6, L1 / 1e6); ax.set_ylim(0, 6); ax.set_ylabel("Hi-C O/E\n(viewpoint = SE-candidate core)"); ax.set_xticklabels([])
ax.legend(frameon=False, fontsize=5.3, loc="lower left", ncol=4, bbox_to_anchor=(0.0, 0.98), handlelength=1.2, columnspacing=1)
ax.set_title("a  Virtual 4C from the super-enhancer candidate (WT HEK293T Hi-C, 10 kb)", loc="left", fontsize=7, pad=13)
# arcs
ax2 = fig.add_axes([0.05, 0.46, 0.43, 0.11], sharex=ax); ax2.set_axis_off(); ax2.set_ylim(0, 1.1)
for r in part.itertuples():
    a_, b_ = sorted((cpos / 1e6, r.pos / 1e6)); xs = np.linspace(a_, b_, 60); h = min(1.0, 0.25 + 0.75 * (b_ - a_) / 0.65)
    ax2.plot(xs, h * np.sin(np.pi * (xs - a_) / (b_ - a_)), color=RED, lw=0.8, ls="-" if r.cls == "promoter" else (0, (1, 1.3)))
ax2.text(L0 / 1e6, 0.95, "12 Cicero loop-edges from the SE candidate\n(on a Peakachu 0.7 loop;\nno Cicero-only links)", fontsize=5.3, color=RED, va="top", ha="left")
# TF bars
ax3 = fig.add_axes([0.05, 0.36, 0.43, 0.09], sharex=ax)
ax3.bar((B.start + 1000) / 1e6, B.n_tfs, width=0.0025, color=GREY, lw=0); ax3.set_ylabel("TFs /\n2 kb", fontsize=5.8); ax3.set_xticklabels([]); ax3.set_ylim(0, 40); ax3.set_yticks([0, 20, 40])
# genes + partner labels
ax4 = fig.add_axes([0.05, 0.11, 0.43, 0.24], sharex=ax); ax4.set_ylim(4.2, -0.6); ax4.set_yticks([]); ax4.spines["left"].set_visible(False)
row_end = [-1e9] * 2
for g in G.sort_values("start").itertuples():
    rw = 0 if g.start > row_end[0] + 5000 else 1; row_end[rw] = max(row_end[rw], g.end)
    ax4.plot([g.start / 1e6, g.end / 1e6], [rw * 0.5, rw * 0.5], color="#8A8A8A", lw=1.6, solid_capstyle="butt")
Ps = part.sort_values("pos"); px = (Ps.pos.values / 1e6).copy(); lxs = px.copy(); gap = 0.045
for _ in range(200):
    for k in range(1, len(lxs)): lxs[k] = max(lxs[k], lxs[k - 1] + gap)
    lxs -= (lxs - px).mean(); lxs = np.clip(lxs, L0 / 1e6 + 0.05, L1 / 1e6 - 0.05)
for k, (r, lx0) in enumerate(zip(Ps.itertuples(), lxs)):
    ax4.plot([r.pos / 1e6, r.pos / 1e6, lx0], [-0.3, 1.1, 2.0], color=RED, lw=0.4)
    ax4.plot(lx0, 2.75, "o" if r.cls == "promoter" else "D", ms=7.5 if r.cls == "promoter" else 6.8, mfc=RED, mec="none", zorder=4)
    ax4.text(lx0, 2.75, str(k + 1), fontsize=4.8, ha="center", va="center", color="white", fontweight="bold", zorder=5)
ax4.set_xlabel("chr19 (Mb)", labelpad=1); ax4.text(L0 / 1e6, -0.25, "genes", fontsize=5.3, color="#666666", va="center", ha="right")
# ---- b: star
axs = fig.add_axes([0.49, 0.0, 0.3, 0.93]); axs.set_xlim(-45, 45); axs.set_ylim(-36, 32); axs.set_aspect("equal"); axs.set_axis_off()
n = len(part); P = part.sort_values("pos").reset_index(drop=True); Rr = 19; P["num"] = np.arange(1, n + 1)
for i, r in P.iterrows():
    th = np.pi / 2 - 2 * np.pi * i / n; x, y = Rr * np.cos(th), Rr * np.sin(th)
    axs.plot([0, x], [0, y], color=RED if r.oe >= 1 else LRED, lw=max(0.7, 1.1 * r.oe), zorder=1, dash_capstyle="butt", solid_capstyle="round", ls="-" if r.cls == "promoter" else (0, (1, 1)))
    axs.text(0.55 * x, 0.55 * y, f"{r.span_kb:.0f} kb", fontsize=4.6, ha="center", va="center", zorder=3, bbox=dict(fc="white", ec="none", pad=0.3))
    pie(axs, x, y, r.bead, 2.2, diamond=r.cls == "element"); axs.add_patch(Circle((x, y), 0.85, color=INK, lw=0, zorder=5))
    lx_, ly_ = 1.15 * x + (3.2 if x > 1 else -3.2 if x < -1 else 0), 1.15 * y + (2.6 if y > 15 else -2.6 if y < -15 else 0)
    axs.text(lx_, ly_, f"{r.num}  {r.label}", fontsize=5, ha="left" if x > 1 else "right" if x < -1 else "center", va="center", fontstyle="italic")
pie(axs, 0, 0, int(core.bead), 6.3)
axs.add_patch(RegularPolygon((0, 0), 6, radius=4.6, fc="#F4F4F4", ec=SEC, lw=1.3, zorder=5)); axs.text(0, 1.1, "SE\ncand.", ha="center", va="center", fontsize=5.8, linespacing=0.95, fontweight="bold", color=SEC, zorder=6)
axs.text(0, -2.1, f"{len(tfset(int(core.bead)))} TFs", ha="center", va="center", fontsize=4.8, zorder=6)
axs.set_title("b  SE candidate → 6 promoters + 6 other elements (width ~ Hi-C O/E)", loc="left", fontsize=7)
axs.text(0, -32.5, "Spoke = O/E of the SE candidate\u2013partner 10 kb pixel; label = genomic span.\nNo partner\u2013partner links: the 12 partners do not contact each other\nbeyond matched sets (module 1.08×, p = 0.27; n.s.)", ha="center", va="center", fontsize=5.3)
# ---- c: test numbers
axc = fig.add_axes([0.875, 0.68, 0.11, 0.22])
vals = [("SE cand.–partner\nloop-edges", R10["edges_loop_edges_18"]["ratio_tf"], R10["edges_loop_edges_18"]["p_tf"], RED),
        ("SE cand. + partners\nas a module", R10["hub_all_tf"]["ratio"], R10["hub_all_tf"]["p"], INK),
        ("partner–partner\npairs only", R10["hub_nonedge_tf"]["ratio"], R10["hub_nonedge_tf"]["p"], GREY)]
for k, (nm, v, p, c) in enumerate(vals):
    axc.barh(k, v, color=c, height=0.6); axc.text(v + 0.04, k, f"{v:.2f}×, p = {p:.3f}" if p >= 0.001 else f"{v:.2f}×, p < 0.001", va="center", fontsize=5.5)
axc.axvline(1, color="#777777", lw=0.6, ls=(0, (2, 1.5))); axc.set_yticks(range(3)); axc.set_yticklabels([v[0] for v in vals], fontsize=5.5); axc.invert_yaxis()
axc.set_xlim(0, 2.6); axc.set_xlabel("O/E vs distance- and TF-matched\nchr19 null (10 kb; 1,000 draws)", fontsize=5.5)
axc.text(-0.62, 1.05, "c  Direct Hi-C test", transform=axc.transAxes, fontsize=7)
# legend
lx = 0.815
fig.text(lx, 0.5, "TF with a MACS2 peak", fontsize=6, fontweight="bold")
for k, (c, l) in enumerate([(COL[t], t) for t in TFS] + [(OTHER, "other TFs")]):
    xx, yy = lx + (k // 4) * 0.06, 0.46 - (k % 4) * 0.042; fig.text(xx, yy, "\u25CF", color=c, fontsize=7, va="center")
    fig.text(xx + 0.009, yy, l, fontsize=5.5, va="center", fontstyle="italic" if l in TFS else "normal")
fig.text(lx, 0.255, "Partner class and edge", fontsize=6, fontweight="bold")
LG = [("o", RED, "-", "promoter (TSS within 2.5 kb), solid edge"), ("D", RED, (0, (1, 1)), "other element, dotted edge"), (None, LRED, "-", "light thin red: Hi-C O/E < 1")]
for k, (mk, c, ls, l) in enumerate(LG):
    yy = 0.215 - k * 0.042; fig.add_artist(Line2D([lx, lx + 0.018], [yy, yy], transform=fig.transFigure, color=c, lw=1.2, ls=ls))
    if mk: fig.add_artist(Line2D([lx + 0.009], [yy], transform=fig.transFigure, marker=mk, ms=4, mfc=c, mec="white", mew=0.3, ls="none"))
    fig.text(lx + 0.024, yy, l, fontsize=5.5, va="center")
for e in ("svg", "pdf"): fig.savefig(O / f"se_data.{e}", transparent=True)
fig.savefig(O / "se_data.png", dpi=600, facecolor="white"); fig.savefig(O / "se_data_preview.png", dpi=250, facecolor="white")
print(part.groupby("cls").oe.agg(["mean", "median", "count"]).round(2)); print(part[["label", "cls", "span_kb", "oe"]].round(2).to_string(index=False))
