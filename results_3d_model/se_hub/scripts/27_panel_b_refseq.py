"""Slide 16 panel b: star with TF pies, equal-width lines coloured by the partner dot test (20_partner_dot_test.py): solid red = suggestive dot, dashed faint red = no clear dot, dotted grey = too few reads to test. Drawn at its exact frame size."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge, RegularPolygon
Bd = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/se_hub"); O = Bd / "figure_v7_refseq"; O6 = Bd / "figure_v6_hubstyle"; V = Bd / "figure_v2_cand"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 1, "xtick.major.width": 1, "ytick.major.width": 1, "xtick.major.size": 4, "ytick.major.size": 4, "axes.spines.top": False, "axes.spines.right": False})
TFS = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"]; COL = dict(zip(TFS, ["#C48311", "#32A190", "#D9B637", "#C681F1", "#739B64", "#8C486E", "#4DC7E5", "#8AC975", "#65278F", "#712F55"]))
OTHER, INK, RED, LRED, SEC = "#BDBDBD", "#222222", "#C0392B", "#E6A79F", "#4D4D4D"; DOT = (0, (1, 1)); L0, L1 = 430000, 2250000; SE0, SE1 = 1203658, 1267059
S = pd.read_csv(V / "spokes.tsv", sep="\t").sort_values("pos").reset_index(drop=True); C = pd.read_csv(V / "virtual4c_10kb.tsv", sep="\t"); G = pd.read_csv(Bd / "figure_v2/genes.tsv", sep="\t"); B = pd.read_csv(Bd / "model/inputs/beads.tsv", sep="\t").fillna({"tfs": ""}); EL = pd.read_csv(Bd / "model/inputs/hub_elements.tsv", sep="\t"); core = int(EL.bead[EL.is_core].iloc[0]); Dt = pd.read_csv(O6 / "partner_dots.tsv", sep="\t").set_index("label"); T_SE, T_P, T_E = "#C4650A", "#0F7C72", "#666666"
# ---- b
Wb, Hb = 5.55, 5.75; fig = plt.figure(figsize=(Wb, Hb)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, Wb); ax.set_ylim(0, Hb); ax.set_axis_off()
ax.text(0.08, Hb - 0.2, "Partners contact the SE candidate, not each other", fontsize=14, fontweight="bold", va="center")
def tfset(b): return set(x for x in B.tfs[b].split(",") if x)
def pie(x, y, b, r, dia=False):
    tf = tfset(b); tl = [t for t in TFS if t in tf]; no = len(tf - set(TFS)); tot = max(1, len(tl) + no); st = 90.0; clip = None
    if dia: clip = RegularPolygon((x, y), 4, radius=r * 1.32, fc="none", ec="none", zorder=3); ax.add_patch(clip); r *= 1.4
    for t in tl + (["other"] if no else []):
        w = 360.0 * (no if t == "other" else 1) / tot; p_ = Wedge((x, y), r, st, st + w, color=COL.get(t, OTHER), lw=0, zorder=4); ax.add_patch(p_); p_.set_clip_path(clip) if clip is not None else None; st += w
RS = pd.read_csv(Bd / "figure_v7_refseq/partners_refseq.tsv", sep="\t").set_index("pos"); S["old_label"] = S.label; S["label"] = [RS.label[p].replace("RNU6-2/TMEM259", "TMEM259") for p in S.pos]; S["cls"] = [RS.cls[p] for p in S.pos];
cx, cy, R = 2.66, 3.72, 1.15; n = len(S)
for i, r in S.iterrows():
    th = np.pi / 2 - 2 * np.pi * i / n; x, y = cx + R * np.cos(th), cy + R * np.sin(th); e = r.cls == "element"
    q = Dt.loc[r.old_label]; st = "real" if q.real_dot else ("few" if q.best_reads < 30 or not np.isfinite(q.best_don) else "no"); ax.plot([cx, x], [cy, y], color={"real": RED, "no": LRED, "few": "#AAAAAA"}[st], lw=2.2, zorder=1, ls={"real": "-", "no": (0, (4, 2)), "few": (0, (1, 1.5))}[st], solid_capstyle="round", dash_capstyle="butt")
    pie(x, y, r.bead, 0.17, e); ax.add_patch(RegularPolygon((x, y), 4, radius=0.17 * 1.32, fc="none", ec=T_E, lw=1.8, zorder=6) if e else Circle((x, y), 0.17, fc="none", ec=T_P, lw=1.8, zorder=6))
    ax.text(x + 0.27 * np.cos(th) + (0.03 if np.cos(th) > 0.1 else -0.03 if np.cos(th) < -0.1 else 0), y + (0.27 + 0.09 * (abs(np.cos(th)) < 0.1)) * np.sin(th), r.label, fontsize=12, fontstyle="italic", color=T_E if e else T_P, ha="left" if np.cos(th) > 0.1 else "right" if np.cos(th) < -0.1 else "center", va="center")
pie(cx, cy, core, 0.5); ax.add_patch(RegularPolygon((cx, cy), 6, radius=0.38, fc="#FDF1E2", ec=T_SE, lw=2, zorder=5)); ax.text(cx, cy, "SE\ncand.", ha="center", va="center", fontsize=12, fontweight="bold", color=T_SE, linespacing=0.9, zorder=6)
ax.text(Wb / 2, 1.83, "Partners contact the SE candidate 1.76× more than matched pairs\n(p = 0.004); partner–partner 0.92× (n.s.)", ha="center", va="center", fontsize=12, fontweight="bold", color=RED, linespacing=1.1)
y = 1.38; ax.plot([0.12, 0.46], [y, y], color=RED, lw=2.2); ax.text(0.54, y, "suggestive dot", va="center"); ax.plot([1.72, 2.06], [y, y], color=LRED, lw=2.2, ls=(0, (4, 2))); ax.text(2.14, y, "no clear dot", va="center"); ax.plot([3.2, 3.54], [y, y], color="#AAAAAA", lw=2.2, ls=(0, (1, 1.5))); ax.text(3.62, y, "too few reads", va="center")
y = 1.12; ax.plot(0.2, y, "o", ms=9, mfc="none", mec=T_P, mew=1.8); ax.text(0.32, y, "promoter", va="center", color=T_P); ax.plot(1.22, y, "D", ms=8, mfc="none", mec=T_E, mew=1.8); ax.text(1.34, y, "other element", va="center", color=T_E); ax.add_patch(RegularPolygon((2.62, y), 6, radius=0.09, fc="#FDF1E2", ec=T_SE, lw=1.8)); ax.text(2.74, y, "SE candidate", va="center", color=T_SE); ax.add_patch(Wedge((4.05, y), 0.1, 90, 270, color=COL["SOX4"], lw=0)); ax.add_patch(Wedge((4.05, y), 0.1, 270, 90, color=COL["NFYA"], lw=0)); ax.text(4.2, y, "pie = TFs bound", va="center")
items = [(COL[t], t) for t in TFS] + [(OTHER, "other")]
for k, (c_, l) in enumerate(items):
    col, row = k % 4, k // 4; x_, y_ = 0.2 + col * 1.33, 0.78 - row * 0.26; ax.add_patch(Circle((x_, y_), 0.075, color=c_, lw=0)); ax.text(x_ + 0.13, y_, l, va="center", fontstyle="italic" if l in TFS else "normal")
for e in ("svg", "pdf"): fig.savefig(O / f"se_panel_b_refseq.{e}", dpi=600, transparent=True)
fig.savefig(O / "se_panel_b_refseq.png", dpi=600, facecolor="white"); fig.savefig(O / "se_panel_b_refseq_check.png", dpi=144, facecolor="white")
