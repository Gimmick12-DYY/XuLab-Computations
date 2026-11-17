# Fig 4c draft (2026-11-17, user: "draw it"): shared enriched motifs (exploratory) for the 3 chosen example pairs (DDIT3-RBBP7, ADNP-MTF2, DR1-NFYA).
# Per pair up to 3 specific shared motifs from motif_tables/pair_shared_motifs_p005.tsv (HOMER first, then DB); each panel = logo (information
# content) + -log10 p in each TF on discovery (odd chr) and held-out (even + X) halves, dashed line at p = 0.05. Palette C. Output fig4/fig4c_motifs.{svg,png}
import sys, re, math, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath; from matplotlib.patches import PathPatch; from matplotlib.transforms import Affine2D; from matplotlib.font_manager import FontProperties
from scipy.stats import norm
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; OUT = f"{O}/fig4"; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{X}/figures_paletteC"); import regulome_palette as rp; Pc = rp.use()
PAIRS = os.environ.get("PAIRS", "DDIT3-RBBP7,ADNP-MTF2,DR1-NFYA").split(","); TOPN = int(os.environ.get("TOPN", "3")); TAG = os.environ.get("TAG", ""); MODE = os.environ.get("MODE", "bars")
T = pd.read_csv(f"{O}/motif_tables/pair_shared_motifs_p005.tsv", sep="\t").set_index("pair")
hom = {}; cur = None; rows = []
for l in list(open("/nas/longleaf/rhel9/apps/homer/5.1/data/knownTFs/vertebrates/known.motifs")) + [">end\tend\t0"]:
    if l.startswith(">"):
        if cur: hom[cur] = np.array(rows)
        cur = l.split("\t")[1].split("/")[0]; rows = []
    elif l.strip(): rows.append([float(x) for x in l.split()])
db, alt, cur, rows = {}, {}, None, []
for l in list(open(f"{O}/db/known.meme")) + ["MOTIF __end__"]:
    if l.startswith("MOTIF "):
        if cur and rows: db[cur] = np.array(rows)
        x = l.split(); cur = x[1]; alt[cur] = x[2] if len(x) > 2 else x[1]; rows = []
    elif cur and len(l.split()) == 4 and l.strip()[0] in "0123456789.": rows.append([float(v) for v in l.split()])
name2id = {}
for k, v in alt.items(): name2id.setdefault(v, k); name2id.setdefault(k, k)
H = {}
def homer_lp(t, pt, m):
    if (t, pt) not in H:
        k = pd.read_csv(f"{O}/recall/homer/raw_lab/{t}/{pt}/knownResults.txt", sep="\t"); s = pd.Series((-k.iloc[:, 3] / math.log(10)).values, index=k.iloc[:, 0].str.split("/").str[0]); H[(t, pt)] = s[~s.index.duplicated()]
    return float(H[(t, pt)].get(m, np.nan))
Fd = pd.read_csv(f"{O}/motif_valid/scores_fisher_disc.tsv", sep="\t", index_col=0); Fv = pd.read_csv(f"{O}/motif_valid/scores_fisher_val.tsv", sep="\t", index_col=0)
def db_lp(t, pt, mid): z = (Fd if pt == "disc" else Fv).at[mid, t]; return float(-np.log10(max(norm.sf(z), 1e-300)))
COL = {"A": "#2C7A6B", "C": "#2166AC", "G": "#D98C1F", "T": "#B2182B"}; FP = FontProperties(family="DejaVu Sans", weight="bold")
def logo(ax, M):
    M = np.asarray(M, float); M = M / M.sum(1, keepdims=True); ic = 2 + (M * np.log2(np.maximum(M, 1e-9))).sum(1)
    for i, (row, h) in enumerate(zip(M, ic)):
        y = 0
        for k in np.argsort(row):
            hh = row[k] * h
            if hh < .01: continue
            tp = TextPath((0, 0), "ACGT"[k], size=1, prop=FP); bb = tp.get_extents(); tr = Affine2D().translate(-bb.x0, -bb.y0).scale(.9 / bb.width, hh / bb.height).translate(i + .05, y)
            ax.add_patch(PathPatch(tr.transform_path(tp), fc=COL["ACGT"[k]], lw=0)); y += hh
    ax.set_xlim(0, len(M)); ax.set_ylim(0, 2); ax.set_xticks([]); ax.set_yticks([0, 2]); ax.set_ylabel("bits", fontsize=5); ax.spines["bottom"].set_visible(False)
sel = {}
for p in PAIRS:
    r = T.loc[p]; h = [x.strip() for x in str(r.shared_homer_specific).split(";") if x.strip() and x.strip() != "nan"]; d = [x.strip() for x in str(r.shared_db_specific).split(";") if x.strip() and x.strip() != "nan"]
    sel[p] = [("H", m) for m in h if m in hom][:3]; sel[p] += [("D", m) for m in d if name2id.get(m) in db][:3 - len(sel[p])]
def worst(p, src, m):
    a_, b_ = p.split("-")
    if src == "H": return max(homer_lp(t, pt, m) * -1 for t in (a_, b_) for pt in ("disc", "val"))
    return max(db_lp(t, pt, name2id[m]) * -1 for t in (a_, b_) for pt in ("disc", "val"))
for p in PAIRS: sel[p] = sorted(sel[p], key=lambda x: worst(p, *x))[:TOPN]
W, Hh = (7.1, 5.4) if TOPN == 3 else ((4.2, 3.0) if MODE == "logo" else (4.7, 4.4))
fig = plt.figure(figsize=(W, Hh)); gs = fig.add_gridspec(3, TOPN, hspace=(.75 if MODE == "logo" else .95), wspace=.45, left=.17 if TOPN < 3 else .14, right=.98, top=(.86 if MODE == "logo" else .9), bottom=.07); rows_out = []
for i, p in enumerate(PAIRS):
    a, b = p.split("-"); r = T.loc[p]
    if MODE != "logo": fig.text(.035, .9 - i * (.83 / 3) - .13, f"{p}\ncorr pct {r.correlation_pct:.2f} | AF3 {r.af3}", fontsize=6, rotation=90, va="center", ha="center", linespacing=1.6)
    for j in range(TOPN):
        if MODE == "logo":
            ax1 = fig.add_subplot(gs[i, j]); ax2 = None
            if j == 0: ax1.text(-.38, .45, f"{p}\ncorr pct {r.correlation_pct:.2f}\nAF3 {r.af3}", transform=ax1.transAxes, fontsize=6, rotation=90, va="center", ha="center", linespacing=1.4)
            if j >= len(sel[p]): ax1.axis("off"); continue
        else:
            sub = gs[i, j].subgridspec(2, 1, height_ratios=[1.1, 1], hspace=.35); ax1 = fig.add_subplot(sub[0]); ax2 = fig.add_subplot(sub[1])
            if j >= len(sel[p]): ax1.axis("off"); ax2.axis("off"); continue
        src, m = sel[p][j]
        if src == "H": M = hom[m]; lp = {(t, pt): homer_lp(t, pt, m) for t in (a, b) for pt in ("disc", "val")}; label = m
        else: mid = name2id[m]; M = db[mid]; lp = {(t, pt): db_lp(t, pt, mid) for t in (a, b) for pt in ("disc", "val")}; label = alt.get(mid, m).split(".")[0]
        logo(ax1, M); ax1.set_title(f"{label}  ({'HOMER' if src == 'H' else 'JASPAR/HOCOMOCO/Codebook'})", fontsize=5.5)
        xs = np.arange(4); vals = [lp[(a, "disc")], lp[(a, "val")], lp[(b, "disc")], lp[(b, "val")]]
        if MODE == "logo":
            pw = 10 ** -min(vals); ax1.text(.5, -.16, f"{a} and {b}: all 4 tests p ≤ {pw:.2g}", transform=ax1.transAxes, ha="center", va="top", fontsize=5.5)
            rows_out.append(dict(pair=p, motif=label, source="HOMER" if src == "H" else "DB", max_p_of_4=round(pw, 4), **{f"{t}_{pt}_p": round(10 ** -lp[(t, pt)], 4) for t in (a, b) for pt in ("disc", "val")})); continue
        ax2.bar(xs, vals, color=[Pc.neutral.dark, Pc.neutral.dark, Pc.neutral.light, Pc.neutral.light], edgecolor=Pc.neutral.ink, linewidth=.3, alpha=1, width=.7)
        for x_, v in zip(xs, vals): ax2.text(x_, v, f"{10 ** -v:.1g}", ha="center", va="bottom", fontsize=4.5)
        ax2.axhline(-math.log10(.05), color=Pc.neutral.mid, lw=.6, ls="--"); ax2.set_xticks(xs); ax2.set_xticklabels([f"{a}\ndisc", f"{a}\nheld-out", f"{b}\ndisc", f"{b}\nheld-out"], fontsize=4.5)
        ax2.set_ylabel("-log10 p", fontsize=5); ax2.set_ylim(0, max(vals) * 1.35 + .2)
        rows_out.append(dict(pair=p, motif=label, source="HOMER" if src == "H" else "DB", **{f"{t}_{pt}_p": round(10 ** -lp[(t, pt)], 4) for t in (a, b) for pt in ("disc", "val")}))
fig.suptitle("Shared enriched motifs (exploratory)" if TOPN < 3 else "Shared enriched motifs in both TFs' single-cell binding sites (exploratory; p ≤ 0.05 on discovery and held-out chromosomes)", fontsize=6.5)
fig.savefig(f"{OUT}/fig4c_motifs{TAG}.svg"); fig.savefig(f"{OUT}/fig4c_motifs{TAG}.png", dpi=300); pd.DataFrame(rows_out).to_csv(f"{OUT}/fig4c_motifs_values{TAG}.tsv", sep="\t", index=False)
print(pd.DataFrame(rows_out).to_string(index=False))
