"""RBBP4 Hi-C locus panel (user 2026-11-20): an RBBP4 version of the ETS2 locus panel (hic_ets2_locus, results_3d_model/clique_triangle/
scripts/06_figure_hires_full.py), same layout and style.
Screen: RBBP4 higher-order modules (RBBP4_sweep_k50_i10000_s555/cliques.tsv) with exactly 3 regions, all 3 edges >25 kb and on a Peakachu
0.7 loop (panel d rule: both ends within +-30 kb of the anchors of one 5 kb or 10 kb loop), module span 150-600 kb. Hi-C dots (WT HEK293T,
10 kb, balanced) scored as in clique_triangle/02_score_dots.py: centre = 3x3 pixels on the anchor midpoints, O/E vs the distance mean in
+-1 Mb, donut = Chebyshev distance 2-5 excluding the centre row/column; enrichment = centre O/E / donut O/E; ranked by the weakest of the
three dots. C00355 (chr2, the earlier panel-a candidate) is always scored. Elements: promoter of a GENCODE v46 protein-coding/lncRNA gene
if a TSS lies within +-1 kb, else distal; n TFs = TFs with a Cicero peak (ourrun_k50 + RBBP4 run) within +-1 kb.
Draws the top 3 (plus C00355 if not among them) with the ETS2 panel code: circles at the exact anchor midpoints.
  python hic_rbbp4_locus.py <outdir>   (Longleaf, hic env)"""
import glob, gzip, re, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import h5py, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
X = Path("/work/users/d/y/dyy12/XuLab"); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
for f in glob.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): fm.fontManager.addfont(f)
MCOOL = X / "hic/work_wt293t/coolers-hg38/HEK293TWT-MboI-R1-filtered.mcool"; LD = X / "hic/work_wt293t/loops"
class Hic:   # = clique_triangle/scripts/hicread.py
    def __init__(self, path, res):
        self.f = h5py.File(path, "r"); self.g = self.f[f"resolutions/{res}"]; self.res = res; g = self.g
        self.names = [x.decode() for x in g["chroms/name"][:]]; self.co = g["indexes/chrom_offset"][:]; self.b1o = g["indexes/bin1_offset"]
    def _c(self, c):
        i = self.names.index(c); return int(self.co[i]), int(self.co[i + 1])
    def fetch(self, c, s, e):
        c0, c1 = self._c(c); i0 = c0 + s // self.res; i1 = min(c1, c0 + (e - 1) // self.res + 1); n = i1 - i0; g = self.g
        lo, hi = int(self.b1o[i0]), int(self.b1o[i1]); a = g["pixels/bin1_id"][lo:hi]; b = g["pixels/bin2_id"][lo:hi]; v = g["pixels/count"][lo:hi]; m = b < i1; a, b, v = a[m] - i0, b[m] - i0, v[m]
        R = np.zeros((n, n)); R[a, b] = v; R[b, a] = v; w = g["bins/weight"][i0:i1]; return R * np.outer(w, w), R, w
H = Hic(MCOOL, 10000); RES = 10000
def score(c, ms, PAD=1000000):   # = 02_score_dots.score at 10 kb
    r = RES; L0 = max(0, min(ms) - PAD) // r * r; L1 = max(ms) + PAD; M, R, w = H.fetch(c, L0, L1); n = M.shape[0]; M = np.where(np.isfinite(M), M, np.nan); M[M == 0] = np.where(R[M == 0] > 0, np.nan, 0)
    bad = ~np.isfinite(w); M[bad, :] = np.nan; M[:, bad] = np.nan
    exp = np.array([np.nanmean(np.diagonal(M, d)) if np.isfinite(np.diagonal(M, d)).any() else np.nan for d in range(n)]); ij = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    with np.errstate(all="ignore"): E = M / exp[ij]
    b = [(m - L0) // r for m in ms]; out = {}
    for nm, (i, j) in dict(AB=(b[0], b[1]), BC=(b[1], b[2]), AC=(b[0], b[2])).items():
        cen = E[i - 1:i + 2, j - 1:j + 2]; v = []
        for di in range(-5, 6):
            for dj in range(-5, 6):
                if max(abs(di), abs(dj)) < 2 or di == 0 or dj == 0 or (j + dj) - (i + di) < 2: continue
                v.append(E[i + di, j + dj])
        bg = np.nanmean(v); oe = np.nanmean(cen); out.update({f"{nm}_oe": oe, f"{nm}_don": oe / bg if bg > 0 else np.nan, f"{nm}_reads": R[i - 1:i + 2, j - 1:j + 2].sum()})
    return out
L = defaultdict(list)
for f in ("peakachu_5000.loops.0.7.bedpe", "peakachu_10000.loops.0.7.bedpe"):
    for ln in open(LD / f):
        x = ln.split()
        if x[0] == x[3]:
            a, b = sorted(((int(x[1]), int(x[2])), (int(x[4]), int(x[5])))); L[x[0]].append((*a, *b))
LA = {c: np.array(v) for c, v in L.items()}
def iv(r): c, se = r.strip().split(":"); s, e = se.split("-"); return c, int(s), int(e)
mid = lambda p: (p[1] + p[2]) // 2
def on_loop(p, q, slop=30000):
    if mid(p) > mid(q): p, q = q, p
    A = LA.get(p[0]); return A is not None and bool(((p[1] <= A[:, 1] + slop) & (p[2] >= A[:, 0] - slop) & (q[1] <= A[:, 3] + slop) & (q[2] >= A[:, 2] - slop)).any())
# genes and TF peaks
G = []
for ln in gzip.open(X / "data/gencode_v46/gencode.v46.basic.annotation.gtf.gz", "rt"):
    if ln[0] == "#": continue
    x = ln.split("\t")
    if x[2] != "gene": continue
    ty = re.search('gene_type "([^"]+)"', x[8]).group(1)
    if ty not in ("protein_coding", "lncRNA"): continue
    G.append((x[0], int(x[3]), int(x[4]), x[6], re.search('gene_name "([^"]+)"', x[8]).group(1), "pc" if ty == "protein_coding" else "lnc"))
G = pd.DataFrame(G, columns="c s e st g ty".split()); G["tss"] = np.where(G.st == "+", G.s, G.e)
PK = []
files = {Path(f).parent.name: f for f in glob.glob(str(X / "cobinding/work/ourrun_k50/*/*.peak_annot.tsv"))}; files["RBBP4"] = str(X / "cobinding/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_annot.tsv")
for tf, f in files.items():
    for r in pd.read_csv(f, sep="\t", usecols=["peak"]).peak:
        c, s, e = iv(r); PK.append((c, s, e, tf, (s + e) // 2))
PK = pd.DataFrame(PK, columns="c s e tf sm".split())
def elem(c, s, e):
    g = G[(G.c == c) & (G.tss >= s - 1000) & (G.tss <= e + 1000)]
    k = PK[(PK.c == c) & (PK.sm >= s - 1000) & (PK.sm <= e + 1000)].tf.nunique()
    if len(g): gg = g.sort_values("ty").g.iloc[0]; return f"{gg} promoter", k, gg
    return "distal", k, ""
# screen
C = pd.read_csv(X / "cobinding/results/RBBP4_sweep_k50_i10000_s555/cliques.tsv", sep="\t")
rows = []
for cid, n, rr in zip(C.clique_id, C.n_regions, C.regions.astype(str)):
    if n != 3: continue
    pk = sorted([iv(x) for x in rr.split(";")], key=mid); A_, B_, C_ = pk; span = mid(C_) - mid(A_)
    allloop = all(abs(mid(p) - mid(q)) > 25000 and on_loop(p, q) for p, q in combinations(pk, 2))
    if not ((allloop and 150000 <= span <= 600000) or cid == "C00355"): continue
    d = dict(clique_id=cid, chrom=A_[0], A=f"{A_[0]}:{A_[1]}-{A_[2]}", B=f"{B_[0]}:{B_[1]}-{B_[2]}", C=f"{C_[0]}:{C_[1]}-{C_[2]}", mA=mid(A_), mB=mid(B_), mC=mid(C_),
             span_kb=span / 1e3, all_edges_on_loop=allloop, AB_kb=(mid(B_) - mid(A_)) / 1e3, BC_kb=(mid(C_) - mid(B_)) / 1e3)
    d.update(score(A_[0], (mid(A_), mid(B_), mid(C_))))
    for nm, p in zip("ABC", pk): lab, k, gg = elem(*p); d[f"{nm}_label"] = lab; d[f"{nm}_tfs"] = k; d[f"{nm}_gene"] = gg
    d["min_don"] = min(d["AB_don"], d["BC_don"], d["AC_don"]); d["n_promoter"] = sum(d[f"{x}_label"] != "distal" for x in "ABC"); rows.append(d)
S = pd.DataFrame(rows).sort_values("min_don", ascending=False); S.to_csv(OUT / "rbbp4_locus_candidates.tsv", sep="\t", index=False)
print(len(S), "candidates"); print(S[["clique_id", "chrom", "span_kb", "all_edges_on_loop", "AB_don", "BC_don", "AC_don", "min_don", "A_label", "B_label", "C_label", "A_tfs", "B_tfs", "C_tfs"]].round(2).to_string(index=False))
mix = S[S.all_edges_on_loop & (S.n_promoter.between(1, 2))]
pick = list(mix.clique_id[:3]) + [x for x in S[S.all_edges_on_loop].clique_id if x not in mix.clique_id[:3].values][: max(0, 3 - len(mix[:3]))]
if "C00355" not in pick: pick.append("C00355")
# draw (= 06_figure_hires_full.py layout, exact anchors)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 1, "xtick.major.width": 1,
                     "ytick.major.width": 1, "xtick.major.size": 4, "ytick.major.size": 4, "axes.spines.top": False, "axes.spines.right": False})
RED, BLUE, INK, GREY = "#C0392B", "#4A6FA5", "#222222", "#9E9E9E"; W, Hh = 8.30, 6.60
for cid in pick:
    t = S[S.clique_id == cid].iloc[0]; c = t.chrom; r0 = RES
    X0, X1 = int(t.mA - 80000) // r0 * r0, int(t.mC + 80000) // r0 * r0 + r0; span = (X1 - X0) / 1e3
    M, _, w = H.fetch(c, X0, X1); n = M.shape[0]; Z = np.log10(M + np.nanmedian(M[M > 0]) * 0.1)
    P = PK[(PK.c == c) & (PK.sm >= X0) & (PK.sm < X1)]; GG = G[(G.c == c) & (G.e > X0) & (G.s < X1)]
    fig = plt.figure(figsize=(W, Hh)); Lx, AW = 1.05 / W, 6.90 / W; XL = (X0 / 1e6, X1 / 1e6); ymax = span / 2; hh = 6.90 * ymax / span
    top = 6.32; ax = fig.add_axes([Lx, (top - hh) / Hh, AW, hh / Hh])
    xs = X0 + np.arange(n + 1) * r0; XX, YY = np.meshgrid(xs, xs, indexing="ij"); Zm = np.ma.masked_invalid(np.ma.masked_where(np.tril(np.ones_like(Z), -1) > 0, Z))
    pc = ax.pcolormesh((XX + YY) / 2e6, (YY - XX) / 2e3, Zm, cmap="Reds", vmin=np.nanpercentile(Z, 20), vmax=np.nanpercentile(Z, 98.5), shading="flat", rasterized=True, lw=0, edgecolors="none", antialiased=False)
    for nm, i, j in (("A–B", t.mA, t.mB), ("B–C", t.mB, t.mC), ("A–C", t.mA, t.mC)):
        ax.plot((i + j) / 2e6, (j - i) / 2e3, "o", mfc="none", mec=INK, ms=25, mew=2, zorder=5); ax.text((i + j) / 2e6, (j - i) / 2e3 + 0.075 * ymax, nm, ha="center", va="bottom", fontsize=14, fontweight="bold", color=INK, zorder=6)
    ax.set_xlim(*XL); ax.set_ylim(0, ymax); ax.tick_params(bottom=False, labelbottom=False); ax.spines["bottom"].set_visible(False); ax.set_ylabel("Distance / 2 (kb)", fontsize=14)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True)); ax.set_title("WT HEK293T Hi-C, 10 kb", loc="left", fontsize=14, fontweight="bold", pad=6)
    cb = fig.add_axes([Lx + AW - 1.45 / W, (top - 0.42) / Hh, 1.4 / W, 0.14 / Hh]); plt.colorbar(pc, cax=cb, orientation="horizontal"); cb.set_title("log10 contact", fontsize=12, pad=3); cb.tick_params(labelsize=12); cb.xaxis.set_major_locator(plt.MaxNLocator(3))
    y2 = top - hh - 0.86; ax2 = fig.add_axes([Lx, y2 / Hh, AW, 0.82 / Hh], sharex=ax); ax2.set_axis_off(); ax2.set_ylim(0, 1.2); AN = [("A", t.mA), ("B", t.mB), ("C", t.mC)]
    for a, b in ((t.mA, t.mB), (t.mB, t.mC), (t.mA, t.mC)):
        q = np.linspace(a, b, 100) / 1e6; ax2.plot(q, 0.5 + (0.65 * (b - a) / (t.mC - t.mA)) * np.sin(np.pi * (q - a / 1e6) / ((b - a) / 1e6)), color=BLUE, lw=2)
    for nm, m in AN: ax2.plot(m / 1e6, 0.5, "v", ms=10, color=INK, zorder=4); ax2.text(m / 1e6, 0.33, nm, ha="center", va="top", fontsize=14, fontweight="bold")
    ax2.text(XL[1], 1.17, "Coordinated binding (RBBP4)", color=BLUE, fontsize=12, va="top", ha="right")
    y3 = y2 - 0.84; ax3 = fig.add_axes([Lx, y3 / Hh, AW, 0.68 / Hh], sharex=ax); ed = np.arange(X0 // 5000 * 5000, X1 + 5000, 5000); cnt = np.array([P[(P.sm >= e) & (P.sm < e + 5000)].tf.nunique() for e in ed])
    ymx = max(26, int(cnt.max() * 1.3)); ax3.bar((ed + 2500) / 1e6, cnt, width=0.0048 * span / 840, color="#BFBFBF", lw=0); ax3.set_ylabel("TFs", fontsize=14); ax3.set_ylim(0, ymx); ax3.set_yticks([0, 20]); ax3.tick_params(bottom=False, labelbottom=False)
    for (nm, m), lab, k, ha in zip(AN, (t.A_label, t.B_label, t.C_label), (t.A_tfs, t.B_tfs, t.C_tfs), ("left", "center", "right")):
        ax3.bar(m / 1e6, max(k, 0.8), width=0.0048 * span / 840, color=INK, lw=0); x_ = m / 1e6 if nm != "C" else XL[1]
        ax3.text(x_, ymx * 1.06, f"{lab}, {k} TF" + ("s" if k != 1 else ""), fontsize=12, ha=ha, va="bottom", clip_on=False)
    # genes: anchor-promoter genes first, then protein-coding genes by length; two rows, greedy to avoid label overlap
    key = [g for g in (t.A_gene, t.B_gene, t.C_gene) if g]; rest = GG[(GG.ty == "pc") & ~GG.g.isin(key) & ~GG.g.str.startswith("ENSG") & ~GG.g.str.contains("-")].assign(L=lambda d: d.e - d.s).sort_values("L", ascending=False).g.tolist()
    names = list(dict.fromkeys(key + rest))[:9]; y4 = y3 - 0.62; ax4 = fig.add_axes([Lx, y4 / Hh, AW, 0.56 / Hh], sharex=ax); ax4.set_ylim(1.75, -0.65); ax4.set_yticks([]); ax4.spines["left"].set_visible(False)
    occ = {0: [], 1: []}; cw = 0.092 * span / 1e3 / 6.9   # Mb per character at 12 pt (~0.092 in per italic Arial char)
    for nm in sorted(names, key=lambda g: GG[GG.g == g].s.iloc[0]):
        g = GG[GG.g == nm].iloc[0]; a, b = max(g.s, X0) / 1e6, min(g.e, X1) / 1e6; xc = min(max((a + b) / 2, XL[0] + 0.5 * cw * len(nm)), XL[1] - 0.5 * cw * len(nm)); half = 0.5 * cw * len(nm) + cw
        rw = next((r_ for r_ in (0, 1) if all(xc - half > hi or xc + half < lo for lo, hi in occ[r_])), None)
        if rw is None: continue
        occ[rw].append((xc - half, xc + half)); ax4.plot([a, b], [rw + 0.32] * 2, color=INK, lw=2.2, solid_capstyle="butt"); ax4.text(xc, rw + 0.2, nm, fontsize=12, fontstyle="italic", ha="center", va="bottom")
    ax4.set_xlim(*XL); ax4.xaxis.set_major_locator(plt.MaxNLocator(4)); ax4.set_xlabel(f"{c} (Mb)", fontsize=14, labelpad=2); ax4.tick_params(labelsize=12)
    stem = OUT / f"hic_rbbp4_locus_{cid}"; fig.savefig(f"{stem}.svg", dpi=600, transparent=True); fig.savefig(f"{stem}.pdf", dpi=600, transparent=True); fig.savefig(f"{stem}.png", dpi=200, facecolor="white"); plt.close(fig)
print("drawn:", pick)
