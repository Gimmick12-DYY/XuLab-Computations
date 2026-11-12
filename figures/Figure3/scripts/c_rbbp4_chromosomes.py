"""Fig 3 panel c: RBBP4 coordinated binding pairs and module edges along every chromosome, with only the Hi-C loops that support them.
Above each chromosome: all RBBP4 Cicero pairs >25 kb (first-step pairs of the RBBP4_sweep_k50_i10000_s555 run, peak_edges.tsv, the
same pair set as the S1D / panel d loop tests) in blue (user 2026-11-12: not grey); module edges (pairs inside higher-order coordinated binding modules,
results/RBBP4_sweep_k50_i10000_s555/cliques.tsv, >25 kb) on top in dark purple. Below: only the Peakachu score >=0.7 loops
(wild-type HEK293T, 5 kb + 10 kb) that support at least one shown pair under the panel d rule (both anchors within +-30 kb;
compare_union_pairs_and_clique_edges.assign_loop): full red if they support a module edge, light red if they support plain pairs only.
Paper style: 180 mm wide, Arial 6-7 pt, white background; SVG text as explicit Arial attributes (Illustrator-safe).
Runs on Longleaf (hic env):  python scripts/c_rbbp4_chromosomes.py <outdir>"""
import glob, math, re, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
X = Path("/work/users/d/y/dyy12/XuLab"); sys.path.insert(0, str(X / "hic/scripts"))
import fill_slide_loop_table as F  # noqa: E402
for f in glob.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "xtick.major.size": 2.5, "xtick.labelsize": 6, "axes.labelsize": 7})
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."); COB = X / "cobinding"; L = X / "hic/work_wt293t/loops"
PAIRC, MODC, LOOPM, LOOPP, CHR, INK = "#7A9CCB", "#5B2A86", "#B83A2E", "#F4A582", "#7F7F7F", "#1A1A1A"   # pairs Cicero blue (palette C), module edges dark purple, loops Hi-C reds
fine = F.U.load_bedpe(L / "peakachu_5000.loops.0.7.bedpe", "5000"); coarse = F.U.load_bedpe(L / "peakachu_10000.loops.0.7.bedpe", "10000")
ix5 = F.U.index_anchors(fine, 5000); ix10 = F.U.index_anchors(coarse, 10000)
pairs = [p for p in F.U.load_edges(COB / "work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv") if F.span_bp(p) > F.MIN_SPAN]
edges = [e for e in F.C.load_clique_edges(COB / "results/RBBP4_sweep_k50_i10000_s555/cliques.tsv") if F.span_bp(e) > F.MIN_SPAN]
key = lambda lp: (lp["a"][0], lp["a"][1], lp["a"][2], lp["b"][1], lp["b"][2], lp["res"])
sup_pair, sup_mod = {}, {}
for p in pairs:
    lp = F.C.assign_loop(p, fine, ix5, coarse, ix10, F.SLOP)
    if lp is not None: sup_pair[key(lp)] = lp
for e in edges:
    lp = F.C.assign_loop(e, fine, ix5, coarse, ix10, F.SLOP)
    if lp is not None: sup_mod[key(lp)] = lp
loops_mod = list(sup_mod.values()); loops_pair_only = [v for k, v in sup_pair.items() if k not in sup_mod]
n_on = sum(1 for p in pairs if F.C.assign_loop(p, fine, ix5, coarse, ix10, F.SLOP) is not None)
n_loops_total = len(fine) + len(coarse); n_sup = len(set(sup_pair) | set(sup_mod))
print(f"pairs {len(pairs):,} (on a loop {n_on:,}); module edges {len(edges):,}; supporting loops {n_sup:,} (module {len(loops_mod):,}, pairs only {len(loops_pair_only):,}) of {n_loops_total:,}")
CH = list(F.HG38); gap = 1.0; y_of = {c: (len(CH) - 1 - i) * gap for i, c in enumerate(CH)}
def curve(c, a, b, side, hmax=0.42):
    x1, x2 = sorted((a / 1e6, b / 1e6)); h = min(hmax, 0.07 + 0.2 * math.sqrt(max(x2 - x1, 1e-3))); t = np.linspace(0, 1, 13)
    return np.c_[x1 + (x2 - x1) * t, y_of[c] + side * h * np.sin(np.pi * t)]
mid = lambda p: (p[1] + p[2]) / 2
pc = [curve(p["p1"][0], mid(p["p1"]), mid(p["p2"]), 1) for p in pairs if p["p1"][0] in y_of]
ec = [curve(e["p1"][0], mid(e["p1"]), mid(e["p2"]), 1) for e in edges if e["p1"][0] in y_of]
lc = lambda ls: [curve(l["a"][0], mid(l["a"]), mid(l["b"]), -1) for l in ls if l["a"][0] in y_of]
MM = 1 / 25.4; fig, ax = plt.subplots(figsize=(180 * MM, 150 * MM)); fig.subplots_adjust(left=0.06, right=0.995, top=0.995, bottom=0.075)
for c in CH:
    ax.plot([0, F.HG38[c] / 1e6], [y_of[c], y_of[c]], color=CHR, lw=1.4, solid_capstyle="round", zorder=4)
    ax.text(-3, y_of[c], c.replace("chr", ""), ha="right", va="center", fontsize=6, color=INK)
ax.add_collection(LineCollection(pc, colors=PAIRC, linewidths=0.3, alpha=0.75, zorder=1))
ax.add_collection(LineCollection(lc(loops_pair_only), colors=LOOPP, linewidths=0.3, alpha=0.9, zorder=2))
ax.add_collection(LineCollection(lc(loops_mod), colors=LOOPM, linewidths=0.55, alpha=1, zorder=3))
ax.add_collection(LineCollection(ec, colors=MODC, linewidths=0.7, alpha=1, zorder=5))
mx = max(F.HG38.values()) / 1e6; ax.set_xlim(-8, mx + 6); ax.set_ylim(-0.6, y_of[CH[0]] + 0.55)
ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ("top", "right", "left")]; ax.spines["bottom"].set_color("#7F7F7F")
ax.set_xlabel("Position (Mb, hg38)"); ax.tick_params(axis="x", colors=INK)
ax.text(-7.5, y_of[CH[0]] + 0.42, "pairs ↑   loops ↓", fontsize=5.5, color="#555555", ha="left", va="center")
h = [Line2D([], [], color=PAIRC, lw=1.2), Line2D([], [], color=MODC, lw=1.2), Line2D([], [], color=LOOPM, lw=1.2), Line2D([], [], color=LOOPP, lw=1.2)]
lab = [f"Coordinated binding pair (n = {len(pairs):,})", f"Module edge (n = {len(edges):,})",
       f"Hi-C loop supporting a module edge (n = {len(loops_mod):,})", f"Hi-C loop supporting other pairs only (n = {len(loops_pair_only):,})"]
ax.legend(h, lab, loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=6, frameon=False, handlelength=1.4, labelspacing=0.35,
          title=f"Supporting loops: {n_sup:,} of {n_loops_total:,} Peakachu loops", title_fontsize=6, alignment="left")
PS = {("normal", "normal"): "ArialMT", ("bold", "normal"): "Arial-BoldMT"}
def illustrator_text(svg):
    def one(m):
        tag = m.group(0); sm = re.search(r'style="([^"]*)"', tag)
        if not sm: return tag
        keep, size, wt = [], None, "normal"
        for d in [d.strip() for d in sm.group(1).split(";") if d.strip()]:
            k, _, v = d.partition(":"); k, v = k.strip(), v.strip()
            if k == "font":
                mm_ = re.match(r"(?:(?:italic|oblique)\s+)?(?:(bold|[1-9]00)\s+)?([\d.]+)px", v)
                if mm_: wt = "bold" if mm_.group(1) in ("bold", "700") else wt; size = mm_.group(2)
            else: keep.append(f"{k}: {v}")
        if size is None: return tag
        tag = tag.replace(sm.group(0), f'style="{"; ".join(keep)}"' if keep else "")
        return tag[:-1].rstrip() + f' font-family="Arial, {PS[(wt, "normal")]}, Helvetica, sans-serif" font-size="{float(size):.3f}"' + (' font-weight="bold"' if wt == "bold" else "") + ">"
    svg = re.sub(r"<text\b[^>]*>", one, svg)
    return re.sub(r"<tspan\b[^>]*>", lambda m: one(re.match(r"[\s\S]*", m.group(0).replace("<tspan", "<text", 1))).replace("<text", "<tspan", 1), svg)
stem = OUT / "c_rbbp4_chromosomes"; fig.savefig(f"{stem}.svg"); fig.savefig(f"{stem}.png", dpi=300)   # PNG + SVG only (user)
p = Path(f"{stem}.svg"); p.write_text(illustrator_text(p.read_text())); print("font shorthand left:", p.read_text().count("font: "))
