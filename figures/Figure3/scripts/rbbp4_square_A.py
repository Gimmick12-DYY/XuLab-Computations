"""Fig 3 RBBP4 characterization, SQUARE variant for the top-row slot (2026-11-20; same data/colours as b_rbbp4_characterization.py).\nOriginal docstring: (user 2026-11-17): top = compact bars of RBBP4 coordinated binding pairs and module edges by module size,
with Hi-C loop support; bottom = one stacked strand of all RBBP4 pairs by the element class of their two ends.
Data = panels b/c: RBBP4 first-step Cicero pairs (RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv, p <= 0.05, ends >25 kb apart) and its
higher-order coordinated binding modules (cliques.tsv, >= 3 regions, all pairs linked). Module edges = pairs of regions inside one module
(>25 kb); an edge in a module of 3 / 4 / >= 5 regions is counted under that size.
On a loop: panel d rule (clique_coverage.py): both ends within +-30 kb of the two anchors of one Peakachu score >= 0.7 loop (5 kb or 10 kb,
wild-type HEK293T). Module loop coverage = fraction of the module's >25 kb edges on a loop, panel d classes and colours.
Element class of a pair end (panel e rule, annot_chromhmm.py): promoter = RefSeq Select TSS +-1 kb; otherwise the HEK293T 18-state
ChromHMM state at the region midpoint: enhancer (EnhA1/A2/G1/G2/Wk/Biv), repressive (ReprPC/ReprPCWk/Het/ZNF-Rpts), other (TSS flank,
transcribed, quiescent).
  python Figure3/scripts/b_rbbp4_characterization.py <input_dir> <outdir>   (Mac; input_dir holds RBBP4.peak_edges.tsv, cliques.tsv,
  peakachu_5000/10000.loops.0.7.bedpe copied from Longleaf)"""
import glob, gzip, json, re, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path("/Users/dyy/Desktop/HEK293T-ATGTAG"); sys.path.insert(0, str(ROOT / "palette")); import regulome_palette as rp
P = rp.use(); N = P.neutral; MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False})
I = Path(sys.argv[1]); OUT = Path(sys.argv[2]); CI = ROOT / "hub_figures/basic/descriptive/chromhmm_rebuild_inputs"
# ---- loops (clique_coverage.py) ----
L = defaultdict(list)
for f in ("peakachu_5000.loops.0.7.bedpe", "peakachu_10000.loops.0.7.bedpe"):
    for ln in open(I / f):
        x = ln.split()
        if x[0] == x[3]:
            a = (int(x[1]), int(x[2])); b = (int(x[4]), int(x[5]))
            if a > b: a, b = b, a
            L[x[0]].append((*a, *b))
LA = {c: np.array(v) for c, v in L.items()}
def iv(r):
    c, se = r.strip().split(":"); s, e = se.split("-"); return c, int(s), int(e)
mid = lambda p: (p[1] + p[2]) // 2
def on_loop(p, q, slop=30000):
    if mid(p) > mid(q): p, q = q, p
    A = LA.get(p[0])
    return A is not None and bool(((p[1] <= A[:, 1] + slop) & (p[2] >= A[:, 0] - slop) & (q[1] <= A[:, 3] + slop) & (q[2] >= A[:, 2] - slop)).any())
# ---- element classes (annot_chromhmm.py / rebuild_hub_regions_chromhmm.py) ----
GROUP = {"EnhA1": "enhancer", "EnhA2": "enhancer", "EnhG1": "enhancer", "EnhG2": "enhancer", "EnhWk": "enhancer", "EnhBiv": "enhancer",
         "ReprPC": "repressive", "ReprPCWk": "repressive", "Het": "repressive", "ZNF/Rpts": "repressive"}
TSS = {}
for f in glob.glob(str(CI / "rs/*.json")):
    v = json.load(open(f))["ncbiRefSeqSelect"]
    if v: TSS[Path(f).stem] = np.sort([(g["txStart"] if g["strand"] == "+" else g["txEnd"]) for g in v])
H = defaultdict(list)
for ln in gzip.open(CI / "HEK293T_chromHMM18.bed.gz", "rt"):
    f = ln.split("\t"); H[f[0]].append((int(f[1]), int(f[2]), f[3]))
HS = {c: (np.array([a for a, _, _ in sorted(v)]), np.array([b for _, b, _ in sorted(v)]), [s for _, _, s in sorted(v)]) for c, v in H.items()}
def cls(c, s, e):
    if c in TSS:
        p = TSS[c]; i = np.searchsorted(p, s - 1000)
        if i < len(p) and p[i] <= e + 1000: return "promoter"
    if c not in HS: return "other"
    st, en, lab = HS[c]; m = (s + e) // 2; i = np.searchsorted(st, m, side="right") - 1
    return GROUP.get(lab[i], "other") if i >= 0 and en[i] > m else "other"
# ---- pairs ----
E = pd.read_csv(I / "RBBP4.peak_edges.tsv", sep="\t"); E = E[E.pval <= 0.05]
pairs = []
for a, b in zip(E.peak1, E.peak2):
    p, q = iv(a), iv(b)
    if p[0] == q[0] and abs(mid(p) - mid(q)) > 25000: pairs.append((p, q))
ploop = np.array([on_loop(p, q) for p, q in pairs])
# ---- modules ----
C = pd.read_csv(I / "cliques.tsv", sep="\t", usecols=["clique_id", "n_regions", "regions"]); mods = []
for cid, rr in zip(C.clique_id, C.regions.astype(str)):
    pk = [iv(x) for x in rr.split(";")]; es = [(p, q) for p, q in combinations(pk, 2) if abs(mid(p) - mid(q)) > 25000]
    if es: mods.append(dict(size=len(pk), edges=len(es), loop=sum(on_loop(p, q) for p, q in es)))
M = pd.DataFrame(mods); M["sz"] = np.where(M["size"] >= 5, "≥5", M["size"].astype(str))
SZ = ["3", "4", "≥5"]
cats = ["Pairs"] + [f"{s}-region\nmodules" if s != "≥5" else "≥5-region\nmodules" for s in SZ]
ne = [len(pairs)] + [int(M.edges[M.sz == s].sum()) for s in SZ]; nl = [int(ploop.sum())] + [int(M.loop[M.sz == s].sum()) for s in SZ]
labs = ["no loop edge", ">0–25%", "25–50%", "50–75%", "75–<100%", "100% (all edges)"]; cols = P.loop_strata; cv = M.loop / M.edges
M["stratum"] = np.select([M.loop == 0, cv <= .25, cv <= .5, cv <= .75, cv < 1], labs[:5], labs[5])
Ms = M.groupby(["sz", "stratum"]).size().unstack(fill_value=0).reindex(index=SZ, columns=labs, fill_value=0)
# ---- pair classes ----
pc = []
for p, q in pairs:
    a, b = sorted([cls(*p), cls(*q)], key=["promoter", "enhancer", "repressive", "other"].index); pc.append(f"{a}-{b}")
pc = pd.Series(pc).value_counts()
SHOW = [("promoter-promoter", "P–P"), ("promoter-enhancer", "P–E"), ("enhancer-enhancer", "E–E"), ("promoter-repressive", "P–R"),
        ("enhancer-repressive", "E–R"), ("repressive-repressive", "R–R")]                                     # user's classes; all the rest = one part
seg = [(lab, int(pc.get(k, 0))) for k, lab in SHOW]; seg.append(("Rest", len(pairs) - sum(v for _, v in seg)))
SEGC = {"P–P": "#2B5C9E", "P–E": "#7FA6CE", "E–E": "#E8821E", "P–R": "#8E7CC3", "E–R": "#F6CB8F", "R–R": "#55606E", "Rest": "#E3E3E3"}
LOOP, NOLOOP = "#B83A2E", "#E3E3E3"
# ---- square layout v2 (user 2026-11-20, our RBBP4 run): three equal 100% bars, labels inside the segments ----
# Pairs = first-step Cicero pairs >25 kb, unique by unordered region pair; module edges = pairs of regions inside one module (>25 kb),
# unique by unordered region pair across modules; on a loop = panel d rule. P/D = the pipeline's peak type (build_pairs_from_matrix.py:
# ChromHMM-18 promoter states Tss* = proximal, else distal), taken from the peak_edges type columns.
E2 = pd.read_csv(I / "RBBP4.peak_edges.tsv", sep="\t"); E2 = E2[E2.pval <= 0.05]
pk = {}
for a_, b_, t1, t2 in zip(E2.peak1, E2.peak2, E2.type1, E2.type2):
    p_, q_ = iv(a_), iv(b_)
    if p_[0] != q_[0] or abs(mid(p_) - mid(q_)) <= 25000: continue
    k = tuple(sorted([a_, b_])); pk[k] = (p_, q_, "".join(sorted([t1[0].upper(), t2[0].upper()])))
pl = {k: on_loop(p_, q_) for k, (p_, q_, _) in pk.items()}
ME = {}
for rr in C.regions.astype(str):
    regs = [x.strip() for x in rr.split(";")]
    for x, y in combinations(regs, 2):
        p_, q_ = iv(x), iv(y)
        if abs(mid(p_) - mid(q_)) > 25000: ME[tuple(sorted([x, y]))] = on_loop(p_, q_)
pdcls = pd.Series([v[2] for v in pk.values()]).value_counts(); NP = len(pk)
# pair classes (user 2026-11-20): element class of the two ends (panel e rule: promoter = RefSeq Select TSS +-1 kb, else ChromHMM state at the
# midpoint; enhancer = EnhA1/A2/G1/G2/Wk/Biv, repressive = ReprPC/ReprPCWk/Het/ZNF-Rpts); all other combinations = Rest
CLS5 = [("promoter-promoter", "P–P", "#2B5C9E"), ("promoter-enhancer", "P–E", "#7FA6CE"), ("enhancer-enhancer", "E–E", "#E8821E"),
        ("promoter-repressive", "P–R", "#7B5EA7"), ("repressive-repressive", "R–R", "#55606E")]; NP5 = len(pairs)
BARS = [("Pairs", [("on loop", sum(pl.values()), LOOP, "white"), ("", NP - sum(pl.values()), NOLOOP, N.dark)]),
        ("Module\nedges", [("on loop", sum(ME.values()), LOOP, "white"), ("", len(ME) - sum(ME.values()), NOLOOP, N.dark)]),
        ("Pair\nclasses", [(lab, int(pc.get(k, 0)), col, "white") for k, lab, col in CLS5] + [("Rest", NP5 - sum(int(pc.get(k, 0)) for k, _, _ in CLS5), "#E3E3E3", N.dark)])]
plt.rcParams.update({"font.size": 6.5})
# ---- two-ring donut (user 2026-11-20): inner = pair class (element classes of the two ends); outer = within each class, on a Hi-C loop
#      (dark shade) vs not (light tint). Palette validated with the dataviz validator (all checks pass; orange contrast relieved by labels). ----
from matplotlib.colors import to_rgb
CL6 = [("promoter-promoter", "Promoter–promoter", "#2B5C9E"), ("promoter-enhancer", "Promoter–enhancer", "#2A9D8F"), ("enhancer-enhancer", "Enhancer–enhancer", "#E8821E"),
       ("promoter-repressive", "Promoter–repressive", "#7E57C2"), ("repressive-repressive", "Repressive–repressive", "#C0392B"), ("rest", "Rest", "#BDBDBD")]
def klass(p_, q_):
    a_, b_ = sorted([cls(*p_), cls(*q_)], key=["promoter", "enhancer", "repressive", "other"].index); k = f"{a_}-{b_}"
    return k if k in {c[0] for c in CL6[:5]} else "rest"
mekeys = set(ME)
T = pd.DataFrame([(klass(p_, q_), pl[k], k in mekeys) for k, (p_, q_, _) in pk.items()], columns=["cls", "loop", "module"])
SUM = T.groupby("cls").agg(n=("loop", "size"), on_loop=("loop", "sum"), module=("module", "sum")).reindex([c[0] for c in CL6])
SUM["pct"] = 100 * SUM.n / SUM.n.sum(); SUM["pct_loop"] = 100 * SUM.on_loop / SUM.n; SUM["pct_module"] = 100 * SUM.module / SUM.n
print(SUM.round(1).to_string()); print("module edges matched to pairs:", int(T.module.sum()), "of", len(ME))
shade = lambda c, f: tuple(np.array(to_rgb(c)) * (1 - f))                       # darker
tint = lambda c, f: tuple(np.array(to_rgb(c)) + (1 - np.array(to_rgb(c))) * f)    # lighter
fig = plt.figure(figsize=(60 * MM, 60 * MM)); pa = fig.add_axes([0.0, 0.22, 0.57, 0.74]); pa.set_aspect("equal"); pa.axis("off")
vals = SUM.n.values; cols6 = [c[2] for c in CL6]
pa.pie(vals, colors=cols6, startangle=90, counterclock=False, radius=0.8, wedgeprops=dict(width=0.32, edgecolor="white", linewidth=0.8))
outer, ocol = [], []
for (k, nm, c), (_, r) in zip(CL6, SUM.iterrows()):
    outer += [r.on_loop, r.n - r.on_loop]; ocol += [shade(c, 0.35), tint(c, 0.6)]
pa.pie(outer, colors=ocol, startangle=90, counterclock=False, radius=1.0, wedgeprops=dict(width=0.17, edgecolor="white", linewidth=0.6))
ang0 = 90; tot = vals.sum()
for (k, nm, c), (_, r) in zip(CL6, SUM.iterrows()):
    span = 360 * r.n / tot; mid_ = np.deg2rad(ang0 - span / 2); ang0 -= span
    if r.pct >= 8: pa.text(0.64 * np.cos(mid_), 0.64 * np.sin(mid_), f"{r.pct:.0f}%", ha="center", va="center", fontsize=6, color="white" if k != "rest" else N.ink)
pa.text(0, 0.07, f"{int(tot):,}", ha="center", va="center", fontsize=7, color=N.dark, weight="bold"); pa.text(0, -0.13, "RBBP4 pairs", ha="center", va="center", fontsize=6, color=N.dark)
pa.set_xlim(-1.03, 1.03); pa.set_ylim(-1.03, 1.03)
# legend/table: class, % of pairs, % on a Hi-C loop
tx = fig.add_axes([0.59, 0.28, 0.39, 0.62]); tx.axis("off"); tx.set_xlim(0, 1); tx.set_ylim(len(CL6) - 0.4, -1.1)
tx.text(0.66, -0.75, "pairs", ha="right", va="center", fontsize=6, color=N.dark); tx.text(1.0, -0.75, "on loop", ha="right", va="center", fontsize=6, color=N.dark)
for i, ((k, nm, c), (_, r)) in enumerate(zip(CL6, SUM.iterrows())):
    tx.add_patch(plt.Rectangle((0.0, i - 0.27), 0.08, 0.54, color=c, lw=0))
    tx.text(0.12, i, {"Promoter–promoter": "P–P", "Promoter–enhancer": "P–E", "Enhancer–enhancer": "E–E", "Promoter–repressive": "P–R", "Repressive–repressive": "R–R"}.get(nm, nm), va="center", fontsize=6)
    tx.text(0.66, i, f"{r.pct:.1f}%", ha="right", va="center", fontsize=6); tx.text(1.0, i, f"{r.pct_loop:.0f}%", ha="right", va="center", fontsize=6)
fig.text(0.04, 0.165, "Inner ring: element classes of the two pair ends\n(P promoter, E enhancer, R repressive; Rest: other).\nOuter ring: dark = on a Hi-C loop, light = not.", fontsize=6, color=N.dark, va="top", linespacing=1.2)
PS = {("normal", "normal"): "ArialMT", ("bold", "normal"): "Arial-BoldMT"}
def illustrator_text(svg):
    def one(m):
        tag = m.group(0); sm = re.search(r'style="([^"]*)"', tag)
        if not sm: return tag
        keep, size, wt = [], None, "normal"
        for d in [d.strip() for d in sm.group(1).split(";") if d.strip()]:
            k, _, v = d.partition(":"); k, v = k.strip(), v.strip()
            if k == "font":
                mm_ = re.match(r"(?:(bold|[1-9]00)\s+)?([\d.]+)px", v)
                if mm_: wt = "bold" if mm_.group(1) in ("bold", "700", "800", "900") else wt; size = mm_.group(2)
            else: keep.append(f"{k}: {v}")
        if size is None: return tag
        tag = tag.replace(sm.group(0), f'style="{"; ".join(keep)}"' if keep else "")
        return tag[:-1].rstrip() + f' font-family="Arial, {PS[(wt, "normal")]}, Helvetica, sans-serif" font-size="{float(size):.3f}"' + (' font-weight="bold"' if wt == "bold" else "") + ">"
    return re.sub(r"<(?:text|tspan)\b[^>]*>", one, svg)
stem = OUT / "rbbp4_square_A"; fig.savefig(f"{stem}.svg"); fig.savefig(f"{stem}.png", dpi=300)
p = Path(f"{stem}.svg"); p.write_text(illustrator_text(p.read_text()))
