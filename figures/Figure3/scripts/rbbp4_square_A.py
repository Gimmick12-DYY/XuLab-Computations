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
cls = pd.Series([v[2] for v in pk.values()]).value_counts(); NP = len(pk)
BARS = [("Pairs", [("on loop", sum(pl.values()), LOOP, "white"), ("", NP - sum(pl.values()), NOLOOP, N.dark)]),
        ("Module\nedges", [("on loop", sum(ME.values()), LOOP, "white"), ("", len(ME) - sum(ME.values()), NOLOOP, N.dark)]),
        ("Pair\nclasses", [("P–P", int(cls.get("PP", 0)), "#2B5C9E", "white"), ("P–D", int(cls.get("DP", 0)), "#7B5EA7", "white"), ("D–D", int(cls.get("DD", 0)), "#E8821E", "white")])]
plt.rcParams.update({"font.size": 6.5})
fig = plt.figure(figsize=(60 * MM, 60 * MM)); ax = fig.add_axes([0.17, 0.13, 0.8, 0.8])
for i, (lab, segs) in enumerate(BARS):
    tot, left = sum(v for _, v, _, _ in segs), 0
    for nm, v, col, tc in segs:
        f = v / tot; ax.barh(i, f, 0.74, left=left, color=col, lw=0.8, edgecolor="white")
        ax.text(left + f / 2, i, (f"{nm}\n" if nm else "") + f"{100 * f:.1f}%\nn = {v:,}", ha="center", va="center", fontsize=6, color=tc, linespacing=1.1)
        left += f
ax.set_yticks(range(len(BARS))); ax.set_yticklabels([b[0] for b in BARS], fontsize=7, linespacing=1.0); ax.set_ylim(len(BARS) - 0.5, -0.5)
ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(["0", "50", "100"], fontsize=6); ax.set_xlabel("%", fontsize=6, labelpad=1); ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=2, pad=1)
for sp_ in ("left", "top", "right"): ax.spines[sp_].set_visible(False)
print("pairs", NP, "on loop", sum(pl.values()), "| module edges unique", len(ME), "on loop", sum(ME.values()), "| classes", cls.to_dict())
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
