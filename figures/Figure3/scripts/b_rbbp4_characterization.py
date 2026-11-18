"""Fig 3 RBBP4 characterization (user 2026-11-17): top = compact bars of RBBP4 coordinated binding pairs and module edges by module size,
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
# ---- figure: top = 100% bars (on a Hi-C loop vs not) per edge set; bottom = one strand of pair classes, same x range and left edge ----
fig = plt.figure(figsize=(120 * MM, 60 * MM)); X0, XW = 0.25, 0.53
a1 = fig.add_axes([X0, 0.47, XW, 0.38]); a3 = fig.add_axes([X0, 0.335, XW, 0.07])
nmod = {s: int((M.sz == s).sum()) for s in SZ}
ROWS = [("Pairs", ne[0], nl[0], f"n = {ne[0]:,}")] + [(f"{s}-region module edges", e, l, f"n = {e:,} ({nmod[s]} module{'s' if nmod[s] != 1 else ''})" if e else "none")
                                                     for s, e, l in zip(SZ, ne[1:], nl[1:])]
for i, (lab, e, l, nt) in enumerate(ROWS):
    if e:
        f = l / e; a1.barh(i, f, 0.78, color=LOOP, lw=0); a1.barh(i, 1 - f, 0.78, left=f, color=NOLOOP, lw=0)
        a1.text(f / 2, i, f"{100 * f:.0f}%", va="center", ha="center", fontsize=7, color="white")
    a1.text(1.02, i, nt, va="center", ha="left", fontsize=6.5, color=N.dark, clip_on=False)
a1.set_yticks(range(len(ROWS))); a1.set_yticklabels([r[0] for r in ROWS], fontsize=7); a1.set_ylim(len(ROWS) - 0.5, -0.5)
a1.set_xlim(0, 1); a1.set_xticks([]); a1.tick_params(axis="y", length=0)
for sp in ("left", "bottom"): a1.spines[sp].set_visible(False)
fig.text(0.02, 0.965, "RBBP4 coordinated binding", fontsize=7.5, va="top")
hl = [plt.Rectangle((0, 0), 1, 1, color=LOOP), plt.Rectangle((0, 0), 1, 1, color=NOLOOP)]
a1.legend(hl, ["On a Hi-C loop", "Not on a loop"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=6.5, handlelength=0.9, handleheight=0.8, borderaxespad=0.2)
left, tot = 0, len(pairs); pct = []
for lab, v in seg:
    f = v / tot; a3.barh(0, f, 1.0, left=left, color=SEGC[lab], lw=0)
    if f >= 0.08: a3.text(left + f / 2, 0, f"{100 * f:.0f}%", ha="center", va="center", fontsize=7, color="white" if lab in ("P–P", "R–R") else N.ink)
    left += f; pct.append(100 * f)
a3.set_xlim(0, 1); a3.set_ylim(-0.5, 0.5); a3.set_yticks([0]); a3.set_yticklabels(["Pair classes"], fontsize=7); a3.tick_params(axis="y", length=0)
a3.set_xticks([0, .25, .5, .75, 1]); a3.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=6.5); a3.set_xlabel(f"% of RBBP4 pairs >25 kb (n = {tot:,})", fontsize=7, labelpad=2)
a3.spines["left"].set_visible(False)
fmt = lambda x: f"{x:.0f}%" if x >= 8 else f"{x:.1f}%"
hs = [plt.Rectangle((0, 0), 1, 1, color=SEGC[lab]) for lab, _ in seg]
fig.legend(hs, [f"{lab} {fmt(x)}" for (lab, _), x in zip(seg, pct)], loc="lower left", bbox_to_anchor=(0.02, 0.075), ncol=7, fontsize=6.5, handlelength=0.8,
           handleheight=0.8, handletextpad=0.3, columnspacing=0.8, borderaxespad=0)
fig.text(0.02, 0.012, "P promoter, E enhancer, R repressive (ChromHMM); Rest: all other combinations", fontsize=6, color=N.dark)
print("strand %:", {lab: round(x, 2) for (lab, _), x in zip(seg, pct)}, "sum", round(sum(pct), 2), "| rows:", ROWS)
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
stem = OUT / "b_rbbp4_characterization"; fig.savefig(f"{stem}.svg"); fig.savefig(f"{stem}.png", dpi=300)
p = Path(f"{stem}.svg"); p.write_text(illustrator_text(p.read_text()))
print("pairs", len(pairs), "on loop", int(ploop.sum())); print(pd.DataFrame({"cat": cats, "edges": ne, "on_loop": nl}).to_string(index=False)); print(Ms)
print(pc.to_string()); print(seg)
