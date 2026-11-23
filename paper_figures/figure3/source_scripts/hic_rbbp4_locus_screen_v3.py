"""RBBP4 Hi-C locus screen v3 (2026-11-23): widens the pool beyond the 315 q-graph MCCs of v2 (hic_rbbp4_locus_screen_v2.py).
Candidate triplets (A < B < C, every pair >= 60 kb apart, span 100 kb - 2 Mb):
  loop-first : three Peakachu 0.7 loop anchors (5 kb + 10 kb lists; anchors within 10 kb merged) with A-B, B-C and A-C all called loops;
               every anchor holds an RBBP4 peak within +-10 kb and >= 2 of the 3 anchor pairs are RBBP4 coordinated binding pairs (p <= 0.05)
               between those peaks; the element of an anchor = its RBBP4 peak with the most p <= 0.05 pairs.
  p-MCC      : maximal cliques (>= 3 regions, all pairs linked) of the RBBP4 p <= 0.05 pair graph (all first-step Cicero pairs, <= 1 Mb),
               every 3-region subset (so 4+-region MCCs contribute their best subset; the q <= 0.05 MCCs of v2 are a subset of these).
Hi-C (WT HEK293T, balanced): 10 kb for spans <= 1 Mb, 25 kb above; a pair closer than 6 bins at 25 kb is scored at 10 kb. One fixed setting
per resolution: best centre within +-1 bin (10 kb +-10 kb, 25 kb +-25 kb). Per pair: (a) dot / donut = centre 3x3 O/E over donut
(Chebyshev 2-5, centre row/column excluded), (b) O/E = centre 3x3 observed / chromosome-wide distance expectation. Candidate score = the
weakest pair on the weaker metric; pass = every pair >= 1.5 on both. Draws the top 5 (raw contacts and log2 O/E) on one contact sheet.
  python hic_rbbp4_locus_screen_v3.py <outdir> <contact_sheet.png>   (Longleaf, hic env)"""
import glob, gzip, re, sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import h5py, numpy as np, pandas as pd, networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
X = Path("/work/users/d/y/dyy12/XuLab"); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); SHEET = Path(sys.argv[2])
for f in glob.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): fm.fontManager.addfont(f)
MCOOL = X / "hic/work_wt293t/coolers-hg38/HEK293TWT-MboI-R1-filtered.mcool"; LD = X / "hic/work_wt293t/loops"
MIN_SEP, MIN_SPAN, MAX_SPAN, PASS = 60_000, 100_000, 2_000_000, 1.5
def iv(x): c, se = x.strip().split(":"); s, e = se.split("-"); return c, int(s), int(e)
mid = lambda p: (p[1] + p[2]) // 2

# Hi-C: banded balanced matrix per chromosome --------------------------------------------------------------------------------
class Band:
    def __init__(self, res, dmax):
        self.f = h5py.File(MCOOL, "r"); self.g = self.f[f"resolutions/{res}"]; self.res = res; self.D = dmax // res + 8
        self.names = [x.decode() for x in self.g["chroms/name"][:]]; self.co = self.g["indexes/chrom_offset"][:]; self.cache = {}
    def load(self, c):
        if c in self.cache: return self.cache[c]
        self.cache.clear(); g = self.g; i = self.names.index(c); c0, c1 = int(self.co[i]), int(self.co[i + 1]); n = c1 - c0
        w = g["bins/weight"][c0:c1]; B = np.full((self.D + 1, n), np.nan); Rw = np.zeros((self.D + 1, n))
        lo, hi = int(g["indexes/bin1_offset"][c0]), int(g["indexes/bin1_offset"][c1]); step = 20_000_000
        for a0 in range(lo, hi, step):
            a = g["pixels/bin1_id"][a0:min(hi, a0 + step)] - c0; b = g["pixels/bin2_id"][a0:min(hi, a0 + step)] - c0; v = g["pixels/count"][a0:min(hi, a0 + step)]
            d = b - a; m = (d <= self.D) & (b < n); a, d, v, b = a[m], d[m], v[m], b[m]
            B[d, a] = v * w[a] * w[b]; Rw[d, a] = v
        ok = np.isfinite(w)
        for d in range(self.D + 1):   # unobserved pixels between balanced bins are zeros
            row = B[d]; valid = np.zeros(n, bool); valid[: n - d] = ok[: n - d] & ok[d:]; row[valid & ~np.isfinite(row)] = 0; row[~valid] = np.nan
        exp = np.nanmean(B, axis=1)
        with np.errstate(all="ignore"): E = B / exp[:, None]
        self.cache[c] = (B, E, exp, Rw); return self.cache[c]
    def oe(self, E, i, j):
        d = j - i
        if d < 0 or d > self.D or i < 0 or i >= E.shape[1]: return np.nan
        return E[d, i]
    def score_pair(self, c, m1, m2):
        """best centre within +-1 bin: (donut ratio, O/E, centre bp i, centre bp j)"""
        B, E, exp, Rw = self.load(c); r = self.res; i0, j0 = m1 // r, m2 // r; best = (np.nan, np.nan, None, None)
        if j0 - i0 < 6 or j0 - i0 > self.D - 6: return best
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                i, j = i0 + di, j0 + dj
                cen = np.nanmean([self.oe(E, i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1)])
                don = [self.oe(E, i + a, j + b) for a in range(-5, 6) for b in range(-5, 6) if max(abs(a), abs(b)) >= 2 and a != 0 and b != 0]
                bg = np.nanmean(don)
                if not (np.isfinite(cen) and np.isfinite(bg) and bg > 0): continue
                dr = cen / bg
                if not (best[0] >= dr): best = (dr, cen, i * r + r // 2, j * r + r // 2)
        return best
H10, H25 = Band(10000, 1_100_000), Band(25000, 2_100_000)

# loops, annotation ------------------------------------------------------------------------------------------------------------
LOOPS = []
for f in ("peakachu_5000.loops.0.7.bedpe", "peakachu_10000.loops.0.7.bedpe"):
    for ln in open(LD / f):
        x = ln.split()
        if x[0] == x[3]: LOOPS.append((x[0], (int(x[1]) + int(x[2])) // 2, (int(x[4]) + int(x[5])) // 2))
L = defaultdict(list)
for c, a, b in LOOPS: L[c].append((min(a, b), max(a, b)))
LA = {c: np.array(v) for c, v in L.items()}
def on_loop(c, m1, m2, slop=30000):
    A = LA.get(c); m1, m2 = min(m1, m2), max(m1, m2)
    return A is not None and bool(((np.abs(A[:, 0] - m1) <= slop) & (np.abs(A[:, 1] - m2) <= slop)).any())
G = []
for ln in gzip.open(X / "data/gencode_v46/gencode.v46.basic.annotation.gtf.gz", "rt"):
    if ln[0] == "#": continue
    x = ln.split("\t")
    if x[2] != "gene": continue
    ty = re.search('gene_type "([^"]+)"', x[8]).group(1)
    if ty not in ("protein_coding", "lncRNA"): continue
    G.append((x[0], int(x[3]), int(x[4]), x[6], re.search('gene_name "([^"]+)"', x[8]).group(1), "pc" if ty == "protein_coding" else "lnc"))
G = pd.DataFrame(G, columns="c s e st g ty".split()); G["tss"] = np.where(G.st == "+", G.s, G.e)
HM = defaultdict(list)
for ln in gzip.open(X / "data/HEK293T_chromHMM18.bed.gz", "rt"):
    f = ln.split("\t"); HM[f[0]].append((int(f[1]), int(f[2]), f[3]))
HMS = {c: (np.array([a for a, _, _ in sorted(v)]), np.array([b for _, b, _ in sorted(v)]), [s for _, _, s in sorted(v)]) for c, v in HM.items()}
def state(c, m):
    if c not in HMS: return "none"
    a, b, lab = HMS[c]; i = np.searchsorted(a, m, side="right") - 1
    return lab[i] if i >= 0 and b[i] > m else "none"
PK = []
files = {Path(f).parent.name: f for f in glob.glob(str(X / "cobinding/work/ourrun_k50/*/*.peak_annot.tsv"))}
files["RBBP4"] = str(X / "cobinding/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_annot.tsv")
for tf, f in files.items():
    for r in pd.read_csv(f, sep="\t", usecols=["peak"]).peak:
        c, s, e = iv(r); PK.append((c, (s + e) // 2, tf))
PK = pd.DataFrame(PK, columns="c sm tf".split())
BL = defaultdict(list)
for ln in gzip.open(X / "downstream/cache/hg38-blacklist.v2.bed.gz", "rt"):
    f = ln.split("\t"); BL[f[0]].append((int(f[1]), int(f[2])))
def in_bl(c, s, e): return any(a < e and b > s for a, b in BL.get(c, ()))
def elem(c, s, e):
    g = G[(G.c == c) & (G.tss >= s - 1000) & (G.tss <= e + 1000)]
    k = PK[(PK.c == c) & (PK.sm >= s - 1000) & (PK.sm <= e + 1000)].tf.nunique(); st = state(c, (s + e) // 2)
    if len(g): return f"{g.sort_values('ty').g.iloc[0]} promoter", k
    return f"{st}", k

# RBBP4 pair graph (p <= 0.05; every row of peak_edges) -----------------------------------------------------------------------------
E = pd.read_csv(X / "cobinding/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv", sep="\t"); E = E[E.pval <= 0.05]
PG = nx.Graph(); PG.add_edges_from(zip(E.peak1, E.peak2))
qmods = pd.read_csv(X / "cobinding/results/RBBP4_sweep_k50_i10000_s555/cliques.tsv", sep="\t")
qset = {frozenset(r.split(";")) for r in qmods.regions.astype(str)}
cand = {}   # frozenset of 3 regions -> set of types
def ok_trip(t):
    m = sorted(mid(iv(x)) for x in t); c = {iv(x)[0] for x in t}
    return len(c) == 1 and MIN_SPAN <= m[2] - m[0] <= MAX_SPAN and m[1] - m[0] >= MIN_SEP and m[2] - m[1] >= MIN_SEP
n_cl = 0
for cl in nx.find_cliques(PG):
    if len(cl) < 3: continue
    n_cl += 1
    for t in combinations(sorted(cl), 3):
        if ok_trip(t):
            k = frozenset(t); typ = "p-MCC" if len(cl) == 3 else f"subset of {len(cl)}-region p-MCC"
            if any(k <= q for q in qset): typ += " (in q-MCC)"
            cand.setdefault(k, set()).add(typ)
print(f"p-graph: {PG.number_of_nodes()} regions, {PG.number_of_edges()} pairs, {n_cl} maximal cliques >= 3; triplets after span/spacing filter: {len(cand)}", flush=True)
# loop-first triangles
RP = defaultdict(list)
for r in PG.nodes: c, s, e = iv(r); RP[c].append(((s + e) // 2, r))
RP = {c: sorted(v) for c, v in RP.items()}
anc = {}
for c in sorted({l[0] for l in LOOPS}):
    ms = sorted({m for cc, a, b in LOOPS if cc == c for m in (a, b)}); cur = None
    for m in ms:
        if cur is None or m - cur[-1] > 10000: cur = [m]
        else: cur.append(m)
        anc[(c, m)] = (c, cur[0])
AG = nx.Graph()
for c, a, b in LOOPS: AG.add_edge(anc[(c, a)], anc[(c, b)])
n_tri = n_tri_ok = 0
for u, v in AG.edges:
    for w in set(AG[u]) & set(AG[v]):
        tri = sorted((u, v, w), key=lambda z: z[1])
        if tri[0] != min(u, v, key=lambda z: z[1]) or (u, v) != tuple(sorted((u, v), key=lambda z: z[1])): pass
        key = tuple(tri)
        if key in anc: continue
        anc[key] = True; n_tri += 1
        c = tri[0][0]; peaks = []
        for _, a in tri:
            lst = [r for m, r in RP.get(c, []) if a - 10000 - 5000 <= m <= a + 15000]   # anchor start .. +10 kb bin, +-10 kb
            peaks.append(sorted(lst, key=lambda r: -PG.degree(r)))
        if not all(peaks): continue
        best = None
        for t in ((a, b, cc) for a in peaks[0][:5] for b in peaks[1][:5] for cc in peaks[2][:5]):
            npair = sum(PG.has_edge(x, y) for x, y in combinations(t, 2))
            if npair >= 2 and (best is None or npair > best[0]): best = (npair, t)
        if best and ok_trip(best[1]):
            n_tri_ok += 1; cand.setdefault(frozenset(best[1]), set()).add(f"loop-first ({best[0]}/3 RBBP4 pairs)")
print(f"loop triangles: {n_tri}; with RBBP4 peaks at all anchors and >= 2 RBBP4 pairs: {n_tri_ok}; total candidates: {len(cand)}", flush=True)

# score ---------------------------------------------------------------------------------------------------------------------------
rows = []
for k, typ in sorted(cand.items(), key=lambda kv: sorted(iv(x) for x in kv[0])[0]):
    pk = sorted([iv(x) for x in k], key=mid); c = pk[0][0]; ms = [mid(p) for p in pk]; span = ms[2] - ms[0]
    if any(in_bl(*p) for p in pk): continue
    H = H10 if span <= 1_000_000 else H25
    d = dict(chrom=c, A=f"{c}:{pk[0][1]}-{pk[0][2]}", B=f"{c}:{pk[1][1]}-{pk[1][2]}", C=f"{c}:{pk[2][1]}-{pk[2][2]}", span_kb=span / 1e3, res_kb=H.res // 1000,
             type="; ".join(sorted(typ)))
    vals = []
    for nm, (a, b) in (("AB", (0, 1)), ("BC", (1, 2)), ("AC", (0, 2))):
        HH = H
        dr, oe, ci, cj = HH.score_pair(c, ms[a], ms[b])
        if not np.isfinite(dr) and HH is H25: HH = H10; dr, oe, ci, cj = HH.score_pair(c, ms[a], ms[b])
        d.update({f"{nm}_donut": dr, f"{nm}_oe": oe, f"{nm}_ci": ci, f"{nm}_cj": cj, f"{nm}_res_kb": HH.res // 1000, f"{nm}_loop": on_loop(c, ms[a], ms[b]),
                  f"{nm}_rbbp4_pair": PG.has_edge(*sorted([x for x in k if iv(x) in (pk[a], pk[b])]))})
        vals += [dr, oe]
    d["weakest"] = np.nanmin(vals) if np.isfinite(vals).all() else np.nan
    d["min_donut"] = min(d["AB_donut"], d["BC_donut"], d["AC_donut"]); d["min_oe"] = min(d["AB_oe"], d["BC_oe"], d["AC_oe"])
    d["passes"] = bool(np.isfinite(vals).all() and min(vals) >= PASS); d["loops"] = sum(d[f"{p}_loop"] for p in ("AB", "BC", "AC"))
    rows.append(d)
T = pd.DataFrame(rows).sort_values("weakest", ascending=False)
for i, (_, t) in enumerate(T.head(60).iterrows()):
    for nm in "ABC":
        lab, kk = elem(*iv(t[nm])); T.loc[t.name, f"{nm}_elem"] = lab; T.loc[t.name, f"{nm}_tfs"] = kk
T.to_csv(OUT / "rbbp4_locus_candidates_v3.tsv", sep="\t", index=False)
print(f"scored {len(T)}; all pairs scorable {T.weakest.notna().sum()}; pass (every pair donut >= {PASS} and O/E >= {PASS}): {int(T.passes.sum())}", flush=True)
print("pass by type:", Counter(t for ty in T[T.passes].type for t in ty.split("; ")).most_common())
cols = ["type", "chrom", "A", "C", "span_kb", "res_kb", "AB_donut", "BC_donut", "AC_donut", "AB_oe", "BC_oe", "AC_oe", "weakest", "passes", "loops",
        "A_elem", "A_tfs", "B_elem", "B_tfs", "C_elem", "C_tfs"]
pd.set_option("display.width", 500); pd.set_option("display.max_colwidth", 40)
print(T[cols].head(15).round(2).to_string(index=False))

# contact sheet: raw (top) and log2 O/E (bottom) ---------------------------------------------------------------------------------
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
top = T.head(5)
fig, axs = plt.subplots(3, 5, figsize=(24, 11), gridspec_kw=dict(height_ratios=[3, 3, 1.1], hspace=0.12, wspace=0.18))
for k, (_, t) in enumerate(top.iterrows()):
    H = H10 if t.res_kb == 10 else H25; B, Eo, exp, Rw = H.load(t.chrom); r = H.res
    pk = [iv(t[nm]) for nm in "ABC"]; ms = [mid(p) for p in pk]; pad = max(80000, int(0.12 * (ms[2] - ms[0])))
    X0, X1 = (ms[0] - pad) // r, (ms[2] + pad) // r + 1; n = X1 - X0
    M = np.full((n, n), np.nan); O = np.full((n, n), np.nan)
    for i in range(n):
        for d in range(0, min(n - i, H.D + 1)):
            M[i, i + d] = B[d, X0 + i]; O[i, i + d] = Eo[d, X0 + i]
    span = (X1 - X0) * r / 1e3; xs = (X0 + np.arange(n + 1)) * r; XX, YY = np.meshgrid(xs, xs, indexing="ij")
    for row, (Z, cmap, kw) in enumerate(((np.log10(M + np.nanmedian(M[M > 0]) * 0.1), "Reds", {}), (np.log2(O), "RdBu_r", dict(vmin=-1.5, vmax=1.5)))):
        ax = axs[row, k]; Zm = np.ma.masked_invalid(np.ma.masked_where(np.tril(np.ones_like(Z), -1) > 0, Z))
        if not kw: kw = dict(vmin=np.nanpercentile(Z, 20), vmax=np.nanpercentile(Z, 98.5))
        pc = ax.pcolormesh((XX + YY) / 2e6, (YY - XX) / 2e3, Zm, cmap=cmap, shading="flat", rasterized=True, **kw)
        for nm in ("AB", "BC", "AC"):
            ci, cj = t[f"{nm}_ci"], t[f"{nm}_cj"]
            if ci is None or not np.isfinite(ci): continue
            ax.plot((ci + cj) / 2e6, (cj - ci) / 2e3, "o", mfc="none", mec="#222222", ms=12, mew=1.2)
            if row == 0: ax.text((ci + cj) / 2e6, (cj - ci) / 2e3 + span * 0.03, f"{nm[0]}–{nm[1]} {t[nm + '_donut']:.2f} / {t[nm + '_oe']:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xlim(X0 * r / 1e6, X1 * r / 1e6); ax.set_ylim(0, span / 2); ax.set_xticks([])
        ax.set_ylabel(("contacts (log10)" if row == 0 else "log2 O/E") + "\nDistance / 2 (kb)" if k == 0 else "")
        if k == 4: plt.colorbar(pc, ax=ax, fraction=0.04, pad=0.01)
    axs[0, k].set_title(f"#{k + 1} {t.chrom}:{ms[0] / 1e6:.2f}–{ms[2] / 1e6:.2f} Mb ({t.span_kb:.0f} kb, {t.res_kb} kb Hi-C)\n{t.type[:70]}\n"
                        f"weakest {t.weakest:.2f}; {'PASS' if t.passes else 'fail'}; {t.loops}/3 on a loop", fontsize=8, loc="left")
    a3 = axs[2, k]; a3.set_xlim(X0 * r / 1e6, X1 * r / 1e6); a3.set_ylim(0, 1); a3.set_yticks([]); a3.spines["left"].set_visible(False)
    for j, (nm, m) in enumerate(zip("ABC", ms)):
        a3.plot(m / 1e6, 0.9, "v", color="#222222", ms=7); a3.text(m / 1e6, 0.75 - 0.22 * j, f"{nm}: {t[nm + '_elem']}, {int(t[nm + '_tfs'])} TFs", ha="center", va="top", fontsize=7.5)
    a3.set_xlabel(f"{t.chrom} (Mb)")
fig.suptitle("RBBP4 Hi-C locus candidates v3 (WT HEK293T); labels = dot / donut  /  O/E; circles = best centre within ±1 bin; bottom triangles = log2 O/E",
             x=0.01, ha="left", fontsize=11)
fig.savefig(SHEET, dpi=120, bbox_inches="tight", facecolor="white")
print("sheet:", SHEET)
