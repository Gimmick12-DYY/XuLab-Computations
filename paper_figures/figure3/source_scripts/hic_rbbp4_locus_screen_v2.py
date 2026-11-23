"""Wider RBBP4 Hi-C locus screen (2026-11-23; v1 = hic_rbbp4_locus_screen.py, 18 candidates, all edges on a loop required).
Candidates: every RBBP4 Multi-region Coordinated Cluster (MCC; RBBP4_sweep_k50_i10000_s555/cliques.tsv, 3 and 4 regions), span between the
outermost element midpoints 100 kb - 1.5 Mb, no element in the hg38 blacklist v2 and every element in a balanced 10 kb bin. No loop requirement.
Hi-C: WT HEK293T (GSE216399), balanced, at 5, 10 and 25 kb. For each element pair: O/E = observed / mean of its diagonal within the window
(MCC span + 1 Mb each side); centre = 3x3 pixels; donut = Chebyshev distance 2-5 bins, centre row/column excluded (= v1 and
clique_triangle/02_score_dots.py); dot enrichment = centre O/E / donut O/E. Anchor tolerance: exact bins (0), or the best centre within
+-10 kb or +-20 kb of each element (rounded to whole bins; 0 bins at 25 kb for +-10 kb). Pairs closer than 6 bins at a resolution are not
scored there. MCC score at a setting = the weakest pair. Ranking = the 10 kb +-10 kb score (fixed setting). The best setting per MCC is also
reported but not used to rank: at 5 kb the max over +-2-4 bin shifts inflates every MCC (median 1.6 at 5 kb +-20 kb), so it is noise. The ETS2 locus of hic_ets2_locus is scored the same way.
Elements: promoter = GENCODE v46 protein-coding/lncRNA TSS within +-1 kb (gene named), else the 18-state ChromHMM class at the midpoint
(E/R/O as in pair_classes); n TFs = TFs with a Cicero peak within +-1 kb. Loops: Peakachu 0.7 (5 kb, 10 kb), both ends within +-30 kb.
Draws the top 5 at 10 kb (circles at the best +-10 kb centres) on one contact sheet.
  python hic_rbbp4_locus_screen_v2.py <outdir> <contact_sheet.png>   (Longleaf, hic env)"""
import glob, gzip, json, re, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import h5py, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
X = Path("/work/users/d/y/dyy12/XuLab"); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); SHEET = Path(sys.argv[2])
for f in glob.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): fm.fontManager.addfont(f)
MCOOL = X / "hic/work_wt293t/coolers-hg38/HEK293TWT-MboI-R1-filtered.mcool"; LD = X / "hic/work_wt293t/loops"
RESS, TOLS = (5000, 10000, 25000), (0, 10000, 20000)


class Hic:
    def __init__(self, path, res):
        self.f = h5py.File(path, "r"); self.g = self.f[f"resolutions/{res}"]; self.res = res; g = self.g
        self.names = [x.decode() for x in g["chroms/name"][:]]; self.co = g["indexes/chrom_offset"][:]; self.b1o = g["indexes/bin1_offset"]

    def fetch(self, c, s, e):
        i = self.names.index(c); c0, c1 = int(self.co[i]), int(self.co[i + 1]); s = max(0, s)
        i0 = c0 + s // self.res; i1 = min(c1, c0 + (e - 1) // self.res + 1); n = i1 - i0; g = self.g
        lo, hi = int(self.b1o[i0]), int(self.b1o[i1]); a = g["pixels/bin1_id"][lo:hi]; b = g["pixels/bin2_id"][lo:hi]; v = g["pixels/count"][lo:hi]
        m = b < i1; a, b, v = a[m] - i0, b[m] - i0, v[m]
        R = np.zeros((n, n)); R[a, b] = v; R[b, a] = v; w = g["bins/weight"][i0:i1]
        return R * np.outer(w, w), R, w, (s // self.res) * self.res


HICS = {r: Hic(MCOOL, r) for r in RESS}


def oe_window(c, lo, hi, r, pad=1_000_000):
    M, R, w, L0 = HICS[r].fetch(c, lo - pad, hi + pad); n = M.shape[0]
    M = np.where(np.isfinite(M), M, np.nan); bad = ~np.isfinite(w); M[bad, :] = np.nan; M[:, bad] = np.nan
    exp = np.array([np.nanmean(np.diagonal(M, d)) if np.isfinite(np.diagonal(M, d)).any() else np.nan for d in range(n)])
    ij = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    with np.errstate(all="ignore"): E = M / exp[ij]
    return E, R, L0


def dot(E, i, j):
    n = E.shape[0]
    if not (5 <= i < n - 5 and 5 <= j < n - 5) or j - i < 6: return np.nan
    cen = np.nanmean(E[i - 1:i + 2, j - 1:j + 2]); v = []
    for di in range(-5, 6):
        for dj in range(-5, 6):
            if max(abs(di), abs(dj)) < 2 or di == 0 or dj == 0: continue
            v.append(E[i + di, j + dj])
    bg = np.nanmean(v)
    return cen / bg if bg and np.isfinite(bg) and bg > 0 else np.nan


def score_mcc(c, mids):
    """{(res, tol): (min over pairs, {pair: (enrichment, best centre i, j in bp)})}"""
    out = {}
    for r in RESS:
        E, R, L0 = oe_window(c, min(mids), max(mids), r)
        b = [(m - L0) // r for m in mids]
        for tol in TOLS:
            k = int(round(tol / r)); per = {}
            for (a, ia), (bb, ib) in combinations(enumerate(b), 2):
                best = (np.nan, None, None)
                for di in range(-k, k + 1):
                    for dj in range(-k, k + 1):
                        v = dot(E, ia + di, ib + dj)
                        if np.isfinite(v) and not (best[0] >= v): best = (v, L0 + (ia + di) * r + r // 2, L0 + (ib + dj) * r + r // 2)
                per[(a, bb)] = best
            vals = [v[0] for v in per.values()]
            out[(r, tol)] = (min(vals) if all(np.isfinite(vals)) else np.nan, per)
    return out


# annotation ---------------------------------------------------------------------------------------------------------------
def iv(x): c, se = x.strip().split(":"); s, e = se.split("-"); return c, int(s), int(e)
mid = lambda p: (p[1] + p[2]) // 2
L = defaultdict(list)
for f in ("peakachu_5000.loops.0.7.bedpe", "peakachu_10000.loops.0.7.bedpe"):
    for ln in open(LD / f):
        x = ln.split()
        if x[0] == x[3]:
            a, b = sorted(((int(x[1]), int(x[2])), (int(x[4]), int(x[5])))); L[x[0]].append((*a, *b))
LA = {c: np.array(v) for c, v in L.items()}
def on_loop(p, q, slop=30000):
    if mid(p) > mid(q): p, q = q, p
    A = LA.get(p[0]); return A is not None and bool(((p[1] <= A[:, 1] + slop) & (p[2] >= A[:, 0] - slop) & (q[1] <= A[:, 3] + slop) & (q[2] >= A[:, 2] - slop)).any())
BL = defaultdict(list)
for ln in gzip.open(X / "downstream/cache/hg38-blacklist.v2.bed.gz", "rt"):
    f = ln.split("\t"); BL[f[0]].append((int(f[1]), int(f[2])))
def in_bl(c, s, e): return any(a < e and b > s for a, b in BL.get(c, ()))
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
E_ = {"EnhA1", "EnhA2", "EnhG1", "EnhG2", "EnhWk", "EnhBiv"}; R_ = {"ReprPC", "ReprPCWk", "Het", "ZNF/Rpts"}
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
def elem(c, s, e):
    g = G[(G.c == c) & (G.tss >= s - 1000) & (G.tss <= e + 1000)]
    k = PK[(PK.c == c) & (PK.sm >= s - 1000) & (PK.sm <= e + 1000)].tf.nunique()
    st = state(c, (s + e) // 2)
    if len(g):
        gg = g.sort_values("ty").g.iloc[0]; return "P", f"{gg} promoter", k, gg, st
    cl = "E" if st in E_ else "R" if st in R_ else "O"
    return cl, {"E": "enhancer", "R": "repressed", "O": "other"}[cl] + f" ({st})", k, "", st

# screen ---------------------------------------------------------------------------------------------------------------------
C = pd.read_csv(X / "cobinding/results/RBBP4_sweep_k50_i10000_s555/cliques.tsv", sep="\t")
w10 = HICS[10000]
def balanced(c, m):
    i = w10.names.index(c); b = int(w10.co[i]) + m // 10000; return np.isfinite(w10.g["bins/weight"][b])
rows, keep = [], {}
n_all = n_span = n_bl = 0
for cid, n, rr in zip(C.clique_id, C.n_regions, C.regions.astype(str)):
    n_all += 1
    pk = sorted([iv(x) for x in rr.split(";")], key=mid); c = pk[0][0]; mids = [mid(p) for p in pk]; span = mids[-1] - mids[0]
    if not (100_000 <= span <= 1_500_000): continue
    n_span += 1
    if any(in_bl(*p) for p in pk) or not all(balanced(c, m) for m in mids): n_bl += 1; continue
    S = score_mcc(c, mids)
    best = max(S, key=lambda k: (S[k][0] if np.isfinite(S[k][0]) else -1))
    d = dict(mcc=cid, n_regions=n, chrom=c, start=pk[0][1], end=pk[-1][2], span_kb=span / 1e3, best_res_kb=best[0] // 1000, best_tol_kb=best[1] // 1000,
             score=S[best][0], score_10kb_tol10=S[(10000, 10000)][0], score_10kb_exact=S[(10000, 0)][0])
    for (a, b), (v, _, _) in S[best][1].items(): d[f"{'ABCD'[a]}{'ABCD'[b]}"] = v
    d["pairs_on_loop"] = sum(on_loop(p, q) for p, q in combinations(pk, 2)); d["n_pairs"] = n * (n - 1) // 2
    cls = []
    for nm, p in zip("ABCD", pk):
        cl, lab, k, gg, st = elem(*p); d[f"{nm}_elem"] = lab; d[f"{nm}_tfs"] = k; d[f"{nm}_gene"] = gg; cls.append(cl)
    d["classes"] = "".join(cls); d["n_promoter"] = cls.count("P")
    for (r, t), (v, _) in S.items(): d[f"s_{r // 1000}kb_tol{t // 1000}"] = v
    rows.append(d); keep[cid] = (pk, S)
print(f"MCCs {n_all}; span 100 kb-1.5 Mb {n_span}; dropped blacklist/unbalanced {n_bl}; scored {len(rows)}", flush=True)
T = pd.DataFrame(rows).sort_values(["score_10kb_tol10", "s_10kb_tol20"], ascending=False)

# ETS2 reference, same scoring ----------------------------------------------------------------------------------------------
J = json.load(open(X / "paper_figures/figure3/source_data/hic_loci.json"))["ets2"]
ep = [iv(x) for x in J["regions"]]; ES = score_mcc(ep[0][0], [mid(p) for p in ep])
ref = {f"s_{r // 1000}kb_tol{t // 1000}": v for (r, t), (v, _) in ES.items()}
eb = max(ES, key=lambda k: (ES[k][0] if np.isfinite(ES[k][0]) else -1))
ref.update(mcc="ETS2 reference (hic_ets2_locus)", score=ES[eb][0], best_res_kb=eb[0] // 1000, best_tol_kb=eb[1] // 1000, score_10kb_tol10=ES[(10000, 10000)][0],
           score_10kb_exact=ES[(10000, 0)][0], span_kb=(mid(ep[-1]) - mid(ep[0])) / 1e3, chrom=ep[0][0])
for (a, b), (v, _, _) in ES[eb][1].items(): ref[f"{'ABCD'[a]}{'ABCD'[b]}"] = v
T = pd.concat([pd.DataFrame([ref]), T], ignore_index=True)
T.to_csv(OUT / "rbbp4_locus_candidates_v2.tsv", sep="\t", index=False)
cols = ["mcc", "n_regions", "chrom", "start", "end", "span_kb", "score", "best_res_kb", "best_tol_kb", "score_10kb_tol10", "score_10kb_exact", "AB", "AC", "BC", "AD", "BD", "CD",
        "pairs_on_loop", "n_pairs", "classes", "A_elem", "A_tfs", "B_elem", "B_tfs", "C_elem", "C_tfs", "D_elem", "D_tfs"]
pd.set_option("display.width", 400)
print(T[[c for c in cols if c in T]].head(16).round(2).to_string(index=False))
print("score >= ETS2 reference:", int((T.score.iloc[1:] >= T.score.iloc[0]).sum()), "; 10 kb +-10 kb >= ETS2:", int((T.score_10kb_tol10.iloc[1:] >= T.score_10kb_tol10.iloc[0]).sum()))
print("settings, RBBP4 MCC distribution:"); print(T.iloc[1:][[c for c in T if c.startswith("s_")]].describe().round(2).to_string())
print("with >= 1 promoter, top 5:"); print(T.iloc[1:][T.iloc[1:].n_promoter >= 1][[c for c in cols if c in T]].head(5).round(2).to_string(index=False))

# contact sheet: top 5 at 10 kb, circles at the best +-20 kb centres -----------------------------------------------------------
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
top = T.iloc[1:6]
fig, axs = plt.subplots(2, 5, figsize=(22, 7.5), gridspec_kw=dict(height_ratios=[3.2, 1], hspace=0.08, wspace=0.18))
for k, (_, t) in enumerate(top.iterrows()):
    pk, S = keep[t.mcc]; c = t.chrom; r0 = 10000; mids = [mid(p) for p in pk]
    X0, X1 = (mids[0] - 80000) // r0 * r0, (mids[-1] + 80000) // r0 * r0 + r0; span = (X1 - X0) / 1e3
    M, _, w, L0 = HICS[r0].fetch(c, X0, X1); n = M.shape[0]; Z = np.log10(M + np.nanmedian(M[M > 0]) * 0.1)
    ax = axs[0, k]; xs = L0 + np.arange(n + 1) * r0; XX, YY = np.meshgrid(xs, xs, indexing="ij")
    Zm = np.ma.masked_invalid(np.ma.masked_where(np.tril(np.ones_like(Z), -1) > 0, Z))
    ax.pcolormesh((XX + YY) / 2e6, (YY - XX) / 2e3, Zm, cmap="Reds", vmin=np.nanpercentile(Z, 20), vmax=np.nanpercentile(Z, 98.5), shading="flat", rasterized=True)
    for (a, b), (v, ci, cj) in S[(10000, 10000)][1].items():
        if ci is None: continue
        ax.plot((ci + cj) / 2e6, (cj - ci) / 2e3, "o", mfc="none", mec="#222222", ms=13, mew=1.3)
        ax.text((ci + cj) / 2e6, (cj - ci) / 2e3 + span * 0.03, f"{'ABCD'[a]}–{'ABCD'[b]} {v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlim(X0 / 1e6, X1 / 1e6); ax.set_ylim(0, span / 2); ax.set_xticks([]); ax.set_ylabel("Distance / 2 (kb)" if k == 0 else "")
    ax.set_title(f"#{k + 1} {t.mcc} ({t.n_regions} regions), {c}:{t.start / 1e6:.2f}–{t.end / 1e6:.2f} Mb\nweakest dot 10 kb ±10 kb {t.score_10kb_tol10:.2f} "
                 f"(ETS2 example {T.score_10kb_tol10.iloc[0]:.2f}); {t.pairs_on_loop}/{t.n_pairs} pairs on a loop", fontsize=8.5, loc="left")
    a2 = axs[1, k]; a2.set_xlim(X0 / 1e6, X1 / 1e6); a2.set_ylim(0, 1); a2.set_yticks([]); a2.spines["left"].set_visible(False)
    for j, (nm, m) in enumerate(zip("ABCD", mids)):
        a2.plot(m / 1e6, 0.85, "v", color="#222222", ms=7)
        a2.text(m / 1e6, 0.72 - 0.17 * j, f"{nm}: {t[f'{nm}_elem']}, {int(t[f'{nm}_tfs'])} TFs", ha="center", va="top", fontsize=7.5)
    a2.set_xlabel(f"{c} (Mb)")
fig.suptitle("RBBP4 MCC Hi-C locus candidates (WT HEK293T, 10 kb); circles = best centre within ±10 kb; numbers = dot / donut", x=0.01, ha="left", fontsize=11)
fig.savefig(SHEET, dpi=130, bbox_inches="tight", facecolor="white")
print("sheet:", SHEET)
