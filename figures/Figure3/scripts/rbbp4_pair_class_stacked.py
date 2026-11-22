"""RBBP4 pair-class panel in the HiChIP/ChIA-PET convention (2026-11-22): 100% stacked bars per set of interactions.
Rows: RBBP4 pairs (first-step Cicero, p <= 0.05, ends >25 kb apart, unique; RBBP4_sweep_k50_i10000_s555), RBBP4 module edges (region pairs
inside one higher-order module, >25 kb, unique), All TFs (the same pairs for every TF with Cicero output, ourrun_k50 + RBBP4, each pair
counted once per TF), Hi-C loops (Peakachu score >= 0.7, WT HEK293T, 5 kb + 10 kb loop lists, each listed loop once).
Element class of each end (panel e rule): promoter = RefSeq Select TSS within the region +-1 kb; otherwise the HEK293T 18-state ChromHMM
state at the region midpoint: enhancer (EnhA1/A2/G1/G2/Wk/Biv), repressive (ReprPC/ReprPCWk/Het/ZNF-Rpts), other. Classes: P-P, P-E,
E-E, P-R, R-R, Other (all remaining combinations).
  python rbbp4_pair_class_stacked.py <outdir>   (Longleaf, hic env; inputs in checks/pair_class_inputs)"""
import glob, gzip, json, re, sys
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
X = Path("/work/users/d/y/dyy12/XuLab"); F3 = X / "figures/Figure3"; CI = F3 / "checks/pair_class_inputs"; OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
for f in glob.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Arial", "font.size": 6, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
GROUP = {"EnhA1": "E", "EnhA2": "E", "EnhG1": "E", "EnhG2": "E", "EnhWk": "E", "EnhBiv": "E", "ReprPC": "R", "ReprPCWk": "R", "Het": "R", "ZNF/Rpts": "R"}
TSS = {}
for f in glob.glob(str(CI / "rs/*.json")):
    v = json.load(open(f))["ncbiRefSeqSelect"]
    if v: TSS[Path(f).stem] = np.sort([(g["txStart"] if g["strand"] == "+" else g["txEnd"]) for g in v])
H = defaultdict(list)
for ln in gzip.open(CI / "HEK293T_chromHMM18.bed.gz", "rt"):
    f = ln.split("\t"); H[f[0]].append((int(f[1]), int(f[2]), f[3]))
HS = {c: (np.array([a for a, _, _ in sorted(v)]), np.array([b for _, b, _ in sorted(v)]), [s for _, _, s in sorted(v)]) for c, v in H.items()}
memo = {}
def ecls(c, s, e):
    k = (c, s, e)
    if k in memo: return memo[k]
    r = "O"
    if c in TSS:
        p = TSS[c]; i = np.searchsorted(p, s - 1000)
        if i < len(p) and p[i] <= e + 1000: r = "P"
    if r == "O" and c in HS:
        st, en, lab = HS[c]; m = (s + e) // 2; i = np.searchsorted(st, m, side="right") - 1
        if i >= 0 and en[i] > m: r = GROUP.get(lab[i], "O")
    memo[k] = r; return r
KEEP = {"PP": "P–P", "EP": "P–E", "EE": "E–E", "PR": "P–R", "RR": "R–R"}
def pcls(p, q): return KEEP.get("".join(sorted([ecls(*p), ecls(*q)])), "Other")
def iv(r): c, se = r.strip().split(":"); s, e = se.split("-"); return c, int(s), int(e)
mid = lambda p: (p[1] + p[2]) // 2
def tf_pairs(f):
    E = pd.read_csv(f, sep="\t", usecols=["peak1", "peak2", "pval"]); E = E[E.pval <= 0.05]; out = {}
    for a, b in zip(E.peak1, E.peak2):
        p, q = iv(a), iv(b)
        if p[0] == q[0] and abs(mid(p) - mid(q)) > 25000: out[tuple(sorted([a, b]))] = (p, q)
    return out
rows = {}
R4 = tf_pairs(X / "cobinding/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv"); rows["RBBP4 pairs"] = Counter(pcls(p, q) for p, q in R4.values())
C = pd.read_csv(X / "cobinding/results/RBBP4_sweep_k50_i10000_s555/cliques.tsv", sep="\t"); ME = {}
for rr in C.regions.astype(str):
    for x, y in combinations([z.strip() for z in rr.split(";")], 2):
        p, q = iv(x), iv(y)
        if abs(mid(p) - mid(q)) > 25000: ME[tuple(sorted([x, y]))] = (p, q)
rows["RBBP4 module edges"] = Counter(pcls(p, q) for p, q in ME.values())
allc = Counter(rows["RBBP4 pairs"]); ntf = 1
for f in sorted(glob.glob(str(X / "cobinding/work/ourrun_k50/*/*.peak_edges.tsv"))):
    if Path(f).name.startswith("RBBP4."): continue
    allc.update(pcls(p, q) for p, q in tf_pairs(f).values()); ntf += 1
rows["All TFs"] = allc
lc = Counter()
for f in ("peakachu_5000.loops.0.7.bedpe", "peakachu_10000.loops.0.7.bedpe"):
    for ln in open(X / "hic/work_wt293t/loops" / f):
        x = ln.split()
        if x[0] == x[3]: lc[pcls((x[0], int(x[1]), int(x[2])), (x[3], int(x[4]), int(x[5])))] += 1
rows["Hi-C loops"] = lc
ORDER = ["P–P", "P–E", "E–E", "P–R", "R–R", "Other"]; COL = ["#1F3F8F", "#7A9CCB", "#E8821E", "#6A51A3", "#55606E", "#D9D9D9"]
T = pd.DataFrame(rows).T.reindex(columns=ORDER).fillna(0).astype(int); T["n"] = T.sum(1)
P = T[ORDER].div(T.n, axis=0) * 100; T.join(P.add_suffix(" %").round(2)).to_csv(OUT / "rbbp4_pair_class_stacked.tsv", sep="\t")
print("TFs pooled:", ntf); print(T.join(P.round(1).add_suffix("%")).to_string())
MM = 1 / 25.4; fig = plt.figure(figsize=(65 * MM, 50 * MM)); ax = fig.add_axes([0.27, 0.24, 0.66, 0.72])
for i, (nm, r) in enumerate(P.iterrows()):
    left = 0
    for k, c in zip(ORDER, COL):
        f = r[k] / 100; ax.barh(i, f, 0.72, left=left, color=c, lw=0.6, edgecolor="white")
        if f >= 0.06: ax.text(left + f / 2, i, f"{r[k]:.0f}%", ha="center", va="center", fontsize=6, color="white" if k in ("P–P", "P–R", "R–R") else "#1A1A1A")
        left += f
ax.set_yticks(range(len(P))); ax.set_yticklabels([""] * len(P)); ax.set_ylim(len(P) - 0.5, -0.5); ax.tick_params(axis="y", length=0)
for i, nm in enumerate(P.index):
    ax.text(-0.03, i - 0.12, {"RBBP4 module edges": "Module edges"}.get(nm, nm), ha="right", va="center", fontsize=6.5, color="#1A1A1A", transform=ax.get_yaxis_transform())
    ax.text(-0.03, i + 0.24, f"n = {T.n[nm]:,}", ha="right", va="center", fontsize=6, color="#7F7F7F", transform=ax.get_yaxis_transform())
ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(["0", "50", "100"]); ax.set_xlabel("% of interactions", fontsize=6, labelpad=1); ax.tick_params(axis="x", length=2, pad=1)
for sp in ("left", "top", "right"): ax.spines[sp].set_visible(False)
hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COL]
fig.legend(hs, ORDER, loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=6, fontsize=6, frameon=False, handlelength=0.8, handleheight=0.8, handletextpad=0.25, columnspacing=0.75, borderaxespad=0.2)
for ext in ("png", "svg", "pdf"): fig.savefig(OUT / f"rbbp4_square_stacked.{ext}", dpi=600 if ext == "png" else None)
