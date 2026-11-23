"""Pair-class panel in the HiChIP/ChIA-PET convention (2026-11-23; was rbbp4_pair_class_stacked.py): 100% stacked bars per set of interactions.
Rows: RBBP4 pairs (first-step Cicero, p <= 0.05, ends >25 kb apart, unique; RBBP4_sweep_k50_i10000_s555), RBBP4 module edges (region pairs
inside one higher-order module, >25 kb, unique), ETS2 pairs and module edges (same rules; ourrun_k50, Cicero k = 50, 2026-10-09), All TFs (the
same pairs for every TF with Cicero output, ourrun_k50 + RBBP4, each pair counted once per TF), Hi-C loops (Peakachu score >= 0.7, WT HEK293T,
5 kb + 10 kb loop lists, each listed loop once).
Element class of each end: promoter (P) = RefSeq Select TSS within the region +-1 kb; otherwise the HEK293T 18-state ChromHMM state at the
region midpoint (now: any state within the peak +-1 kb, see PAD below): enhancer E (EnhA1/A2/G1/G2/Wk/Biv), repressor R (ReprPC/ReprPCWk/Het/ZNF/Rpts), other O (TssA/TssFlnk/TssFlnkU/TssFlnkD/
TssBiv outside TSS +-1 kb, Tx, TxWk, Quies, or no state). Categories: P-P, P-E, P-R, P-O, E-E, R-R, Rest (E-R, E-O, R-O, O-O).
Element rule (env PAD = bp >= 0, default 1000 = the panel since 2026-11-23; "mid" = the earlier midpoint rule): non-promoter elements are E if any E segment overlaps the element +-PAD,
else R if any R segment does, else O (PRIO=RE swaps E and R). Hi-C loop anchors always use the midpoint rule. PLOT=0 skips the figure.
  python pair_classes.py <outdir>   (Longleaf, hic env; inputs in checks/pair_class_inputs)"""
import glob, gzip, json, os, re, sys
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
GROUP = {"EnhA1": "E", "EnhA2": "E", "EnhG1": "E", "EnhG2": "E", "EnhWk": "E", "EnhBiv": "E", "ReprPC": "R", "ReprPCWk": "R", "Het": "R", "ZNF/Rpts": "R",
         "TssA": "O", "TssFlnk": "O", "TssFlnkU": "O", "TssFlnkD": "O", "TssBiv": "O", "Tx": "O", "TxWk": "O", "Quies": "O"}
UNKNOWN = Counter()
TSS = {}
for f in glob.glob(str(CI / "rs/*.json")):
    v = json.load(open(f))["ncbiRefSeqSelect"]
    if v: TSS[Path(f).stem] = np.sort([(g["txStart"] if g["strand"] == "+" else g["txEnd"]) for g in v])
H = defaultdict(list)
for ln in gzip.open(CI / "HEK293T_chromHMM18.bed.gz", "rt"):
    f = ln.split("\t"); H[f[0]].append((int(f[1]), int(f[2]), f[3]))
HS = {c: (np.array([a for a, _, _ in sorted(v)]), np.array([b for _, b, _ in sorted(v)]), [s for _, _, s in sorted(v)]) for c, v in H.items()}
memo = {}
PAD = os.environ.get("PAD", "1000")   # panel rule since 2026-11-23: peak +-1 kb, enhancer first
PRIO = os.environ.get("PRIO", "ER"); MIDPOINT = [False]
def ecls(c, s, e):
    k = (c, s, e, MIDPOINT[0])
    if k in memo: return memo[k]
    r = "O"
    if c in TSS:
        p = TSS[c]; i = np.searchsorted(p, s - 1000)
        if i < len(p) and p[i] <= e + 1000: r = "P"
    if r == "O" and c in HS:
        st, en, lab = HS[c]
        if PAD == "mid" or MIDPOINT[0]:
            m = (s + e) // 2; i = np.searchsorted(st, m, side="right") - 1
            if i >= 0 and en[i] > m:
                r = GROUP.get(lab[i], "O")
                if lab[i] not in GROUP: UNKNOWN[lab[i]] += 1
        else:
            pad = int(PAD); i0 = np.searchsorted(en, s - pad, side="right"); i1 = np.searchsorted(st, e + pad, side="left")
            got = {GROUP.get(x, "O") for x in lab[i0:i1]}
            for g in (("E", "R") if PRIO == "ER" else ("R", "E")):
                if g in got: r = g; break
    memo[k] = r; return r
ELEM = {}
def elem_share(name, pairs):
    regs = {p for pq in pairs for p in pq}
    ELEM[name] = Counter(ecls(*p) for p in regs)
KEEP = {"PP": "P–P", "EP": "P–E", "PR": "P–R", "OP": "P–O", "EE": "E–E", "RR": "R–R"}
def raw(p, q): return "–".join(sorted([ecls(*p), ecls(*q)], key="PERO".index))
def pcls(p, q): return KEEP.get("".join(sorted([ecls(*p), ecls(*q)])), "Rest")
RAW = {}
def iv(r): c, se = r.strip().split(":"); s, e = se.split("-"); return c, int(s), int(e)
mid = lambda p: (p[1] + p[2]) // 2
def tf_pairs(f):
    E = pd.read_csv(f, sep="\t", usecols=["peak1", "peak2", "pval"]); E = E[E.pval <= 0.05]; out = {}
    for a, b in zip(E.peak1, E.peak2):
        p, q = iv(a), iv(b)
        if p[0] == q[0] and abs(mid(p) - mid(q)) > 25000: out[tuple(sorted([a, b]))] = (p, q)
    return out
def module_edges(f):
    C = pd.read_csv(f, sep="\t"); ME = {}
    for rr in C.regions.astype(str):
        for x, y in combinations([z.strip() for z in rr.split(";")], 2):
            p, q = iv(x), iv(y)
            if abs(mid(p) - mid(q)) > 25000: ME[tuple(sorted([x, y]))] = (p, q)
    return ME
rows = {}
R4 = tf_pairs(X / "cobinding/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv"); rows["RBBP4 pairs"] = Counter(pcls(p, q) for p, q in R4.values()); RAW["RBBP4 pairs"] = Counter(raw(p, q) for p, q in R4.values()); elem_share("RBBP4 pairs", R4.values())
for nm, d in (("RBBP4 MCC edges", module_edges(X / "cobinding/results/RBBP4_sweep_k50_i10000_s555/cliques.tsv")),
              ("ETS2 pairs", tf_pairs(X / "cobinding/work/ourrun_k50/ETS2/ETS2.peak_edges.tsv")),
              ("ETS2 MCC edges", module_edges(X / "cobinding/results/ourrun_k50/ETS2/cliques.tsv"))):
    rows[nm] = Counter(pcls(p, q) for p, q in d.values()); RAW[nm] = Counter(raw(p, q) for p, q in d.values()); elem_share(nm, d.values())
allc = Counter(rows["RBBP4 pairs"]); ntf = 1; alle = Counter(ELEM["RBBP4 pairs"])
for f in sorted(glob.glob(str(X / "cobinding/work/ourrun_k50/*/*.peak_edges.tsv"))):
    if Path(f).name.startswith("RBBP4."): continue
    d = tf_pairs(f); allc.update(pcls(p, q) for p, q in d.values()); ntf += 1
    elem_share("_tmp", d.values()); alle.update(ELEM.pop("_tmp"))
ELEM["All TFs pooled"] = alle
rows["All TFs pooled"] = allc
lc = Counter(); MIDPOINT[0] = True   # Hi-C anchors (5-10 kb bins): midpoint rule in every variant
for f in ("peakachu_5000.loops.0.7.bedpe", "peakachu_10000.loops.0.7.bedpe"):
    for ln in open(X / "hic/work_wt293t/loops" / f):
        x = ln.split()
        if x[0] == x[3]: lc[pcls((x[0], int(x[1]), int(x[2])), (x[3], int(x[4]), int(x[5])))] += 1
rows["Hi-C loops"] = lc; MIDPOINT[0] = False
ORDER = ["P–P", "P–E", "P–R", "P–O", "E–E", "R–R", "Rest"]; COL = ["#1F3F8F", "#7A9CCB", "#6A51A3", "#A9C0D6", "#E8821E", "#55606E", "#DCDCDC"]
DARK = ("P–P", "P–R", "R–R")
T = pd.DataFrame(rows).T.reindex(columns=ORDER).fillna(0).astype(int); T["n"] = T.sum(1)
P = T[ORDER].div(T.n, axis=0) * 100; T.join(P.add_suffix(" %").round(2)).to_csv(OUT / "pair_classes.tsv", sep="\t")
pd.DataFrame(RAW).T.fillna(0).astype(int).to_csv(OUT / "pair_classes_all_combinations.tsv", sep="\t")
pd.DataFrame(ELEM).T.reindex(columns=["P", "E", "R", "O"]).fillna(0).astype(int).to_csv(OUT / "pair_classes_element_classes.tsv", sep="\t")
print("TFs pooled:", ntf); print(pd.DataFrame(RAW).T.fillna(0).astype(int).to_string()); print("states not in the table:", dict(UNKNOWN) or "none"); print(T.join(P.round(1).add_suffix("%")).to_string())
MM = 1 / 25.4; W, Hh = 84, 66; fig = plt.figure(figsize=(W * MM, Hh * MM)); ax = fig.add_axes([28 / W, 17 / Hh, 54 / W, 47 / Hh])
for i, (nm, r) in enumerate(P.iterrows()):
    left = 0
    for k, c in zip(ORDER, COL):
        f = r[k] / 100; ax.barh(i, f, 0.72, left=left, color=c, lw=0.6, edgecolor="white")
        if f >= 0.06: ax.text(left + f / 2, i, f"{r[k]:.0f}%", ha="center", va="center", fontsize=6, color="white" if k in DARK else "#1A1A1A")
        left += f
ax.set_yticks(range(len(P))); ax.set_yticklabels([""] * len(P)); ax.set_ylim(len(P) - 0.5, -0.5); ax.tick_params(axis="y", length=0)
for i, nm in enumerate(P.index):
    ax.text(-0.03, i - 0.14, nm, ha="right", va="center", fontsize=6.5, color="#1A1A1A", transform=ax.get_yaxis_transform())
    ax.text(-0.03, i + 0.24, f"n = {T.n[nm]:,}", ha="right", va="center", fontsize=6, color="#7F7F7F", transform=ax.get_yaxis_transform())
ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(["0", "50", "100"]); ax.set_xlabel("% of interactions", fontsize=6, labelpad=1); ax.tick_params(axis="x", length=2, pad=1)
for sp in ("left", "top", "right"): ax.spines[sp].set_visible(False)
hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COL]
fig.legend(hs, ORDER, loc="lower center", bbox_to_anchor=(0.5, 3.2 / Hh), ncol=7, fontsize=6, frameon=False, handlelength=0.8, handleheight=0.8, handletextpad=0.25, columnspacing=0.7, borderaxespad=0.2)
fig.text(0.5, 1.0 / Hh, "MCC, Multi-region Coordinated Cluster", ha="center", va="bottom", fontsize=6, color="#7F7F7F")
for ext in (("png", "svg", "pdf") if os.environ.get("PLOT", "1") != "0" else ()): fig.savefig(OUT / f"pair_classes.{ext}", dpi=600 if ext == "png" else None)
