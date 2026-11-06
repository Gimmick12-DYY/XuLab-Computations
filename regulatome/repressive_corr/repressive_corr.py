# TF x TF correlation over ChromHMM element classes of elements_v2 (repressive: polycomb, znf_rpts, het; comparisons:
# bivalent, active = promoter + distal_active), built like the existing ChromHMM-element matrices (tf_complex):
#   signal = raw reads in the element / TF cell count (Dr. Wang's style), Pearson across the class's elements with any read,
#   then SpQN (tf_complex spqn, features ordered by cell count).
# Block statistic = mean pairwise percentile of the members within the off-diagonal distribution of that matrix.
# Null = depth-matched random sets (each member replaced by a non-member TF within +/-4 cell-count ranks), 5,000 draws.
# Enrichment (obs/exp per class) is reported separately from M1 v2 (frac_reads_by_class / enrich tables).
import json, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/tf_complex/scripts"); sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream")
from compartment_rpkm_correlation import spqn

X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/regulatome/repressive_corr"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.5, "svg.fonttype": "none",
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "savefig.dpi": 600,
                     "text.color": "#1A1A1A", "axes.labelcolor": "#1A1A1A", "xtick.color": "#1A1A1A", "ytick.color": "#1A1A1A",
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5})
INK, GREY = "#1A1A1A", "#8C8C8C"
MOD_COL = {"PRC2": "#1F3F8F", "NuRD": "#EB6834", "MAZ_ZNF143": "#EDC948", "cohesin_CTCF": "#6250D6", "HP1_ChAHP": "#FF9DA7",
           "MEF2": "#8C564B", "G9a_WIZ": "#B07AA1"}
LIB_COL = {"high": "#3F3F3F", "low": "#BDBDBD"}
BLOCKS = {"PRC2": ["EZH2", "MTF2", "PCGF6", "RBBP4", "RBBP7"], "PRC2_core_EZH2_MTF2": ["EZH2", "MTF2"],
          "PRC2_noRBBP": ["EZH2", "MTF2", "PCGF6"], "HP1_SETDB1_ChAHP": ["HP1", "SETDB1", "ADNP", "CHAMP1"],
          "NuRD": ["MBD3", "RBBP4", "RBBP7"]}
CLASSES = {"polycomb": ["polycomb"], "znf_rpts": ["znf_rpts"], "het": ["het"], "bivalent": ["bivalent"],
           "active": ["promoter", "distal_active"]}

E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None,
                names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"])
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
cells = np.array([json.load(open(f"{X}/regulatome/m1_binding/v2/counts/{t}.json"))["cells"] for t in tfs], float)
M = np.column_stack([np.load(f"{X}/regulatome/m1_binding/v2/counts/{t}.npz")["raw"].astype(np.float32) / c for t, c in zip(tfs, cells)])
F = pd.read_csv(f"{X}/regulatome/m1_binding/v2/frac_reads_by_class.tsv", sep="\t", index_col=0)
lib = (F.all_elements > 0.22).map({True: "high", False: "low"}).reindex(tfs)
rank = pd.Series(cells, index=tfs).rank().values
rng = np.random.default_rng(20261029)
mods = {k: v for k, v in BLOCKS.items() if k in ("PRC2", "NuRD", "HP1_SETDB1_ChAHP")}
fixed = []
for k in ["PRC2", "NuRD", "HP1_SETDB1_ChAHP"]:
    fixed += [t for t in BLOCKS[k] if t not in fixed]
for t in ["MAZ", "ZNF143", "STAG2", "CTCF", "WIZ", "ZNF644", "MEF2A", "MEF2D"]:
    if t not in fixed: fixed.append(t)
fixed += sorted([t for t in tfs if t not in fixed], key=lambda t: (lib[t], t))

def block_stat(P, mem):
    ix = [tfs.index(t) for t in mem]
    return np.mean([P[i, j] for a, i in enumerate(ix) for j in ix[a + 1:]])

def null_dist(P, mem, n=5000):
    ix = [tfs.index(t) for t in mem]; out = np.empty(n)
    others = [i for i in range(len(tfs)) if tfs[i] not in mem]
    for k in range(n):
        pick = []
        for i in ix:
            cand = [j for j in others if abs(rank[j] - rank[i]) <= 4 and j not in pick]
            pick.append(rng.choice(cand))
        out[k] = np.mean([P[a, b] for x, a in enumerate(pick) for b in pick[x + 1:]])
    return out

def pct_matrix(R):
    off = ~np.eye(len(R), dtype=bool); v = np.sort(R[off])
    P = np.searchsorted(v, R, side="right") / len(v); np.fill_diagonal(P, np.nan); return P

def heat(R, order, title, path, cmap="RdBu_r", vmin=-1, vmax=1, label="Pearson r"):
    o = [tfs.index(t) for t in order]; A = R[np.ix_(o, o)]
    fig = plt.figure(figsize=(6.2, 6.0)); ax = fig.add_axes([0.13, 0.12, 0.74, 0.74])
    im = ax.imshow(A, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=4); ax.set_yticklabels(order, fontsize=4)
    ax.tick_params(length=1, width=0.3, pad=1)
    for spn in ax.spines.values(): spn.set_linewidth(0.5)
    for i, t in enumerate(order):  # library strip and module strip
        ax.add_patch(plt.Rectangle((len(order) + 0.2, i - 0.5), 0.9, 1, color=LIB_COL[lib[t]], clip_on=False))
        m = next((k for k, v in mods.items() if t in v), None)
        ax.add_patch(plt.Rectangle((i - 0.5, -2.3), 1, 0.9, color=MOD_COL.get({"HP1_SETDB1_ChAHP": "HP1_ChAHP"}.get(m, m), "#BDBDBD") if m else "#FFFFFF", clip_on=False))
    cax = fig.add_axes([0.9, 0.55, 0.015, 0.25]); cb = fig.colorbar(im, cax=cax); cb.set_label(label, fontsize=6)
    cb.ax.tick_params(labelsize=5, width=0.3, length=1.5); cb.outline.set_linewidth(0.3)
    ax.set_title(title, fontsize=7, pad=10)
    fig.savefig(path, format="svg"); plt.close(fig)

rows = []
for cname, cl in CLASSES.items():
    sel = E["cls"].isin(cl).values
    Xc = M[sel]; Xc = Xc[Xc.sum(1) > 0]
    R = np.nan_to_num(np.corrcoef(Xc, rowvar=False)); Rn = spqn(R, cells)
    pd.DataFrame(R, index=tfs, columns=tfs).to_csv(f"{O}/pearson_{cname}.tsv", sep="\t", float_format="%.4f")
    pd.DataFrame(Rn, index=tfs, columns=tfs).to_csv(f"{O}/spqn_{cname}.tsv", sep="\t", float_format="%.4f")
    for kind, Rm in (("pearson", R), ("spqn", Rn)):
        P = pct_matrix(Rm)
        for b, mem in BLOCKS.items():
            obs = block_stat(P, mem); nd = null_dist(P, mem, 2000 if kind == "pearson" else 5000)
            rows.append(dict(cls=cname, matrix=kind, block=b, members=",".join(mem), n_elements=int(Xc.shape[0]),
                             mean_pair_pct=obs, null_pct=float((nd < obs).mean()), null_median=float(np.median(nd)),
                             null_p95=float(np.percentile(nd, 95)),
                             mean_r=float(np.mean([Rm[tfs.index(a), tfs.index(c)] for x, a in enumerate(mem) for c in mem[x + 1:]]))))
        lk = linkage(1 - Rm[np.triu_indices(len(tfs), 1)], "average")
        heat(Rm, [tfs[i] for i in leaves_list(lk)], f"{cname} elements (n = {Xc.shape[0]:,}), {kind}, clustered",
             f"{O}/heat_{cname}_{kind}_clustered.svg")
        heat(Rm, fixed, f"{cname} elements (n = {Xc.shape[0]:,}), {kind}, fixed order", f"{O}/heat_{cname}_{kind}_fixed.svg")
    print(cname, Xc.shape[0], "elements", flush=True)

T = pd.DataFrame(rows)
enr = pd.read_csv(f"{X}/regulatome/m1_binding/v2/enrich_reads_by_class.tsv", sep="\t", index_col=0)
T["members_enrichment_vs_panel"] = [", ".join(f"{t} {enr.loc[t, c]:.2f}x" for t in m.split(","))
                                    if c in enr.columns else "." for m, c in zip(T.members, T.cls)]
T.to_csv(f"{O}/block_stats.tsv", sep="\t", index=False, float_format="%.3f")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(T[T.matrix == "spqn"].pivot(index="block", columns="cls", values="null_pct").round(3).to_string())
print(T[T.matrix == "spqn"].pivot(index="block", columns="cls", values="mean_pair_pct").round(3).to_string())
print(T[T.matrix == "pearson"].pivot(index="block", columns="cls", values="null_pct").round(3).to_string())
print(T[T.matrix == "pearson"].pivot(index="block", columns="cls", values="mean_r").round(3).to_string())
print(T[(T.matrix == "spqn") & T.block.isin(["PRC2", "HP1_SETDB1_ChAHP"])][["block", "cls", "members_enrichment_vs_panel"]].to_string(index=False))
