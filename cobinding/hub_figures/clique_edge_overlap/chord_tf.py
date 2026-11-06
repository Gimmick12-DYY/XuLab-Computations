"""Chord diagrams from shared clique edges (all cliques, blacklist removed).
A: TF-TF chord, ribbon width = identical clique edges shared by the two TFs (pairs with >= MINE).
B: GOChord-style, TFs (left) linked to the most shared clique edges (right, one per locus)."""
import math, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path as MPath
import numpy as np, pandas as pd
plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})
OUT = sys.argv[1]
MINE = 5
sym = pd.read_csv("chord/ensg_symbol.tsv", sep="\t", header=None, index_col=0)[1].to_dict()
X = pd.read_csv("edges/tf_x_tf_shared_clique_edges.tsv", sep="\t", index_col=0)
np.fill_diagonal(X.values, 0)
pol = lambda t, r=1.0: (r * math.cos(t), r * math.sin(t))

def ribbon(ax, a1, a2, b1, b2, col, alpha=0.55, r=0.94):
    verts = [pol(a1, r), pol((a1 + a2) / 2, r), pol(a2, r), (0, 0), pol(b2, r), pol((b1 + b2) / 2, r), pol(b1, r), (0, 0), pol(a1, r)]
    codes = [MPath.MOVETO] + [MPath.CURVE3] * 8
    ax.add_patch(PathPatch(MPath(verts, codes), fc=col, ec="none", alpha=alpha))

# ---- A: TF-TF chord ----
pairs = [(a, b, int(X.loc[a, b])) for i, a in enumerate(X.index) for b in X.index[i + 1:] if X.loc[a, b] >= MINE]
tfs = sorted({t for a, b, _ in pairs for t in (a, b)})
tot = {t: sum(n for a, b, n in pairs if t in (a, b)) for t in tfs}
# order TFs by hierarchical clustering on shared-edge counts so groups sit together
from scipy.cluster.hierarchy import linkage, leaves_list
S = np.log1p(X.loc[tfs, tfs].values.astype(float))
order = [tfs[i] for i in leaves_list(linkage(S, "average", metric="correlation"))]
gap = math.radians(1.2)
span = 2 * math.pi - gap * len(order)
T = sum(tot.values())
pos, cur = {}, math.pi / 2
for t in order:
    w = span * tot[t] / T
    pos[t] = [cur, cur - w, cur]
    cur -= w + gap
cmap = plt.get_cmap("tab20")
col = {t: cmap(i % 20) for i, t in enumerate(order)}
fig, ax = plt.subplots(figsize=(13, 13)); ax.set_aspect("equal"); ax.axis("off")
for t in order:
    s, e, _ = pos[t]
    ax.add_patch(Wedge((0, 0), 1.0, math.degrees(e), math.degrees(s), width=0.05, color=col[t]))
    m = (s + e) / 2; x, y = pol(m, 1.03)
    deg = math.degrees(m) % 360
    rot, ha = (deg, "left") if (deg < 90 or deg > 270) else (deg - 180, "right")
    ax.text(x, y, f"{t} ({tot[t]})", rotation=rot, rotation_mode="anchor", ha=ha, va="center", fontsize=11)
for a, b, n in sorted(pairs, key=lambda z: z[2]):
    pass
# allocate ribbon slots in a consistent order around each TF
idx = {t: i for i, t in enumerate(order)}
for t in order:
    mine = sorted([(b if a == t else a, n) for a, b, n in pairs if t in (a, b)], key=lambda z: (idx[z[0]] - idx[t]) % len(order))
    pos[t].append({})
    s, e, c = pos[t][:3]
    for o, n in mine:
        w = (s - e) * n / tot[t]
        pos[t][3][o] = (c, c - w); c -= w
for a, b, n in sorted(pairs, key=lambda z: z[2]):
    a1, a2 = pos[a][3][b]; b1, b2 = pos[b][3][a]
    big = a if tot[a] >= tot[b] else b
    ribbon(ax, a1, a2, b1, b2, col[big], alpha=0.25 + 0.5 * min(n, 60) / 60)
ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
ax.set_title(f"TF–TF chord: identical clique edges shared by two TFs (pairs with ≥{MINE} shared edges)\n"
             "ribbon width = shared edges · number = total over drawn pairs · all cliques, blacklist removed",
             loc="left", fontsize=13)
fig.savefig(f"{OUT}/fig_chord_tf_tf_shared_edges.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/fig_chord_tf_tf_shared_edges.svg", bbox_inches="tight", facecolor="white")
print("A:", len(tfs), "TFs", len(pairs), "pairs")

# ---- B: TFs <-> top shared edges (one per locus) ----
d = pd.read_csv("edges/top_shared_clique_edges.tsv", sep="\t")
d["tfl"] = d.tfs.str.split(",").apply(lambda l: list(l))
d["n"] = d.tfl.str.len()
d = d[d.distance_kb > 25].sort_values(["n", "distance_kb"], ascending=[False, False])
pick, seen = [], []
for r in d.itertuples():
    c = r.region1.split(":")[0]; p = int(r.region1.split(":")[1].split("-")[0])
    if any(c == c2 and abs(p - p2) < 200_000 for c2, p2 in seen):
        continue
    seen.append((c, p)); pick.append(r)
    if len(pick) == 10:
        break
def lab(reg, g):
    if isinstance(g, str) and g:
        return "/".join(sym.get(x, x) for x in g.split(",")) + " prom."
    c, se = reg.split(":"); return f"distal {c}:{int(se.split('-')[0]) / 1e6:.2f} Mb"
HUBCOL = ["#4C8BC9", "#F2B705", "#7B3F9E", "#E0475A", "#59C1C4", "#E8743B", "#5E7183", "#2E9E5B", "#B5838D", "#8C6D31"]
links = [(t, i) for i, r in enumerate(pick) for t in r.tfl]
L = pd.DataFrame(links, columns=["tf", "e"])
first = L.groupby("tf").e.min(); cnt = L.groupby("tf").e.count()
tfb = sorted(first.index, key=lambda t: (first[t], -cnt[t]))
gapB = math.radians(0.6)
R0, R1 = math.radians(80), math.radians(-80)
spanR = (R0 - R1) - gapB * (len(pick) - 1)
nE = L.groupby("e").tf.count()
ep, cur = {}, R0
for i in range(len(pick)):
    w = spanR * nE[i] / nE.sum(); ep[i] = [cur, cur - w, cur]; cur -= w + gapB
T0, T1 = math.radians(100), math.radians(260)
spanL = (T1 - T0) - gapB * 0.5 * (len(tfb) - 1)
tp, cur = {}, T0
for t in tfb:
    w = spanL / len(tfb); tp[t] = [cur, cur + w, cur]; cur += w + gapB * 0.5
fig, ax = plt.subplots(figsize=(16, 13)); ax.set_aspect("equal"); ax.axis("off")
ylast = 9
for i, r in enumerate(pick):
    s, e, _ = ep[i]
    ax.add_patch(Wedge((0, 0), 1.0, math.degrees(e), math.degrees(s), width=0.07, color=HUBCOL[i]))
    x, y = pol((s + e) / 2, 1.06); y = min(y, ylast - 0.15); ylast = y
    ax.text(x, y, f"{lab(r.region1, r.region1_promoter_gene_ids)} ↔ {lab(r.region2, r.region2_promoter_gene_ids)}\n"
                  f"{r.distance_kb:g} kb · {r.n} TFs" + (" · on a 0.7 loop" if r._6 else ""), ha="left", va="center", fontsize=11)
for t in tfb:
    s, e, _ = tp[t]
    ax.add_patch(Wedge((0, 0), 1.0, math.degrees(s), math.degrees(e), width=0.05,
                       color=plt.get_cmap("Greys")(0.25 + 0.75 * cnt[t] / len(pick))))
    m = (s + e) / 2; x, y = pol(m, 1.03)
    ax.text(x, y, t, ha="right", va="center", rotation=math.degrees(m) - 180, rotation_mode="anchor", fontsize=10.5)
for t, i in sorted(links, key=lambda z: z[1], reverse=True):
    ts, te, tc = tp[t]; wt = (te - ts) / cnt[t]; a1, a2 = tc, tc + wt; tp[t][2] = a2
    hs, he, hc = ep[i]; wh = (hs - he) / nE[i]; b1, b2 = hc, hc - wh; ep[i][2] = b2
    ribbon(ax, a1, a2, b1, b2, HUBCOL[i], alpha=0.6)
ax.set_xlim(-1.35, 2.15); ax.set_ylim(-1.2, 1.2)
ax.set_title("TFs (left) and the 10 most widely shared clique edges >25 kb (right; one per locus)\n"
             "a ribbon = that TF has this exact edge in one of its cliques · TF shade = how many of the 10 edges it has · "
             "gene names only at TSS ±1 kb", loc="left", fontsize=13)
fig.savefig(f"{OUT}/fig_chord_tf_top_shared_edges.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/fig_chord_tf_top_shared_edges.svg", bbox_inches="tight", facecolor="white")
print("B:", len(tfb), "TFs"); [print(r.region1, r.region2, r.n, r.distance_kb) for r in pick]
