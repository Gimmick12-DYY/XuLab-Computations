"""Example TF-TF hub: BAG3-TIAL1 locus (chr10). Top: Peakachu loops >=0.7; middle: hub elements and
promoter TSSs; rows: each TF's clique edges in the window (coloured = edge shared with >=2 other... TFs,
grey = only this TF); right: TF x shared-edge membership."""
import sys
from itertools import combinations
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np, pandas as pd
plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})
CH, LO, HI = "chr10", 119_585_000, 119_688_000
X = Path("x"); OUT = Path(sys.argv[1])
parse = lambda r: (r.split(":")[0], *map(int, r.split(":")[1].split("-"))) if ":" in r else (r.split("_")[0], int(r.split("_")[1]), int(r.split("_")[2]))
mid = lambda r: (r[1] + r[2]) // 2
EL = {"TIAL1 prom.": (119596258, 119597524), "BAG3 prom.": (119650855, 119652024), "E1": (119669788, 119670253),
      "E2": (119672181, 119672732), "E3": (119676415, 119677440), "E4": (119677457, 119677796)}
def el_of(s, e):
    for k, (a, b) in EL.items():
        if s < b and e > a:
            return k
d = pd.read_csv("edges/top_shared_clique_edges.tsv", sep="\t")
sh = []
for r in d.itertuples():
    a, b = parse(r.region1), parse(r.region2)
    if a[0] == CH and LO <= a[1] and b[2] <= HI:
        tfl = r.tfs.split(",")
        sh.append({"a": el_of(a[1], a[2]), "b": el_of(b[1], b[2]), "ma": mid(a), "mb": mid(b), "kb": r.distance_kb,
                   "tfs": tfl, "loop": r._6})
S = pd.DataFrame(sh).sort_values(["ma", "mb"]).reset_index(drop=True)
S["name"] = [f"{r.a}–{r.b}\n{r.kb:g} kb" for r in S.itertuples()]
tfs_sh = pd.Series([t for l in S.tfs for t in l]).value_counts()
rows = list(tfs_sh.index)
# every clique edge of these TFs inside the window
inputs = [l.split("\t") for l in Path("../basic/inputs.tsv").read_text().splitlines()]
alle = {}
for tf, cp, _ in inputs:
    if tf not in rows:
        continue
    c = pd.read_csv(X / cp.lstrip("/"), sep="\t", usecols=["regions"])
    es = set()
    for rr in c.regions.astype(str):
        pk = [parse(x) for x in rr.split(";")]
        if pk[0][0] != CH:
            continue
        for p, q in combinations(pk, 2):
            if LO <= min(p[1], q[1]) and max(p[2], q[2]) <= HI:
                es.add(tuple(sorted((mid(p), mid(q)))))
    alle[tf] = es
loops = []
for res, colr in (("5000", "#D81B60"), ("10000", "#0072B2")):
    for ln in open(f"../plots/colour/peakachu_{res}.loops.0.7.bedpe"):
        p = ln.split()
        if p[0] == CH:
            m1, m2 = (int(p[1]) + int(p[2])) // 2, (int(p[4]) + int(p[5])) // 2
            if LO <= m1 and m2 <= HI:
                loops.append((m1, m2, colr))
def arc(x1, x2, h, n=50):
    t = np.linspace(0, np.pi, n)
    return np.column_stack(((x1 + x2) / 2 - (x2 - x1) / 2 * np.cos(t), h * np.sin(t)))
CLS = {"TIAL1": "#D55E00", "BAG3": "#0072B2", "dn": "#009E73"}
kind = lambda r: CLS["TIAL1"] if r.a.startswith("TIAL1") else CLS["BAG3"] if r.a.startswith("BAG3") else CLS["dn"]
ecol = lambda i: kind(S.loc[i])
fig = plt.figure(figsize=(19, 3.2 + 0.36 * len(rows)))
ax = fig.add_axes([0.09, 0.05, 0.56, 0.9]); axm = fig.add_axes([0.7, 0.05, 0.29, 0.9])
span = HI - LO; y = 0
for m1, m2, c in loops:
    a = arc(m1, m2, -1.3); ax.plot(a[:, 0], a[:, 1] + y + 0.1, color=c, lw=1.2, alpha=0.7)
ax.text(LO - 0.01 * span, y - 0.6, "Hi-C loops ≥0.7", ha="right", va="center", fontsize=11)
y -= 2.0
for k, (s, e) in EL.items():
    ax.add_patch(plt.Rectangle((s, y - 0.25), max(e - s, span / 300), 0.5, color="#333"))
    ax.text((s + e) / 2 + (900 if k == "E4" else -900 if k == "E3" else 0), y + 0.45, k, ha="center", va="bottom", fontsize=10.5, fontweight="bold" if "prom" in k else None)
for pos, g, strand in ((119597028, "TIAL1", -1), (119651369, "BAG3", 1)):
    ax.annotate("", xy=(pos + strand * 2500, y - 0.6), xytext=(pos, y - 0.6), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(LO - 0.01 * span, y, "hub elements\n(▶ TSS)", ha="right", va="center", fontsize=11)
y -= 1.4
row_y = {}
for tf in rows:
    grey = [arc(a, b, 0.32) + [0, y - 0.15] for a, b in alle.get(tf, set())]
    ax.add_collection(LineCollection(grey, colors="#BDBDBD", linewidths=0.8))
    for i, r in S.iterrows():
        if tf in r.tfs:
            a = arc(r.ma, r.mb, 0.32) + [0, y - 0.15]; ax.plot(a[:, 0], a[:, 1], color=ecol(i), lw=2.2)
    ax.text(LO - 0.01 * span, y, tf, ha="right", va="center", fontsize=10.5)
    row_y[tf] = y; y -= 0.36
ax.set_xlim(LO, HI); ax.set_ylim(y, 1.0); ax.set_yticks([])
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:.2f}")); ax.set_xlabel(f"{CH} position (Mb)")
import os
if not os.environ.get("NOTITLE"): fig.suptitle("Example TF–TF hub: BAG3 promoter, TIAL1 promoter and four elements downstream of BAG3\n"
             "rows = each TF's clique edges in the window · orange = TIAL1 promoter links (on a 0.7 loop, ±30 kb) · blue = BAG3 promoter to E1–E4\n"
             "green = among E1–E4 · grey = edge only this TF has · right: which TF has which shared edge",
             x=0.02, y=1.17, ha="left", fontsize=14)
# membership matrix
M = np.array([[tf in r.tfs for _, r in S.iterrows()] for tf in rows])
for j in range(len(S)):
    ii = np.where(M[:, j])[0]
    axm.scatter([j] * len(ii), [row_y[rows[i]] for i in ii], s=70, color=ecol(j), zorder=3)
axm.set_ylim(y, 1.0 - 3.4 + 0.0 if False else row_y[rows[0]] + 0.6)
axm.set_ylim(ax.get_ylim()[0], row_y[rows[0]] + 0.5)
axm.set_yticks([row_y[t] for t in rows]); axm.set_yticklabels(rows, fontsize=10)
axm.set_xticks(range(len(S))); axm.set_xticklabels([f"{n}{' · loop' if l else ''}" for n, l in zip(S.name, S.loop)], rotation=90, fontsize=9.5)
axm.xaxis.tick_top(); axm.set_xlim(-0.6, len(S) - 0.4)
axm.grid(color="#eee"); axm.set_axisbelow(True)
axm.set_title("TF × shared edge  (number of TFs below)", fontsize=12, pad=8)
for j, r in S.iterrows():
    axm.text(j, ax.get_ylim()[0] + 0.1, str(len(r.tfs)), ha="center", va="bottom", fontsize=10)
suf = "_notitle" if os.environ.get("NOTITLE") else ""
fig.savefig(OUT / f"fig_example_hub_BAG3_TIAL1{suf}.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig_example_hub_BAG3_TIAL1.svg", bbox_inches="tight", facecolor="white")
S.assign(tfs=S.tfs.str.join(","), name=S.name.str.replace("\n", " ")).to_csv(OUT / "example_hub_BAG3_TIAL1_shared_edges.tsv", sep="\t", index=False)
print(len(rows), "TFs;", len(S), "shared edges;", len(loops), "loops in window")
print(S[["a", "b", "kb", "loop"]].assign(n=S.tfs.str.len()).to_string())
