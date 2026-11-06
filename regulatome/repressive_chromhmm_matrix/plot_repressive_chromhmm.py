# Figures + summary for the repressive ChromHMM TF x TF matrices (repressive_chromhmm.py output), in the two deck styles.
#   (a) "Complex slides by matrix" style: SpQN percentile, RdYlBu_r 0-1, deck enhancer dendrogram order (mats/pct_enhancer.tsv) and
#       each set's own dendrogram order; known complexes outlined when recovered (mean pairwise percentile >= 0.90).
#   (b) Dr. Wang style: raw Pearson r, RdYlBu_r 0.65-1, his Pearson TF order (wang_his_order.py).
#   Side-by-side enhancer vs repressive panels in each style. "enhancer" = EnhA1/A2/G1/G2/Wk + EnhBiv, the deck's stored matrix.
#   run from scripts/complex_catalog (uses matfig.py, known_def.py, wang_his_order.py there); usage: python3 <this> <data dir> <out dir>
import sys, json, numpy as np, pandas as pd
from matplotlib.lines import Line2D
exec(open("matfig.py").read().split('if __name__=="__main__":')[0])
exec(open("known_def.py").read())
D, OUT = sys.argv[1], sys.argv[2]
HISP = open("wang_his_order.py").read().split('"pearson": ("')[1].split('"')[0].lower().split()
SETS = [("enhancer_biv", "Enhancer (deck)"), ("repressive", "Repressive: ReprPC+ReprPCWk+Het+ZNF/Rpts"), ("polycomb", "Polycomb: ReprPC+ReprPCWk"),
        ("het", "Het"), ("znf_rpts", "ZNF/Rpts")]
DECK, _ = load("mats/pct_enhancer.tsv")
COLS = ["#000000", "#6A1B9A", "#00695C", "#C2185B", "#1565C0", "#E65100", "#2E7D32", "#795548", "#00838F", "#AD1457"]
SHORT = {"prc2": "PRC2", "nurd": "NuRD", "nurd_z": "NuRD+Z512B", "cohesin": "Cohesin", "maz": "MAZ–ZNF143", "g9a": "G9a/GLP", "nfy": "NF-Y/MLX",
         "mef2": "MEF2", "znf512": "ZNF512", "pcgf6": "PRC1.6", "chahp": "ChAHP", "hp1setdb1": "HP1/SETDB1", "e2f3rbbp4": "E2F3–RBBP4", "hox": "HOX"}
def get(kind, s):
    P = pd.read_csv(f"{D}/{kind}_{s}.tsv", sep="\t", index_col=0); P.index = P.index.str.lower(); P.columns = P.columns.str.lower(); return P
cells = pd.read_csv(f"{D}/tf_cells.tsv", sep="\t"); cells["tf"] = cells.tf.str.lower(); cells = cells.set_index("tf").cells
# depth-matched null for a group: each member swapped for a random TF within +/-4 cell-count ranks (distinct), 5000 draws
rk = cells.rank(method="first").astype(int); byrank = {v: k for k, v in rk.items()}; N = len(rk); rng = np.random.default_rng(0)
def mean_pct(P, mem): return np.mean([P.loc[a, b] for i, a in enumerate(mem) for b in mem[i + 1:]])
def null_p(P, mem, obs, n=5000):
    c = 0
    for _ in range(n):
        while True:
            g = [byrank[int(np.clip(rk[m] + rng.integers(-4, 5), 1, N))] for m in mem]
            if len(set(g)) == len(g): break
        c += mean_pct(P, g) >= obs
    return (c + 1) / (n + 1)
rec = {}; rows = []
for s, lab in SETS:
    P = get("pct", s)
    for k, name, mem in KNOWN:
        v = mean_pct(P, mem); rows.append(dict(set=s, complex=name, members=",".join(mem), mean_pair_pct=round(v, 3), recovered=v >= 0.90,
                                                depth_matched_p=round(null_p(P, mem, v), 4)))
        rec[(s, k)] = v >= 0.90
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/known_complex_recovery.tsv", sep="\t", index=False)
def ov(ax, s, L, fs, title, legend_ax=None):
    P = get("pct", s).loc[L, L].values; n = len(L)
    im = ax.imshow(P, cmap="RdYlBu_r", vmin=0, vmax=1, interpolation="nearest"); ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([t.upper() for t in L], rotation=90, fontsize=fs); ax.set_yticklabels([t.upper() for t in L], fontsize=fs)
    allm = set(); handles = []
    for c, (k, name, mem) in enumerate([x for x in KNOWN if rec[(s, x[0])]]):
        col = COLS[c % len(COLS)]; pos = sorted(L.index(t) for t in mem); allm |= set(pos); rr = runs(pos)
        for a, b in rr:
            if b > a: ax.add_patch(Polygon([(a - 0.5, a - 0.5), (b + 0.5, a - 0.5), (b + 0.5, b + 0.5)], closed=True, fill=False, edgecolor=col, lw=1.5, zorder=5))
        d = 0.22 if k == "nurd_z" else 0.0
        for i in pos:
            for j in pos:
                if i < j and not any(a <= i and j <= b for a, b in rr):
                    ax.add_patch(Rectangle((j - 0.5 + d, i - 0.5 + d), 1 - 2 * d, 1 - 2 * d, fill=False, edgecolor=col, lw=1.1, zorder=6))
        handles.append(Line2D([0], [0], color=col, lw=2.0, label=SHORT[k]))
    for ll in (ax.get_xticklabels(), ax.get_yticklabels()):
        for k_, lb in enumerate(ll):
            if k_ in allm: lb.set_fontweight("bold"); lb.set_color(RED)
    ax.tick_params(length=1.2, width=0.3, pad=1); [s_.set_linewidth(0.4) for s_ in ax.spines.values()]
    ax.set_title(title, fontsize=9); return im, handles
def save(fig, stem): fig.savefig(f"{OUT}/{stem}.png", facecolor="white", bbox_inches="tight", pad_inches=0.04, dpi=300); fig.savefig(f"{OUT}/{stem}.svg", facecolor="white", bbox_inches="tight", pad_inches=0.04); plt.close(fig)
def cbar(fig, im, rect, label):
    cax = fig.add_axes(rect); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6, width=0.3, length=2); cb.outline.set_linewidth(0.3); cb.set_label(label, fontsize=7)
orders = {}
for s, lab in SETS:
    P = get("pct", s); Lall = list(P.index); orders[s] = [Lall[i] for i in order_of(P.values)]
    for oname, L in (("deckorder", DECK), ("clustered", orders[s])):
        fig = plt.figure(figsize=(6.4, 5.95)); ax = fig.add_axes([0.13, 0.06, 0.74, 0.80])
        im, h = ov(ax, s, L, 5.0, f"{lab}" + (" — deck enhancer order" if oname == "deckorder" else " — own dendrogram order"))
        cbar(fig, im, [0.905, 0.30, 0.018, 0.36], "Percentile of TF–TF Pearson r (SpQN)")
        fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.885, 0.05), fontsize=7, frameon=False, handlelength=1.5, title="Recovered", title_fontsize=7.5)
        save(fig, f"a_pct_{s}_{oname}")
    # style (b)
    R = get("pearson", s).loc[HISP, HISP].values; n = len(HISP)
    fig = plt.figure(figsize=(6.4, 6.0)); ax = fig.add_axes([0.13, 0.06, 0.74, 0.80])
    im = ax.imshow(R, cmap="RdYlBu_r", vmin=0.65, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels([t.upper() for t in HISP], rotation=90, fontsize=4.6); ax.set_yticklabels([t.upper() for t in HISP], fontsize=4.6)
    ax.tick_params(length=1.2, width=0.3, pad=1); ax.set_title(f"{lab} — Dr. Wang's Pearson order", fontsize=9)
    cbar(fig, im, [0.905, 0.30, 0.018, 0.36], "Pearson r"); save(fig, f"b_wang_pearson_{s}")
# side by side: (a) enhancer vs repressive, both in deck order, plus repressive in its own order; (b) enhancer vs repressive in his order
for oname, LL in (("deckorder", (DECK, DECK)), ("ownorder", (orders["enhancer_biv"], orders["repressive"]))):
    fig = plt.figure(figsize=(12.6, 6.3)); hs = []
    for i, (s, L) in enumerate(zip(("enhancer_biv", "repressive"), LL)):
        ax = fig.add_axes([0.05 + i * 0.47, 0.08, 0.40, 0.80]); im, h = ov(ax, s, L, 3.9, dict(SETS)[s] + (" (deck order)" if oname == "deckorder" else " (own order)"))
        fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.452 + i * 0.47, 0.08), fontsize=6, frameon=False, handlelength=1.3, title="Recovered", title_fontsize=6.5)
    cbar(fig, im, [0.925, 0.55, 0.009, 0.25], "Percentile (SpQN r)"); save(fig, f"a_side_by_side_enhancer_vs_repressive_{oname}")
fig = plt.figure(figsize=(12.6, 6.3))
for i, s in enumerate(("enhancer_biv", "repressive")):
    ax = fig.add_axes([0.05 + i * 0.47, 0.08, 0.40, 0.80]); R = get("pearson", s).loc[HISP, HISP].values; n = len(HISP)
    im = ax.imshow(R, cmap="RdYlBu_r", vmin=0.65, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels([t.upper() for t in HISP], rotation=90, fontsize=3.7); ax.set_yticklabels([t.upper() for t in HISP], fontsize=3.7)
    ax.tick_params(length=1.0, width=0.3, pad=1); ax.set_title(dict(SETS)[s] + " — Pearson r, Dr. Wang's order", fontsize=9)
cbar(fig, im, [0.925, 0.55, 0.009, 0.25], "Pearson r"); save(fig, "b_side_by_side_enhancer_vs_repressive")
# summary prints
E = pd.read_csv(f"{D}/element_sets.tsv", sep="\t"); pd.set_option("display.width", 250)
print(E[["set", "elements", "elements_with_reads", "bp", "median_offdiag_pearson", "median_offdiag_spqn"]].to_string(index=False))
off = ~np.eye(79, dtype=bool)
for s, _ in SETS: print(s, "fraction of off-diag raw Pearson >= 0.65:", round(float((get("pearson", s).values[off] >= 0.65).mean()), 3))
print(T.pivot(index="complex", columns="set", values="mean_pair_pct")[[s for s, _ in SETS]].reindex([n for _, n, _ in KNOWN]).to_string())
print(T.pivot(index="complex", columns="set", values="depth_matched_p")[[s for s, _ in SETS]].reindex([n for _, n, _ in KNOWN]).to_string())
# repressive-only strong pairs: pct >= 0.95 on repressive, < 0.70 on enhancer; then groups (connected components)
Pe, Pr = get("pct", "enhancer_biv"), get("pct", "repressive"); L = list(Pr.index)
pairs = [(a, b, Pr.loc[a, b], Pe.loc[a, b]) for i, a in enumerate(L) for b in L[i + 1:] if Pr.loc[a, b] >= 0.95 and Pe.loc[a, b] < 0.70]
pd.DataFrame(pairs, columns=["tf1", "tf2", "pct_repressive", "pct_enhancer"]).sort_values("pct_repressive", ascending=False).to_csv(f"{OUT}/repressive_only_pairs.tsv", sep="\t", index=False, float_format="%.3f")
print(len(pairs), "repressive-only pairs (pct >= 0.95 repressive, < 0.70 enhancer)")
import networkx as nx
G = nx.Graph(); G.add_edges_from([(a, b) for a, b, _, _ in pairs])
for cc in sorted(nx.connected_components(G), key=len, reverse=True):
    cc = sorted(cc); print(len(cc), " ".join(t.upper() for t in cc), "| mean pct rep %.2f enh %.2f" % (mean_pct(Pr, cc), mean_pct(Pe, cc)) if len(cc) > 1 else "")
json.dump(orders, open(f"{OUT}/dendrogram_orders.json", "w"))
# (b') same Wang-order side by side on a shared data-fitted scale (2nd-99.5th percentile of off-diagonal r across both sets),
# because only 1-5% of element-level raw r reach his 0.65 floor
both = np.concatenate([get("pearson", s).values[off] for s in ("enhancer_biv", "repressive")]); lo, hi = np.quantile(both, [0.02, 0.995])
fig = plt.figure(figsize=(12.6, 6.3))
for i, s in enumerate(("enhancer_biv", "repressive")):
    ax = fig.add_axes([0.05 + i * 0.47, 0.08, 0.40, 0.80]); R = get("pearson", s).loc[HISP, HISP].values; n = len(HISP)
    im = ax.imshow(R, cmap="RdYlBu_r", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels([t.upper() for t in HISP], rotation=90, fontsize=3.7); ax.set_yticklabels([t.upper() for t in HISP], fontsize=3.7)
    ax.tick_params(length=1.0, width=0.3, pad=1); ax.set_title(dict(SETS)[s] + " — Pearson r, Dr. Wang's order", fontsize=9)
cbar(fig, im, [0.925, 0.55, 0.009, 0.25], f"Pearson r (scale {lo:.2f}–{hi:.2f})"); save(fig, "b_side_by_side_enhancer_vs_repressive_fittedscale")
print("fitted scale", round(lo, 3), round(hi, 3))
