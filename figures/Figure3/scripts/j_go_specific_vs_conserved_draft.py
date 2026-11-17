"""Draft Fig 3 panel j (user 2026-11-17, GO steps 2-3): TF-specific (module genes of exactly 1 TF) and conserved (modules of >=3 TFs) gene
sets, each with its OWN top-10 GO BP terms (no shared term list). Values = pooled lists from scripts/go_per_tf.py (checks/go_per_tf):
x = pooled fold enrichment, colour = pooled BH q (grey = q >= 0.05), size = genes in the term; the number of TFs whose own list has the term
at q < 0.05 is printed next to each dot. Terms ranked by pooled p; near-duplicates dropped (a term whose gene count and size are within
10% / 25% of a better-ranked term, i.e. the same genes under a parent/child name).
  python Figure3/scripts/j_go_specific_vs_conserved_draft.py   (Mac)"""
import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
H = Path(__file__).resolve().parents[1]; G = H / "checks/go_per_tf"; MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "axes.labelsize": 6.5, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False})
cm = LinearSegmentedColormap.from_list("t", ["#C9E3DC", "#2C7A6B", "#0E3B33"])
def core(n):
    n = n.lower().replace("torc1", "tor")
    for w in ("positive regulation of ", "negative regulation of ", "regulation of "): n = n.replace(w, "")
    return n.replace(" process", "").strip()
def top(grp, k=10):
    pq = pd.read_csv(G / f"pooled_{grp}.tsv", sep="\t", index_col=0).sort_values("p"); sm = pd.read_csv(G / f"summary_{grp}.tsv", sep="\t", index_col=0)
    keep = []
    for t, r in pq.iterrows():
        if r.k == 0: continue
        if any(abs(r.k - pq.k[u]) <= 0.1 * pq.k[u] and abs(r.K - pq.K[u]) <= 0.25 * pq.K[u] for u in keep): continue
        c = core(r["name"])
        if any(c in core(pq.name[u]) or core(pq.name[u]) in c for u in keep): continue      # same process under (positive) regulation / TOR-TORC1 names
        keep.append(t)
        if len(keep) == k: break
    T = pq.loc[keep].copy(); T["n_tfs_sig"] = sm.n_tfs_sig.reindex(keep).fillna(0).astype(int).values; T["n_tfs_tested"] = int(sm.n_tfs_tested.iloc[0]); return T
SHORT = {"dolichol-linked oligosaccharide biosynthetic process": "dolichol oligosaccharide biosynthesis", "regulation of monoatomic ion transmembrane transport": "reg. of ion transmembrane transport",
         "monoatomic ion transmembrane transport": "ion transmembrane transport", "monoatomic cation transmembrane transport": "cation transmembrane transport",
         "regulation of mononuclear cell migration": "reg. of mononuclear cell migration", "positive regulation of cell cycle phase transition": "pos. reg. of cell cycle phase transition",
         "positive regulation of triglyceride biosynthetic process": "pos. reg. of triglyceride biosynthesis", "positive regulation of TORC1 signaling": "pos. reg. of TORC1 signaling",
         "positive regulation of catabolic process": "pos. reg. of catabolic process", "regulation of cellular response to stress": "reg. of response to stress", "endoplasmic reticulum to Golgi vesicle-mediated transport": "ER to Golgi vesicle transport"}
cap = lambda s: (lambda t: t[0].upper() + t[1:])(SHORT.get(s, s))
SP, CO = top("specific"), top("conserved")
fig = plt.figure(figsize=(183 * MM, 66 * MM)); axs = [fig.add_axes([0.205, 0.2, 0.16, 0.64]), fig.add_axes([0.655, 0.2, 0.16, 0.64])]
vmax = max(2.5, -np.log10(min(SP.q.min(), CO.q.min())))
for ax, T, ttl in ((axs[0], SP, "TF-specific module genes (1 TF)"), (axs[1], CO, "Conserved module genes (≥3 TFs)")):
    y = np.arange(len(T)); sig = T.q < 0.05
    ax.scatter(T.fold[~sig], y[~sig], s=4 + 0.35 * T.k[~sig], color="#D9D9D9", edgecolors="#9E9E9E", linewidths=0.4, zorder=3)
    ax.scatter(T.fold[sig], y[sig], s=4 + 0.35 * T.k[sig], c=-np.log10(T.q[sig]), cmap=cm, vmin=1.3, vmax=vmax, lw=0, zorder=3)
    ax.axvline(1, color="#9E9E9E", lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.set_yticks(y); ax.set_yticklabels([cap(s) for s in T.name], fontsize=5.5); ax.set_ylim(len(T) - 0.4, -0.6); ax.tick_params(axis="y", length=0)
    xm = max(2.0, T.fold.max() * 1.15); ax.set_xlim(0.5, xm); ax.set_xlabel("Fold enrichment")
    for i, (n, q) in enumerate(zip(T.n_tfs_sig, T.q)):
        ax.text(xm * 1.02, i, f"q = {q:.2g}; {n}/{T.n_tfs_tested.iloc[0]} TFs" if q >= 0.001 else f"q < 0.001; {n}/{T.n_tfs_tested.iloc[0]} TFs", va="center", fontsize=5, clip_on=False)
    ax.set_title(f"{ttl}\nn = {int(T.n_list.iloc[0]):,} annotated genes", fontsize=6.5, pad=4)
cax = fig.add_axes([0.925, 0.5, 0.01, 0.3]); cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(1.3, vmax)), cax=cax); cb.set_label("−log10 q", fontsize=5.5); cb.ax.tick_params(labelsize=5)
hs = [plt.scatter([], [], s=4 + 0.35 * n, color="#4D4D4D", lw=0) for n in (20, 50, 100)] + [plt.scatter([], [], s=20, color="#D9D9D9", edgecolors="#9E9E9E", linewidths=0.4)]
fig.legend(hs, ["20", "50", "100 genes", "q ≥ 0.05"], loc="lower center", ncol=4, fontsize=5.5, bbox_to_anchor=(0.5, 0.05), handletextpad=0.2, columnspacing=1.0)
fig.text(0.5, 0.02, "q = pooled BH q;  n/N TFs = TFs (of those with >= 20 annotated genes in the group) whose own list has the term at q < 0.05", fontsize=5, ha="center", color="#4D4D4D")
PS = {("normal", "normal"): "ArialMT"}
def illustrator_text(svg):
    def one(m):
        tag = m.group(0); sm = re.search(r'style="([^"]*)"', tag)
        if not sm: return tag
        keep, size = [], None
        for d in [d.strip() for d in sm.group(1).split(";") if d.strip()]:
            k, _, v = d.partition(":"); k, v = k.strip(), v.strip()
            if k == "font":
                mm_ = re.match(r"(?:\S+\s+)?([\d.]+)px", v); size = mm_.group(1) if mm_ else None
            else: keep.append(f"{k}: {v}")
        if size is None: return tag
        tag = tag.replace(sm.group(0), f'style="{"; ".join(keep)}"' if keep else "")
        return tag[:-1].rstrip() + f' font-family="Arial, ArialMT, Helvetica, sans-serif" font-size="{float(size):.3f}">'
    return re.sub(r"<(?:text|tspan)\b[^>]*>", one, svg)
stem = H / "j_go_specific_vs_conserved_draft"; fig.savefig(f"{stem}.svg"); fig.savefig(f"{stem}.png", dpi=300)
p = Path(f"{stem}.svg"); p.write_text(illustrator_text(p.read_text())); print("font shorthand left:", p.read_text().count("font: "))
print(SP[["name", "k", "fold", "q", "n_tfs_sig"]].round(4).to_string()); print(CO[["name", "k", "fold", "q", "n_tfs_sig"]].round(4).to_string())
