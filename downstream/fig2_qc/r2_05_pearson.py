#!/usr/bin/env python
"""Round 2, Pearson version (panel-b style) of the summit-centred test, at ZNF777 depth.

Features = summit groups: for each reference j, its summit set (default unique top-5000) is sorted by bulk
signalValue and split into G = 30 equal groups (~50-170 summits each), giving 6 x 30 = 180 groups.
Per group and track:
  bulk_k  = log2(CPM + 1) of ChIP reads in summit +-250 bp (summed over the group's summits; CPM over all groups)
  sc_i    = log2((C_i + .5)/(F_i + .5)) - log2((C_pool-i + .5)/(F_pool-i + .5))
            C = reads within +-100 bp of the summits, F = 1-2 kb flanks, pool-i = 79-TF pool minus TF i.
            sc counts thinned to ZNF777 depth (3 seeds; r averaged) or full depth.
12x12 Pearson (6 bulk + 6 sc), average-linkage clustering on 1 - r; sc-vs-bulk 6x6 block: own-best per sc
track, diag - offdiag, exact p over 720 sc-label shuffles; leave-one-TF-out p (5x5, 120 shuffles).
Outputs OUT/round2/pearson/: r2p_corr_<set>_<depth>.tsv, r2p_link_*.npz, r2p_summary.tsv, figs/r2p_<set>_<depth>.png
"""
import itertools
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from scipy.spatial.distance import squareform

from fig2_common import DM_DEPTH, DM_SEEDS, OUT, ROOT, TFS

sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402

P = rp.use()
INK = P.neutral.ink
R2 = OUT / "round2"
OP = R2 / "pearson"
(OP / "figs").mkdir(parents=True, exist_ok=True)
G = 30
NT = len(TFS)
S = pd.read_csv(R2 / "summits.tsv", sep="\t")
z = np.load(R2 / "summit_counts.npz", allow_pickle=True)
C79, F79, tf79, tot79 = z["C"].astype(np.int64), z["F"].astype(np.int64), list(z["tf79"]), z["tot79"]
pi = [tf79.index(t) for t in TFS]
Cpool, Fpool = C79.sum(0), F79.sum(0)
Bk = S[[f"bulk_{t}" for t in TFS]].values


def groups(typ, n):
    out = []
    for t in TFS:
        m = S.ref.values == t
        if typ != "all":
            m &= S[typ].values.astype(bool)
        ix = np.flatnonzero(m)
        ix = ix[np.argsort(-S.signal.values[ix], kind="stable")][:n]
        out += [g for g in np.array_split(ix, G)]
    return out


def features(gr, Csc, Fsc):
    gB = np.array([Bk[g].sum(0) for g in gr])
    bulk = np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)
    sc = np.zeros((len(gr), NT))
    for i in range(NT):
        k = pi[i]
        for a, g in enumerate(gr):
            c, f = Csc[i, g].sum(), Fsc[i, g].sum()
            cp, fp = Cpool[g].sum() - C79[k, g].sum(), Fpool[g].sum() - F79[k, g].sum()
            sc[a, i] = np.log2((c + .5) / (f + .5)) - np.log2((cp + .5) / (fp + .5))
    return bulk, sc


def block_stats(B):
    k = len(B)
    off = ~np.eye(k, dtype=bool)
    st = lambda M: M.diagonal().mean() - M[off].mean()
    obs = st(B)
    null = np.array([st(B[list(p)]) for p in itertools.permutations(range(k))])
    return obs, float(np.mean(null >= obs - 1e-12))


def cluster_next(R, k):
    Z = linkage(squareform(np.clip(1 - R, 0, None), checks=False), "average")
    root = to_tree(Z)
    nxt = []
    for i in range(k):
        want, node = {i, k + i}, root
        while True:
            for ch in (node.left, node.right):
                if ch is not None and want <= set(ch.pre_order()):
                    node = ch
                    break
            else:
                break
        nxt.append(int(sum(x < k for x in node.pre_order()) == 1))
    return Z, nxt


labs = [f"Bulk {t}" for t in TFS] + [f"sc {t}" for t in TFS]
rows, loo_rows, seed_rows = [], [], []
pdm = np.minimum(1, DM_DEPTH / tot79[pi])
for typ, n in (("unique", 5000), ("unique", 2000), ("preferred", 5000), ("all", 5000)):
    gr = groups(typ, n)
    name = f"{typ}_top{n}"
    for depth, frac in (("znf777_25pct", .25), ("znf777_50pct", .5), ("znf777", 1.0), ("full", None)):
        if frac is None:
            reps = [(C79[pi], F79[pi])]
        else:
            pf = np.minimum(1, frac * pdm)[:, None]
            reps = [(np.random.default_rng(s).binomial(C79[pi], pf),
                     np.random.default_rng(s + 1).binomial(F79[pi], pf)) for s in DM_SEEDS]
        Rs = []
        for si, (Cs, Fs) in enumerate(reps):
            bulk, sc = features(gr, Cs, Fs)
            R1 = np.corrcoef(np.column_stack([bulk, sc]).T)
            Rs.append(R1)
            o1, p1 = block_stats(R1[NT:, :NT])
            seed_rows.append(dict(set=name, depth=depth, seed=si, diag_minus_off=o1, perm_p=p1,
                                  own_best_row=int(sum(np.argmax(R1[NT + i, :NT]) == i for i in range(NT)))))
        R = np.mean(Rs, 0)
        blk = R[NT:, :NT]
        obs, p = block_stats(blk)
        Z, nxt = cluster_next(R, NT)
        own = [int(np.argmax(blk[i]) == i) for i in range(NT)]
        own_c = [int(np.argmax(blk[:, j]) == j) for j in range(NT)]
        tag = f"{name}_{depth}"
        pd.DataFrame(R, labs, labs).to_csv(OP / f"r2p_corr_{tag}.tsv", sep="\t", float_format="%.4f")
        np.savez(OP / f"r2p_link_{tag}.npz", Z=Z, labs=np.array(labs), R=R)
        rows.append(dict(set=name, depth=depth, n_groups=len(gr), diag_minus_off=obs, perm_p=p,
                         own_best_row=sum(own), own_best_col=sum(own_c), clusters_next_to_own=sum(nxt),
                         **{f"r_own_{t}": blk[i, i] for i, t in enumerate(TFS)},
                         bulk_bulk_mean_off=R[:NT, :NT][~np.eye(NT, dtype=bool)].mean()))
        for d in range(NT):
            keep = [k for k in range(NT) if k != d]
            o2, p2 = block_stats(blk[np.ix_(keep, keep)])
            loo_rows.append(dict(set=name, depth=depth, dropped=TFS[d], diag_minus_off=o2, perm_p_5x5=p2))
        print(f"{tag:24s} d-o={obs:+.3f} p={p:.4f} rows={sum(own)} cols={sum(own_c)} next={sum(nxt)}", flush=True)
        if frac not in (1.0, None):
            continue

        fig = plt.figure(figsize=(3.4, 3.5))
        axd = fig.add_axes([0.27, 0.78, 0.53, 0.11])
        d = dendrogram(Z, ax=axd, no_labels=True, color_threshold=0, above_threshold_color=INK)
        axd.axis("off")
        o = d["leaves"]
        axh = fig.add_axes([0.27, 0.22, 0.53, 0.56])
        im = axh.imshow(R[np.ix_(o, o)], cmap=P.div_cmap(), vmin=-1, vmax=1, aspect="auto")
        ol = [labs[i] for i in o]
        axh.set_xticks(range(len(o)), ol, rotation=90)
        axh.set_yticks(range(len(o)), ol)
        for tl in axh.get_xticklabels() + axh.get_yticklabels():
            tl.set_color(INK if tl.get_text().startswith("Bulk") else P.arms.raw)
        axh.tick_params(length=0)
        cax = fig.add_axes([0.83, 0.22, 0.025, 0.25])
        fig.colorbar(im, cax=cax, label="Pearson r")
        dep = "ZNF777 depth" if depth == "znf777" else "full depth"
        fig.suptitle(f"{typ} top-{n} summits, {G} groups/reference, {dep}\n"
                     f"own reference best for {sum(own)}/6 sc tracks; diag−off {obs:+.2f}, p = {p:.3f}",
                     fontsize=6, y=0.995)
        for ext in ("png", "pdf"):
            fig.savefig(OP / "figs" / f"r2p_{tag}.{ext}", dpi=600 if ext == "png" else None)
        plt.close(fig)
pd.DataFrame(rows).to_csv(OP / "r2p_summary.tsv", sep="\t", index=False, float_format="%.4f")
pd.DataFrame(loo_rows).to_csv(OP / "r2p_loo.tsv", sep="\t", index=False, float_format="%.4f")
pd.DataFrame(seed_rows).to_csv(OP / "r2p_per_seed.tsv", sep="\t", index=False, float_format="%.4f")
sm = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(3.2, 2.2))
for nm in sm.set.unique():
    v = sm[sm.set == nm].set_index("depth")
    ax.plot([.25, .5, 1.0], [v.loc[d, "diag_minus_off"] for d in ("znf777_25pct", "znf777_50pct", "znf777")],
            marker="o", ms=3, lw=0.8, label=nm.replace("_", " "))
ax.axhline(0, color=P.neutral.light, lw=0.5)
ax.set_xticks([.25, .5, 1.0], ["25%", "50%", "100%"])
ax.set_xlabel("Fraction of ZNF777 depth (420,339 reads)")
ax.set_ylabel("Pearson diag − off-diag")
ax.set_title("Summit-group Pearson vs depth")
ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OP / "figs" / f"r2p_depth_curve.{ext}", dpi=600 if ext == "png" else None)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(3).to_string())
print(pd.DataFrame(loo_rows).pivot_table(index=["set", "depth"], columns="dropped", values="perm_p_5x5").round(3))
