#!/usr/bin/env python
"""Round 3: Pearson-correlation versions of the round-2 summit-centred result at ZNF777 depth.

Regions: union of each reference's UNIQUE top-5000 summits (round-2 summits.tsv; ranked by bulk signalValue).
Windows per summit: centre +-100 bp; flanks 1-2 kb on both sides (summed). Bulk = ChIP BAM reads
(bedtools multicov, cached in bulk_counts.npz); sc = read 5' ends (round-2 summit_counts.npz) thinned to a
fraction of ZNF777 depth (3 seeds; r averaged, per-seed statistics also reported).
  P1  log2(CPM + 1) of centre counts (CPM over the region set)
  P2  log2((centre + .5) / (flank + .5)) per window, bulk and sc alike
  P3  P2 centred within assay: sc minus the leave-one-out 79-TF pool's window ratio; bulk minus the mean of the 6 bulk P2
  P4  P2 and P3 on distal summits only (outside TSS +-1 kb)
Per version: 12x12 Pearson (6 bulk + 6 sc), average-linkage clustering (1 - r); 6x6 sc-vs-bulk block:
own-best rows/cols, diag - offdiag, exact p over 720 sc-label shuffles, leave-CTCF-out 5x5 p (120 shuffles),
per-TF own r vs best other r, clusters-next-to-own. Depth curve (25/50/100% of ZNF777 depth, plus full depth)
for every version. Outputs OUT/round3_pearson/.
"""
import itertools
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from scipy.spatial.distance import squareform

from fig2_common import BAM, DM_DEPTH, DM_SEEDS, OUT, ROOT, TFS

sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402

P = rp.use()
INK = P.neutral.ink
R2 = OUT / "round2"
O3 = OUT / "round3_pearson"
(O3 / "figs").mkdir(parents=True, exist_ok=True)
NT = len(TFS)
S = pd.read_csv(R2 / "summits.tsv", sep="\t")
z = np.load(R2 / "summit_counts.npz", allow_pickle=True)
C79, F79, tf79, tot79 = z["C"].astype(np.int64), z["F"].astype(np.int64), list(z["tf79"]), z["tot79"]
pi = [tf79.index(t) for t in TFS]

sel = []
for t in TFS:
    ix = np.flatnonzero((S.ref.values == t) & S["unique"].values.astype(bool))
    sel.append(ix[np.argsort(-S.signal.values[ix], kind="stable")][:5000])
sel = np.concatenate(sel)
reg = S.iloc[sel].reset_index(drop=True)
print("regions", len(reg), reg.ref.value_counts().to_dict(), flush=True)
C, F = C79[:, sel], F79[:, sel]
distal = ~reg.promoter.values.astype(bool)

# ---- bulk centre / flank counts ----
cache = O3 / "bulk_counts.npz"
if cache.exists():
    bz = np.load(cache)
    BC, BF = bz["BC"], bz["BF"]
else:
    bed = O3 / "windows.bed"
    x, c = reg.pos.values, reg.chrom.values
    w = pd.concat([pd.DataFrame({0: c, 1: x - 100, 2: x + 101, 3: [f"{k}_c" for k in range(len(reg))]}),
                   pd.DataFrame({0: c, 1: x - 2000, 2: x - 1000, 3: [f"{k}_l" for k in range(len(reg))]}),
                   pd.DataFrame({0: c, 1: x + 1001, 2: x + 2001, 3: [f"{k}_r" for k in range(len(reg))]})])
    w[1] = w[1].clip(lower=0)
    w.sort_values([0, 1]).to_csv(bed, sep="\t", header=False, index=False)
    out = subprocess.run(["bedtools", "multicov", "-bams", *[str(BAM[t]) for t in TFS], "-bed", str(bed)],
                         check=True, capture_output=True, text=True).stdout
    BC = np.zeros((len(reg), NT)); BF = np.zeros((len(reg), NT))
    for l in out.splitlines():
        p = l.split("\t")
        k, side = p[3].split("_")
        v = np.array(p[4:], float)
        if side == "c":
            BC[int(k)] = v
        else:
            BF[int(k)] += v
    np.savez(cache, BC=BC, BF=BF)
print("bulk counts ready", flush=True)

Cpool, Fpool = C79[:, sel].sum(0), F79[:, sel].sum(0)


def lr(c, f):
    return np.log2((c + .5) / (f + .5))


def lcpm(X):
    return np.log2(X / X.sum(0, keepdims=True) * 1e6 + 1)


def feats(version, Cs, Fs, idx):
    """Cs/Fs: NT x regions sc counts (thinned). Returns bulk (n x 6), sc (n x 6)."""
    if version == "P1":
        return lcpm(BC[idx]), lcpm(Cs[:, idx].T.astype(float))
    b2 = lr(BC[idx], BF[idx])
    s2 = lr(Cs[:, idx].T, Fs[:, idx].T)
    if version == "P2":
        return b2, s2
    # P3
    b3 = b2 - b2.mean(1, keepdims=True)
    pool = np.column_stack([lr(Cpool[idx] - C[pi[i], idx], Fpool[idx] - F[pi[i], idx]) for i in range(NT)])
    return b3, s2 - pool


def perm(B, k):
    off = ~np.eye(k, dtype=bool)
    st = lambda M: M.diagonal().mean() - M[off].mean()
    obs = st(B)
    null = np.array([st(B[list(q)]) for q in itertools.permutations(range(k))])
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
versions = [("P1_plain", "P1", None), ("P2_flanknorm", "P2", None), ("P3_centred", "P3", None),
            ("P4_flanknorm_distal", "P2", distal), ("P4_centred_distal", "P3", distal)]
depths = [("znf777_25pct", 0.25), ("znf777_50pct", 0.5), ("znf777", 1.0), ("full", None)]
rows, pertf, seeds_rows = [], [], []
for vname, ver, mask in versions:
    idx = np.arange(len(reg)) if mask is None else np.flatnonzero(mask)
    for dname, frac in depths:
        if frac is None:
            reps = [(C[pi], F[pi])]
        else:
            p = np.minimum(1, frac * DM_DEPTH / tot79[pi])[:, None]
            reps = [(np.random.default_rng(s).binomial(C[pi], p), np.random.default_rng(s + 7).binomial(F[pi], p))
                    for s in DM_SEEDS]
        Rs = []
        for si, (Cs, Fs) in enumerate(reps):
            b, s = feats(ver, Cs, Fs, idx)
            R1 = np.corrcoef(np.column_stack([b, s]).T)
            Rs.append(R1)
            o1, p1 = perm(R1[NT:, :NT], NT)
            seeds_rows.append(dict(version=vname, depth=dname, seed=si, diag_minus_off=o1, perm_p=p1,
                                   own_best_row=int(sum(np.argmax(R1[NT + i, :NT]) == i for i in range(NT)))))
        R = np.nanmean(Rs, 0)
        blk = R[NT:, :NT]
        obs, pv = perm(blk, NT)
        keep = list(range(1, NT))
        o5, p5 = perm(blk[np.ix_(keep, keep)], NT - 1)
        Z, nxt = cluster_next(R, NT)
        own_r = [int(np.argmax(blk[i]) == i) for i in range(NT)]
        own_c = [int(np.argmax(blk[:, j]) == j) for j in range(NT)]
        tag = f"{vname}_{dname}"
        pd.DataFrame(R, labs, labs).to_csv(O3 / f"r3_corr_{tag}.tsv", sep="\t", float_format="%.4f")
        rows.append(dict(version=vname, depth=dname, n_regions=len(idx), diag_minus_off=obs, perm_p=pv,
                         own_best_row=sum(own_r), own_best_col=sum(own_c), clusters_next_to_own=sum(nxt),
                         noCTCF_diag_minus_off=o5, noCTCF_perm_p=p5,
                         bulk_bulk_mean_off=R[:NT, :NT][~np.eye(NT, dtype=bool)].mean(),
                         sc_sc_mean_off=R[NT:, NT:][~np.eye(NT, dtype=bool)].mean()))
        for i, t in enumerate(TFS):
            pertf.append(dict(version=vname, depth=dname, sc_track=t, r_own=blk[i, i],
                              r_best_other=float(np.max(np.delete(blk[i], i))),
                              best_bulk=TFS[int(np.argmax(blk[i]))], clusters_next_to_own=nxt[i]))
        print(f"{tag:34s} n={len(idx):6d} d-o={obs:+.4f} p={pv:.4f} rows={sum(own_r)} cols={sum(own_c)} "
              f"next={sum(nxt)} noCTCF p={p5:.3f}", flush=True)
        if dname not in ("znf777", "full"):
            continue
        # 12x12 clustered
        fig = plt.figure(figsize=(3.4, 3.5))
        axd = fig.add_axes([0.27, 0.78, 0.53, 0.11])
        d = dendrogram(Z, ax=axd, no_labels=True, color_threshold=0, above_threshold_color=INK)
        axd.axis("off")
        o = d["leaves"]
        axh = fig.add_axes([0.27, 0.22, 0.53, 0.56])
        lim = 1 if ver == "P1" else max(0.1, np.abs(R[~np.eye(2 * NT, dtype=bool)]).max())
        im = axh.imshow(R[np.ix_(o, o)], cmap=P.div_cmap(), vmin=-lim, vmax=lim, aspect="auto")
        ol = [labs[k] for k in o]
        axh.set_xticks(range(len(o)), ol, rotation=90)
        axh.set_yticks(range(len(o)), ol)
        for tl in axh.get_xticklabels() + axh.get_yticklabels():
            tl.set_color(INK if tl.get_text().startswith("Bulk") else P.arms.raw)
        axh.tick_params(length=0)
        cax = fig.add_axes([0.83, 0.22, 0.025, 0.25])
        fig.colorbar(im, cax=cax, label="Pearson r")
        dep = "ZNF777 depth" if dname == "znf777" else "full depth"
        fig.suptitle(f"{vname.replace('_', ' ')}, {dep}\nown bulk best for {sum(own_r)}/6 sc; "
                     f"diag−off {obs:+.3f}, p = {pv:.3f}", fontsize=6, y=0.995)
        for ext in ("png", "pdf"):
            fig.savefig(O3 / "figs" / f"r3_cluster_{tag}.{ext}", dpi=600 if ext == "png" else None)
        plt.close(fig)
        # 6x6 block
        fig, ax = plt.subplots(figsize=(2.6, 2.4))
        bl = max(0.02, np.abs(blk).max())
        im = ax.imshow(blk, cmap=P.div_cmap(), vmin=-bl, vmax=bl)
        for a in range(NT):
            for b2 in range(NT):
                ax.text(b2, a, f"{blk[a, b2]:+.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(blk[a, b2]) > 0.6 * bl else INK)
            ax.add_patch(Rectangle((a - .5, a - .5), 1, 1, fill=False, ec=INK, lw=0.8))
        ax.set_xticks(range(NT), [f"Bulk {t}" for t in TFS], rotation=90)
        ax.set_yticks(range(NT), [f"sc {t}" for t in TFS])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Pearson r")
        ax.set_title(f"{vname.replace('_', ' ')}, {dep}\nrows {sum(own_r)}/6, cols {sum(own_c)}/6, "
                     f"p = {pv:.3f} (no CTCF p = {p5:.3f})", fontsize=6)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(O3 / "figs" / f"r3_block_{tag}.{ext}", dpi=600 if ext == "png" else None)
        plt.close(fig)

summ = pd.DataFrame(rows)
summ.to_csv(O3 / "r3_summary.tsv", sep="\t", index=False, float_format="%.4f")
pd.DataFrame(pertf).to_csv(O3 / "r3_pertf.tsv", sep="\t", index=False, float_format="%.4f")
pd.DataFrame(seeds_rows).to_csv(O3 / "r3_per_seed.tsv", sep="\t", index=False, float_format="%.4f")

# depth curve figure
fig, ax = plt.subplots(figsize=(3.2, 2.2))
xs = [0.25, 0.5, 1.0]
for vname, *_ in versions:
    v = summ[(summ.version == vname) & summ.depth.str.startswith("znf777")].set_index("depth")
    y = [v.loc[d, "diag_minus_off"] for d in ("znf777_25pct", "znf777_50pct", "znf777")]
    ax.plot(xs, y, marker="o", ms=3, lw=0.8, label=vname.replace("_", " "))
ax.axhline(0, color=P.neutral.light, lw=0.5)
ax.set_xticks(xs, ["25%", "50%", "100%"])
ax.set_xlabel("Fraction of ZNF777 depth (420,339 reads)")
ax.set_ylabel("Pearson diag − off-diag (sc vs bulk)")
ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(O3 / "figs" / f"r3_depth_curve.{ext}", dpi=600 if ext == "png" else None)
pd.set_option("display.width", 250)
print(summ.round(4).to_string())
