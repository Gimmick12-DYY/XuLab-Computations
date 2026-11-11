#!/usr/bin/env python
"""Round 5 stats: the round-3 6x6 (preferred 2x, top 5000, 30 groups; halves top 2500) for each of the 8 read-filter variants
of r11a_read_variants.py (7 filtered + unfiltered). Depth matching AFTER filtering: each panel TF thinned to ZNF777's read
total under that variant (3 seeds; r averaged); pool = leave-one-out sum of the 79 filtered TFs (full depth).
Reports the 6x6 per variant, own-best rows, diag - offdiag, exact p (720), p without CTCF (120), halves.
Outputs OUT/round5_reads/: variant_summary.tsv, block_<variant>_<half>.tsv, per_tf_read_stats.tsv; corr_matrix PNGs."""
import itertools
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Rectangle

from fig2_common import DM_SEEDS, INP, OUT, ROOT, TFS

O5 = OUT / "round5_reads"
G = 30
NAMES = {0: "none", 1: "L1_dedup", 2: "L2_cellFRiP", 3: "L1+L2", 4: "L3_HOT_blacklist", 5: "L1+L3", 6: "L2+L3", 7: "L1+L2+L3"}
tf79 = [str(t) for t in np.load(INP / "frame.npz", allow_pickle=True)["tf79"]]
S = pd.read_csv(OUT / "round2/summits.tsv", sep="\t", low_memory=False)
Bk = S[[f"bulk_{t}" for t in TFS]].values
mask = S.bulk_spec.values > 1
C = np.zeros((8, 79, len(S)), np.int64); F = np.zeros_like(C); TOT = np.zeros((8, 79))
rs = []
for k, t in enumerate(tf79):
    z = np.load(O5 / "per_tf" / f"{t}.npz")
    C[:, k], F[:, k], TOT[:, k] = z["C"], z["F"], z["tot"]
    rs.append(dict(tf=t, n_reads=int(z["n_reads"]), dup_rate=float(z["dup_rate"]), n_cells=int(z["n_cells"]),
                   n_cells_kept=int(z["n_cells_kept"]), frip_median=float(z["frip_median"]), frac_reads_hot_bl=float(z["frac_reads_hot_bl"]),
                   **{f"reads_{NAMES[v]}": int(z["tot"][v]) for v in range(8)}))
pd.DataFrame(rs).to_csv(O5 / "per_tf_read_stats.tsv", sep="\t", index=False, float_format="%.4f")
pi = [tf79.index(t) for t in TFS]
zi = tf79.index("ZNF777")
cn = S.chrom.str.replace("chr", "").replace("X", "23").astype(int).values
H = {"all": np.ones(len(S), bool), "odd": (cn % 2 == 1) & (cn != 23), "even_X": (cn % 2 == 0) | (cn == 23)}


def groups(half, n):
    gr = []
    for t in TFS:
        ix = np.flatnonzero((S.ref.values == t) & mask & H[half])
        ix = ix[np.argsort(-S.signal.values[ix], kind="stable")][:n]
        gr += [g for g in np.array_split(ix, G) if len(g)]
    return gr


def perm(B):
    k = len(B)
    o_ = ~np.eye(k, dtype=bool)
    st = lambda M: M.diagonal().mean() - M[o_].mean()
    obs = st(B)
    null = np.array([st(B[list(q)]) for q in itertools.permutations(range(k))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


rows, blocks = [], {}
for v in range(8):
    Cv, Fv = C[v], F[v]
    Cp, Fp = Cv.sum(0), Fv.sum(0)
    p = np.minimum(1, TOT[v, zi] / TOT[v, pi])[:, None]
    for half, n in (("all", 5000), ("odd", 2500), ("even_X", 2500)):
        gr = groups(half, n)
        gB = np.array([Bk[g].sum(0) for g in gr])
        bulk = np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)
        Rs = []
        for s in DM_SEEDS:
            Cs = np.random.default_rng(s).binomial(Cv[pi], p); Fs = np.random.default_rng(s + 1).binomial(Fv[pi], p)
            sc = np.zeros((len(gr), 6))
            for a, g in enumerate(gr):
                cp = Cp[g].sum() - Cv[pi][:, g].sum(1); fp = Fp[g].sum() - Fv[pi][:, g].sum(1)
                sc[a] = np.log2((Cs[:, g].sum(1) + .5) / (Fs[:, g].sum(1) + .5)) - np.log2((cp + .5) / (fp + .5))
            Rs.append(np.corrcoef(np.column_stack([bulk, sc]).T)[6:, :6])
        B = np.mean(Rs, 0)
        blocks[(v, half)] = B
        o, pv = perm(B)
        kk = list(range(1, 6)); o5, p5 = perm(B[np.ix_(kk, kk)])
        rows.append(dict(variant=NAMES[v], half=half, own_best_rows=int(sum(np.argmax(B[i]) == i for i in range(6))),
                         own_best_cols=int(sum(np.argmax(B[:, j]) == j for j in range(6))), diag_minus_off=o, perm_p=pv,
                         noCTCF_diag_minus_off=o5, noCTCF_perm_p=p5, depth_matched_to=int(TOT[v, zi]),
                         **{f"r_own_{t}": B[i, i] for i, t in enumerate(TFS)}))
        pd.DataFrame(B, [f"{t} sc" for t in TFS], [f"{t} ChIP" for t in TFS]).to_csv(O5 / f"block_{NAMES[v]}_{half}.tsv", sep="\t", float_format="%.4f")
sm = pd.DataFrame(rows)
sm.to_csv(O5 / "variant_summary.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print(pd.DataFrame(rs)[["tf", "n_reads", "dup_rate", "n_cells", "n_cells_kept", "frac_reads_hot_bl"]].set_index("tf").loc[TFS].round(3).to_string())
print(sm.round(3).to_string())

# render: current (none) next to combined (L1+L2+L3)
for f in (ROOT.parent / "fonts/arial").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402
P = rp.use()
plt.rcParams["font.sans-serif"] = ["Arial"] + plt.rcParams["font.sans-serif"]
plt.rcParams["svg.fonttype"] = "none"
CELL = {"CTCF": "HEK293/HEK293T", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0), layout="constrained")
for ax, v in zip(axs, (0, 7)):
    B = blocks[(v, "all")]
    r = sm[(sm.variant == NAMES[v]) & (sm.half == "all")].iloc[0]
    im = ax.imshow(B, cmap=P.div_cmap(), vmin=-0.5, vmax=0.5)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{B[i, j]:+.2f}", ha="center", va="center", fontsize=5.5, color="white" if abs(B[i, j]) > 0.3 else P.neutral.ink)
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=P.neutral.ink, lw=0.8))
    ax.set_xticks(range(6), [f"{t} ChIP\n({CELL[t]})" for t in TFS], rotation=90, fontsize=5.5)
    ax.set_yticks(range(6), [f"{t} sc" for t in TFS], fontsize=6.5)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(f"{'current reads' if v == 0 else 'dedup + cell filter + HOT/blacklist removed'}\n"
                 f"own best {r.own_best_rows}/6, p = {r.perm_p:.3f} (no CTCF {r.noCTCF_perm_p:.3f})", fontsize=6)
fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02, label="Pearson correlation")
for ext in ("png", "pdf", "svg"):
    fig.savefig(O5 / f"corr_matrix_6x6_current_vs_filtered.{ext}", dpi=600)
