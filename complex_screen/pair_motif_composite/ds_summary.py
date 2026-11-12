# Depth downsampling summary: own-motif rank / fold vs cells per TF (median over seeds), passing = rank <= 10 on both halves; figure with the
# panel's library-size quartiles (q25 1,508; median 3,331; q75 5,837 cells) marked. Output downsample/summary.tsv, downsample/own_motif_vs_depth.{png,svg}
import json, glob, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
X = "/work/users/d/y/dyy12/XuLab"; D = f"{X}/complex_screen/pair_motif_composite/downsample"; sys.path.insert(0, f"{X}/figures_paletteC"); import regulome_palette as rp; P = rp.use()
T = pd.DataFrame([json.load(open(f)) for f in glob.glob(f"{D}/res/*.json")]); T["pass"] = (T.disc_own_rank <= 10) & (T.val_own_rank <= 10)
S = T.groupby(["tf", "n_cells"]).agg(seeds=("seed", "size"), reads=("n_reads", "median"), peaks=("n_peaks_p1e5", "median"), disc_rank=("disc_own_rank", "median"), val_rank=("val_own_rank", "median"),
                                    disc_fold=("disc_own_fold", "median"), val_fold=("val_own_fold", "median"), frac_pass=("pass", "mean")).reset_index(); S.to_csv(f"{D}/summary.tsv", sep="\t", index=False)
pd.set_option("display.width", 200); print(S.to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6)); cols = plt.cm.tab10.colors
for k, (t, g) in enumerate(S.groupby("tf")):
    ax[0].plot(g.n_cells, (g.disc_rank + g.val_rank) / 2, "o-", ms=3, color=cols[k % 10], label=t); ax[1].plot(g.n_cells, (g.disc_fold + g.val_fold) / 2, "o-", ms=3, color=cols[k % 10])
for a in ax:
    a.set_xscale("log"); a.set_xlabel("cells per library (subsampled)")
    for x, ls in [(1508, ":"), (3331, "--"), (5837, ":")]: a.axvline(x, color=P.neutral.mid, lw=.6, ls=ls)
ax[0].set_yscale("log"); ax[0].invert_yaxis(); ax[0].axhline(10, color=P.evidence.hic_loop, lw=.6); ax[0].set_ylabel("own-motif rank (HOMER known, of 472)"); ax[1].set_ylabel("own-motif fold (sites / background)"); ax[1].axhline(1, color=P.neutral.mid, lw=.5)
ax[0].legend(fontsize=5, ncol=2); fig.suptitle("Own-motif recovery vs depth; dashed = panel median library, dotted = quartiles", fontsize=6); fig.tight_layout(); fig.savefig(f"{D}/own_motif_vs_depth.png", dpi=200); fig.savefig(f"{D}/own_motif_vs_depth.svg")
