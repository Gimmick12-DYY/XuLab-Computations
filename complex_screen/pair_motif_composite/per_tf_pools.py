# Per-TF matched pools: for each of the 79 panel TFs, its top 10,000 eligible union-ATAC bins (imputed CPM minus the 79-TF mean,
# same eligibility as build_pools.py) vs a background of eligible bins NOT in that TF's top set, matched per stratum
# (GC 0.025 x CpG o/e quintile x promoter/distal, as in gc_match.py): per stratum m = min(n_fg, n_bg) bins from each, so composition is
# identical; foreground bins in strata without enough background are dropped (counted in *_fg_dropped).
# Discovery = odd autosomes (disc/), validation = even autosomes + chrX (val/).
# Output: per_tf_matched/pools/<TF>/{disc,val}/{fg,bg}.bed, per_tf_matched/tf_pools.tsv, per_tf_matched/tasks_tf.tsv
import os
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = f"{O}/per_tf_matched"; M = f"{X}/complex_screen/motif_shared_20261103/fg"
TOP = 10000; rng = np.random.default_rng(20261110)
z = np.load(f"{M}/signal_mask.npz"); S = z["S"]; tfs = [t.upper() for t in z["tfs"]]; D = S - S.mean(0)
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t"); num = Q.chrom.str.replace("chr", "")
part = np.where(num.isin([str(i) for i in range(1, 23, 2)]), "disc", np.where(num.isin([str(i) for i in range(2, 23, 2)] + ["X"]), "val", "none"))
elig = Q.n_ok.values & ~Q.blacklist.values & (Q.exonic.values <= 0.5) & (part != "none"); ei = np.flatnonzero(elig)
prom = np.zeros(len(Q), bool); prom[[int(x) for x in open(f"{O}/ref/promoter_bins.txt").read().split()]] = True
cq = np.quantile(Q.cpg_oe.values[ei], [0.2, 0.4, 0.6, 0.8]); strat = (np.floor(Q.gc.values / 0.025).astype(int) * 10 + np.searchsorted(cq, Q.cpg_oe.values)) * 2 + prom
def wbed(fn, idx):
    with open(fn, "w") as f:
        for i in sorted(idx, key=lambda i: (Q.chrom[i], Q.s[i])): f.write(f"{Q.chrom[i]}\t{Q.s[i]}\t{Q.e[i]}\tb{Q.bin[i]}\t.\t+\n")
rows, tasks = [], []
for k, t in enumerate(tfs):
    topi = ei[np.argsort(-D[k][ei], kind="stable")[:TOP]]; ins = np.zeros(len(Q), bool); ins[topi] = True; r = dict(tf=t)
    for pt in ["disc", "val"]:
        fg0 = topi[part[topi] == pt]; rest = ei[(part[ei] == pt) & ~ins[ei]]; bg = []; fg = []
        for s in np.unique(strat[fg0]):
            f_ = fg0[strat[fg0] == s]; c = rest[strat[rest] == s]; m = min(len(f_), len(c))
            if m: fg.append(rng.choice(f_, m, replace=False)); bg.append(rng.choice(c, m, replace=False))
        fg, bg = np.concatenate(fg), np.concatenate(bg); short = len(fg0) - len(fg); d = f"{G}/pools/{t}/{pt}"; os.makedirs(d, exist_ok=True); wbed(f"{d}/fg.bed", fg); wbed(f"{d}/bg.bed", bg)
        r.update({f"{pt}_fg": len(fg), f"{pt}_bg": len(bg), f"{pt}_fg_dropped": short, f"{pt}_gc_fg": round(Q.gc.values[fg].mean(), 3), f"{pt}_gc_bg": round(Q.gc.values[bg].mean(), 3),
                  f"{pt}_cpg_fg": round(Q.cpg_oe.values[fg].mean(), 3), f"{pt}_cpg_bg": round(Q.cpg_oe.values[bg].mean(), 3), f"{pt}_prom_fg": round(prom[fg].mean(), 3), f"{pt}_prom_bg": round(prom[bg].mean(), 3)})
    rows.append(r); tasks += [f"{t}\tknown", f"{t}\t8", f"{t}\t10", f"{t}\t12", f"{t}\tval_known"]
pd.DataFrame(rows).to_csv(f"{G}/tf_pools.tsv", sep="\t", index=False); open(f"{G}/tasks_tf.tsv", "w").write("\n".join(tasks) + "\n")
print(pd.DataFrame(rows)[["tf", "disc_fg", "disc_bg", "disc_fg_dropped", "disc_gc_fg", "disc_gc_bg", "disc_prom_fg", "disc_prom_bg", "val_fg", "val_fg_dropped"]].to_string())
