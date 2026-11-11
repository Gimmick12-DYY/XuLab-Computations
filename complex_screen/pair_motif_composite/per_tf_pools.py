# Per-TF pools for de novo motif discovery on imputed data. For each of the 79 panel TFs: top 10,000 eligible union-ATAC bins ranked on the
# imputed signal itself (no 79-TF mean subtraction). Background: other eligible union-ATAC bins not in that TF's top set, random sample of
# 3 per foreground bin, not stratified here (HOMER applies its default GC normalization). Sequences = bin centre +/- 100 bp.
# Discovery = odd autosomes (disc/), held-out = even autosomes + chrX (val/).
# Output: per_tf_matched/pools/<TF>/{disc,val}/{fg,bg}.bed, per_tf_matched/tf_pools.tsv, per_tf_matched/tasks_tf.tsv
import os
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = f"{O}/per_tf_matched"; M = f"{X}/complex_screen/motif_shared_20261103/fg"
TOP = 10000; BGX = 3; rng = np.random.default_rng(20261111)
z = np.load(f"{M}/signal_mask.npz"); S = z["S"]; tfs = [t.upper() for t in z["tfs"]]
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t"); num = Q.chrom.str.replace("chr", "")
part = np.where(num.isin([str(i) for i in range(1, 23, 2)]), "disc", np.where(num.isin([str(i) for i in range(2, 23, 2)] + ["X"]), "val", "none"))
elig = Q.n_ok.values & ~Q.blacklist.values & (Q.exonic.values <= 0.5) & (part != "none"); ei = np.flatnonzero(elig)
prom = np.zeros(len(Q), bool); prom[[int(x) for x in open(f"{O}/ref/promoter_bins.txt").read().split()]] = True
def wbed(fn, idx):
    with open(fn, "w") as f:
        for i in sorted(idx, key=lambda i: (Q.chrom[i], Q.s[i])): f.write(f"{Q.chrom[i]}\t{Q.s[i]}\t{Q.e[i]}\tb{Q.bin[i]}\t.\t+\n")
rows, tasks = [], []
for k, t in enumerate(tfs):
    topi = ei[np.argsort(-S[k][ei], kind="stable")[:TOP]]; ins = np.zeros(len(Q), bool); ins[topi] = True; r = dict(tf=t, min_signal_top=round(float(S[k][topi].min()), 4))
    for pt in ["disc", "val"]:
        fg = topi[part[topi] == pt]; rest = ei[(part[ei] == pt) & ~ins[ei]]; bg = rng.choice(rest, min(BGX * len(fg), len(rest)), replace=False); d = f"{G}/pools/{t}/{pt}"; os.makedirs(d, exist_ok=True); wbed(f"{d}/fg.bed", fg); wbed(f"{d}/bg.bed", bg)
        r.update({f"{pt}_fg": len(fg), f"{pt}_bg": len(bg), f"{pt}_gc_fg": round(Q.gc.values[fg].mean(), 3), f"{pt}_gc_bg": round(Q.gc.values[bg].mean(), 3), f"{pt}_prom_fg": round(prom[fg].mean(), 3), f"{pt}_prom_bg": round(prom[bg].mean(), 3)})
    rows.append(r); tasks += [f"{t}\tknown", f"{t}\t8", f"{t}\t10", f"{t}\t12", f"{t}\tval_known"]
pd.DataFrame(rows).to_csv(f"{G}/tf_pools.tsv", sep="\t", index=False); open(f"{G}/tasks_tf.tsv", "w").write("\n".join(tasks) + "\n")
print(pd.DataFrame(rows)[["tf", "disc_fg", "disc_bg", "disc_gc_fg", "disc_gc_bg", "disc_prom_fg", "disc_prom_bg", "val_fg", "val_bg"]].to_string())
