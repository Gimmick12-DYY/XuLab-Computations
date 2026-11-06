# Control check: does own-motif recovery depend on the window or foreground size? Arms: 1 kb whole bin top 5,000; 200 bp top 1,000; 1 kb top 1,000. Same signal and GC matching as the main run.
import os, subprocess
import numpy as np
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/motif_shared_20261103"; Dg = f"{O}/diag"; os.makedirs(Dg, exist_ok=False); Z = np.load(f"{O}/fg/signal_mask.npz"); S, tfs, mi = Z["S"], list(Z["tfs"]), Z["bins"]; reg = [l.strip() for l in open(f"{X}/unified/work/nfya/impute/regions.tsv")]; D = S - S.mean(0)
with open(f"{Dg}/bins1k.bed", "w") as f:
    for i in mi: c, r = reg[i].split(":"); a, b = r.split("-"); f.write(f"{c}\t{a}\t{b}\t{i}\n")
fa = subprocess.run(f"module load bedtools >/dev/null 2>&1; bedtools getfasta -nameOnly -tab -fi {X}/downstream/cache/hg38.fa -bed {Dg}/bins1k.bed", shell=True, capture_output=True, text=True, executable="/bin/bash"); assert fa.returncode == 0; Sq = [l.split("\t")[1].upper() for l in fa.stdout.splitlines()]; assert len(Sq) == len(mi); c2 = [q[400:600] for q in Sq]; rng = np.random.default_rng(7); ctrl = ["ctcf", "znf143", "nfya", "nrf1", "maz", "mlx", "srebf2"]
for w, Q in (("1kb", Sq), ("200", c2)):
    ok = np.array(["N" not in q for q in Q]); gb = np.round(np.array([(q.count("G") + q.count("C")) / len(q) for q in Q]) * 50).astype(int)
    for n in (5000, 1000):
        if w == "200" and n == 5000: continue
        for t in ctrl:
            k = tfs.index(t); o = np.argsort(-D[k], kind="stable"); o = o[ok[o]][:n]; inf = np.zeros(len(mi), bool); inf[o] = True; bg = []
            for b, m in zip(*np.unique(gb[o], return_counts=True)):
                pool = np.flatnonzero((gb == b) & ~inf & ok); bg.append(rng.choice(pool, min(2 * m, len(pool)), replace=False))
            d = f"{Dg}/{w}_{n}/{t}"; os.makedirs(d)
            for nm, ix in (("fg", o), ("bg", np.concatenate(bg))):
                with open(f"{d}/{nm}.fa", "w") as f:
                    for i in ix: f.write(f">{i}\n{Q[i]}\n")
