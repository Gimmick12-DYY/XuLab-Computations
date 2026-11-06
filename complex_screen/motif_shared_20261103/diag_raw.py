# Control check, raw reads: lab MACS2 summits at p <= 1e-5, top 5,000 by -log10 p, summit +/- 50 bp; background GC-matched union ATAC centres (100 bp). Read-only from the lab folder.
import glob, gzip, os, subprocess, sys
import numpy as np
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/motif_shared_20261103"; Dg = f"{O}/diag/raw_5000"; M = "/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2"; os.makedirs(Dg, exist_ok=False); rng = np.random.default_rng(11)
def fa(bed):
    r = subprocess.run(f"module load bedtools >/dev/null 2>&1; bedtools getfasta -nameOnly -tab -fi {X}/downstream/cache/hg38.fa -bed {bed}", shell=True, capture_output=True, text=True, executable="/bin/bash"); assert r.returncode == 0, r.stderr; return [l.split("\t")[1].upper() for l in r.stdout.splitlines()]
Z = np.load(f"{O}/fg/signal_mask.npz"); mi = Z["bins"]; reg = [l.strip() for l in open(f"{X}/unified/work/nfya/impute/regions.tsv")]; b0 = f"{Dg}/bg.bed"
with open(b0, "w") as f:
    for i in mi: c, r = reg[i].split(":"); a = int(r.split("-")[0]); f.write(f"{c}\t{a + 450}\t{a + 550}\t{i}\n")
B = fa(b0); ok = np.array(["N" not in q for q in B]); gb = np.round(np.array([(q.count("G") + q.count("C")) / max(len(q), 1) for q in B]) * 50).astype(int)
for t in ["ctcf", "znf143", "nfya", "nrf1", "maz", "mlx", "srebf2"]:
    np_ = [f"{M}/TF.{t.upper()}/TF.{t.upper()}_peaks.narrowPeak"]; assert os.path.exists(np_[0]), np_
    op = open; rows = []
    with op(np_[0], "rt") as f:
        for l in f:
            p = l.split("\t")
            if float(p[7]) >= 5 and "_" not in p[0]: s = int(p[1]) + int(p[9]); rows.append((float(p[7]), p[0], s))
    rows = sorted(rows, reverse=True)[:5000]; d = f"{Dg}/{t}"; os.makedirs(d); bed = f"{d}/fg.bed"
    with open(bed, "w") as f:
        for k, (lp, c, s) in enumerate(rows): f.write(f"{c}\t{max(s - 50, 0)}\t{s + 50}\t{k}\n")
    Q = [q for q in fa(bed) if "N" not in q and len(q) == 100]; g = np.round(np.array([(q.count("G") + q.count("C")) / 100 for q in Q]) * 50).astype(int); bg = []
    for b, m in zip(*np.unique(g, return_counts=True)):
        pool = np.flatnonzero((gb == b) & ok)
        if len(pool): bg.append(rng.choice(pool, min(2 * m, len(pool)), replace=False))
    with open(f"{d}/fg.fa", "w") as f:
        for k, q in enumerate(Q): f.write(f">{k}\n{q}\n")
    with open(f"{d}/bg.fa", "w") as f:
        for i in np.concatenate(bg): f.write(f">{i}\n{B[i]}\n")
    print(t, "sites", len(Q), "gc fg", round(g.mean() / 50, 3), flush=True)
