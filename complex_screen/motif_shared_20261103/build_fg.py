# Foregrounds for the shared-motif test: per TF, imputed signal in union ATAC bins (CPM within the mask) minus the 79-TF mean, top 5,000 bins, centre +/- 100 bp; background GC-matched mask bins outside the foreground, 2 per foreground bin where available.
import gzip, json, os, subprocess
import numpy as np, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/motif_shared_20261103"; U = f"{X}/unified/work"; tfs = sorted(d for d in os.listdir(U) if os.path.exists(f"{U}/{d}/impute/matrix_csr.npz")); assert len(tfs) == 79
reg = [l.strip() for l in open(f"{U}/nfya/impute/regions.tsv")]; chrom = np.array([r.split(":")[0] for r in reg]); start = np.array([int(r.split(":")[1].split("-")[0]) for r in reg])
mask = np.zeros(len(reg), bool); iv = {}
with gzip.open(f"{X}/data/HEK293T_ATAC_union3.bed.gz", "rt") as f:
    for l in f:
        c, a, b = l.split("\t")[:3]; iv.setdefault(c, []).append((int(a), int(b)))
for c, L in iv.items():
    ix = np.flatnonzero(chrom == c); s = start[ix]; a = np.array(L)
    o = np.zeros(len(ix), np.int64); i0 = np.clip(np.searchsorted(s, a[:, 0] - 999), 0, len(s)); i1 = np.searchsorted(s, a[:, 1] - 1, side="right"); np.add.at(o, i0[i0 < len(s)], 1); e = i1[i1 < len(s)]; np.add.at(o, e, -1); mask[ix[np.cumsum(o) > 0]] = True
print("mask bins", int(mask.sum()), "expected near 104738", flush=True); mi = np.flatnonzero(mask); S = np.zeros((79, len(mi)), np.float32)
for k, t in enumerate(tfs):
    m = sp.load_npz(f"{U}/{t}/impute/matrix_csr.npz"); assert m.shape[0] == len(reg), (t, m.shape); v = np.asarray(m.sum(1)).ravel()[mi].astype(float); S[k] = v / max(v.sum(), 1) * 1e6; print(t, m.shape[1], int(v.sum()), flush=True); del m
np.savez_compressed(f"{O}/fg/signal_mask.npz", S=S, tfs=np.array(tfs), bins=mi); D = S - S.mean(0)
bed = f"{O}/fg/mask_center.bed"
with open(bed, "w") as f:
    for i in mi: f.write(f"{chrom[i]}\t{start[i] + 400}\t{start[i] + 600}\t{i}\n")
fa = subprocess.run(f"module load bedtools >/dev/null 2>&1; bedtools getfasta -nameOnly -tab -fi {X}/downstream/cache/hg38.fa -bed {bed}", shell=True, capture_output=True, text=True, executable="/bin/bash"); assert fa.returncode == 0, fa.stderr
seqs = {}; 
for l in fa.stdout.splitlines():
    n, q = l.split("\t"); seqs[int(n.split("::")[0])] = q.upper()
sq = [seqs[i] for i in mi]; ok = np.array([("N" not in q) and len(q) == 200 for q in sq]); gc = np.array([(q.count("G") + q.count("C")) / 200 for q in sq]); gb = np.round(gc * 50).astype(int); rng = np.random.default_rng(20261103); rows = []
for k, t in enumerate(tfs):
    o = np.argsort(-D[k], kind="stable"); o = o[ok[o]][:5000]; d = f"{O}/fg/{t}"; os.makedirs(d); inf = np.zeros(len(mi), bool); inf[o] = True; bg = []; short = 0
    for b, n in zip(*np.unique(gb[o], return_counts=True)):
        pool = np.flatnonzero((gb == b) & ~inf & ok); want = 2 * n; short += max(want - len(pool), 0); bg.append(rng.choice(pool, min(want, len(pool)), replace=False))
    bg = np.concatenate(bg)
    for nm, ix in (("fg", o), ("bg", bg)):
        with open(f"{d}/{nm}.fa", "w") as f:
            for i in ix: f.write(f">{chrom[mi[i]]}:{start[mi[i]]}\n{sq[i]}\n")
    rows.append(f"{t}\t{len(o)}\t{len(bg)}\t{short}\t{gc[o].mean():.3f}\t{gc[bg].mean():.3f}\t{D[k][o[-1]]:.3f}\t{np.corrcoef(S[k], S.mean(0))[0, 1]:.3f}")
open(f"{O}/fg/summary.tsv", "w").write("tf\tn_fg\tn_bg\tbg_short\tgc_fg\tgc_bg\tmin_diff\tr_with_mean\n" + "\n".join(rows) + "\n"); open(f"{O}/tfs.txt", "w").write("\n".join(tfs) + "\n"); print("done", flush=True)
