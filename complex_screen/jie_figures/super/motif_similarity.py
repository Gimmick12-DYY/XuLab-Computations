# Column-wise Pearson similarity between JASPAR motifs, best offset and orientation at >= 5 overlapping columns; empirical p = rank of the target among all motifs in the JASPAR file for that query.
import numpy as np
txt = open("img/jaspar.meme").read(); blk = {b.split()[0]: b for b in txt.split("MOTIF ")[1:]}
def pwm(m):
    L = blk[m].strip().split("\n"); i = [k for k, l in enumerate(L) if l.startswith("letter-prob")][0]; return np.array([[float(x) for x in l.split()] for l in L[i + 1:] if l.strip() and not l.startswith("URL")])
PW = {m: pwm(m) for m in blk}
def col(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True); d = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)); return np.where(d > 0, (a * b).sum(1) / np.maximum(d, 1e-12), 0)
def best(q, t, mn=5):
    b = (-9, None)
    for rc, T in ((False, t), (True, t[::-1, ::-1])):
        for off in range(-(len(T) - mn), len(q) - mn + 1):
            i0, j0 = max(off, 0), max(-off, 0); n = min(len(q) - i0, len(T) - j0)
            if n < mn: continue
            s = col(q[i0:i0 + n], T[j0:j0 + n]).sum()
            if s > b[0]: b = (s, (off, rc, n, col(q[i0:i0 + n], T[j0:j0 + n]).mean()))
    return b
for q, t, l in (("MA0663.1", "MA0828.3", "MLX vs SREBF2 E-box"), ("MA0663.1", "MA0596.1", "MLX vs SREBF2 SRE"), ("MA0828.3", "MA0596.1", "SREBF2 E-box vs SRE")):
    S = {m: best(PW[q], PW[m])[0] for m in PW if m != q}; s, (off, rc, n, r) = best(PW[q], PW[t]); rk = sum(v >= s for v in S.values()); print(f"{l}: sum r {s:.2f}, mean column r {r:.2f}, overlap {n}, offset {off}, reverse strand {rc}, rank {rk}/{len(S)}, empirical p {rk / len(S):.4f}")
