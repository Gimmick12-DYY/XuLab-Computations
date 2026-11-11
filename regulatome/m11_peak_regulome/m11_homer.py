# m11 Amendment 4: HOMER known-motif enrichment per TF and peak set (top 2,000 summits by signalValue inside union ATAC, +/- 100 bp) vs 20,000 union-ATAC positions.
# usage: m11_homer.py prep -> recall/homer/<TF>_<set>.bed, bg.bed, jobs.txt ; m11_homer.py rank -> recall_homer_ranks.tsv
import os, sys, glob
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m11_peak_regulome"; RC = f"{O}/recall"; H = f"{RC}/homer"; os.makedirs(H, exist_ok=True); QMIN = -np.log10(0.05)
TFS = ["NRF1", "CTCF", "ZNF143", "MAZ", "NFYA", "E4F1"]; ALIAS = {"NRF1": ["NRF1", "NRF("], "CTCF": ["CTCF("], "ZNF143": ["ZNF143", "STAF"], "MAZ": ["MAZ"], "NFYA": ["NFY"], "E4F1": ["E4F1"]}
SETS = {"V0": lambda t: f"{RC}/{t}/TF.{t}_peaks.narrowPeak", "V1": lambda t: f"{RC}/V1/{t}/TF.{t}_peaks.narrowPeak", "V2": lambda t: f"{RC}/V2/{t}/TF.{t}_peaks.narrowPeak", "labq005": lambda t: f"/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"}
UAdf = pd.read_csv(f"{X}/data/HEK293T_ATAC_union3.bed.gz", sep="\t", header=None, usecols=[0, 1, 2]); UAdf.columns = ["c", "s", "e"]
if sys.argv[1] == "prep":
    UA = {}
    for c, G in UAdf.groupby("c"): iv = G[["s", "e"]].values; UA[c] = iv[np.argsort(iv[:, 0])]
    def inA(c, p):
        r = np.zeros(len(p), bool)
        for ch in np.unique(c):
            if ch not in UA: continue
            m = c == ch; iv = UA[ch]; k = np.searchsorted(iv[:, 0], p[m], side="right") - 1; ok = k >= 0; ok[ok] = p[m][ok] < np.maximum.accumulate(iv[:, 1])[k[ok]]; r[m] = ok
        return r
    jobs = []
    for t in TFS:
        for s, f in SETS.items():
            P = pd.read_csv(f(t), sep="\t", header=None, usecols=[0, 1, 6, 8, 9]); P.columns = ["c", "s", "sig", "q", "off"]
            if s == "labq005": P = P[P.q >= QMIN]
            P["sm"] = P.s + P.off; P = P[inA(P.c.values.astype(str), P.sm.values)].sort_values("sig", ascending=False).head(2000)
            pd.DataFrame(dict(c=P.c, a=P.sm - 100, b=P.sm + 100, n=[f"p{i}" for i in range(len(P))], x=0, st="+")).to_csv(f"{H}/{t}_{s}.bed", sep="\t", header=False, index=False); jobs.append(f"{t}_{s}")
    rng = np.random.default_rng(20261111); L = (UAdf.e - UAdf.s).values; k = rng.choice(len(L), 20000, p=L / L.sum()); pos = UAdf.s.values[k] + (rng.random(20000) * L[k]).astype(int)
    pd.DataFrame(dict(c=UAdf.c.values[k], a=pos - 100, b=pos + 100, n=[f"b{i}" for i in range(20000)], x=0, st="+")).to_csv(f"{H}/bg.bed", sep="\t", header=False, index=False)
    open(f"{H}/jobs.txt", "w").write("\n".join(jobs) + "\n"); print(len(jobs))
else:
    rows = []
    for j in open(f"{H}/jobs.txt").read().split():
        f = f"{H}/{j}/knownResults.txt"
        if not os.path.exists(f): rows.append(dict(job=j, status="missing")); continue
        K = pd.read_csv(f, sep="\t"); t, s = j.split("_"); nm = K.iloc[:, 0].astype(str); hit = [i for i, x in enumerate(nm) if any(a.upper() in x.upper() for a in ALIAS[t])]
        rows.append(dict(label="diagnostic, benchmark already read", tf=t, peak_set=s, n_known_motifs=len(K), best_own_rank=(hit[0] + 1) if hit else np.nan, own_motif_name=nm.iloc[hit[0]] if hit else "not in HOMER known set",
                         own_log_p=K.iloc[hit[0], 3] if hit else np.nan, own_pct_target=K.iloc[hit[0], 6] if hit else "", own_pct_bg=K.iloc[hit[0], 8] if hit else "", top1_motif=nm.iloc[0]))
    pd.DataFrame(rows).to_csv(f"{O}/recall_homer_ranks.tsv", sep="\t", index=False); print(pd.DataFrame(rows).to_string(index=False))
