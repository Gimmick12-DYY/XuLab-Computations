# m10_promoter_regulome Amendment 2, G1-G3 (diagnostic, benchmark already read; no model fitting).
# usage: m10_g.py fimo <i>    own-motif FIMO (p < 1e-4) over open chromatin and over promoters (TSS +/- 500 bp) for motif TF i -> work/g/sites_<TF>.tsv, psites_<TF>.tsv
#        m10_g.py lists       site lists (motif, ChIP summits) + ATAC-decile-matched background positions -> work/g/lists.npz
#        m10_g.py count <i>   fragment 5' ends of TF i within +/- 100 bp of every list's sites; promoter own-motif counts per gene -> work/g/count_<TF>.npz
#        m10_g.py eval        -> G1_signal_fraction.tsv, G2_lists.tsv, G2_counts.tsv, G3_ctcf_groups.tsv
import json, os, subprocess, sys
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m10_promoter_regulome"; M9 = f"{RG}/m9_abc_diagnosis"; W = f"{O}/work/g"; os.makedirs(W, exist_ok=True)
LAB = "diagnostic, benchmark already read"; SEED = 20261111; FIMO = "/nas/longleaf/rhel9/apps/meme/5.5.7/bin/fimo"; BEDTOOLS = "/nas/longleaf/rhel9/apps/bedtools/2.31.1/bedtools2/bin/bedtools"
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]; MOT = sorted(f[:-5] for f in os.listdir(f"{M9}/work/motifs") if f.endswith(".meme"))
EL = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/EnhancerList.txt", sep="\t")
acol = [c for c in EL.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]; EL = EL[["chr", "start", "end", acol]].rename(columns={acol: "atac"}).reset_index(drop=True)
GL = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t", usecols=["chr", "name", "tss"]).dropna()
TSS = {c: np.sort(G.tss.values.astype(int)) for c, G in GL.groupby("chr")}
def near_tss(c, p):
    t = TSS.get(c)
    if t is None: return np.zeros(len(p), bool)
    k = np.searchsorted(t, p); d = np.minimum(np.abs(p - t[np.clip(k - 1, 0, len(t) - 1)]), np.abs(p - t[np.clip(k, 0, len(t) - 1)])); return d <= 1000
mode = sys.argv[1]
if mode == "fimo":
    t = MOT[int(sys.argv[2])]; src = open(f"{M9}/work/motifs/{t}.meme").read(); head, rest = src.split("MOTIF", 1); real = "MOTIF" + rest.split("\nMOTIF")[0]
    mf = f"{W}/motif_{t}.meme"; open(mf, "w").write(head + real + "\n")
    if not os.path.exists(f"{W}/open.fa"):
        EL.assign(name=np.arange(len(EL)))[["chr", "start", "end", "name"]].to_csv(f"{W}/open_{t}.bed", sep="\t", header=False, index=False)
        subprocess.run(f"{BEDTOOLS} getfasta -fi {X}/downstream/cache/hg38.fa -bed {W}/open_{t}.bed -nameOnly -fo {W}/open_{t}.fa && mv -n {W}/open_{t}.fa {W}/open.fa; rm -f {W}/open_{t}.bed {W}/open_{t}.fa", shell=True, check=True)
    for fa, out in ((f"{W}/open.fa", f"{W}/sites_{t}.tsv"), (f"{M9}/work/promoters.fa", f"{W}/psites_{t}.tsv")):
        subprocess.run(f"{FIMO} --thresh 1e-4 --text --bfile --uniform-- {mf} {fa} 2>/dev/null | awk -F'\\t' 'NR>1{{print $3\"\\t\"$4\"\\t\"$5}}' > {out}", shell=True, check=True)
    print(t, sum(1 for _ in open(f"{W}/sites_{t}.tsv")), sum(1 for _ in open(f"{W}/psites_{t}.tsv")))
elif mode == "lists":
    rng = np.random.default_rng(SEED); dec = pd.qcut(EL.atac.rank(method="first"), 10, labels=False).values; L = {}
    def add(name, ei, pos):
        c = EL.chr.values[ei]; o = np.lexsort((pos, c)); ei, pos, c = ei[o], pos[o], c[o]; pr = np.zeros(len(pos), bool)
        for ch in np.unique(c): m = c == ch; pr[m] = near_tss(ch, pos[m])
        L[name] = dict(chr=c, pos=pos, prom=pr)
        # background: same count, uniform positions in open elements of the same ATAC decile with no site of this list
        bad = np.zeros(len(EL), bool); bad[ei] = True; ok = ~bad; cnt = np.bincount(dec[ei], minlength=10); bi = []
        for d_, n_ in enumerate(cnt):
            pool = np.flatnonzero(ok & (dec == d_))
            if n_ and len(pool): bi.append(rng.choice(pool, n_, replace=len(pool) < n_))
        bi = np.concatenate(bi) if bi else np.array([], int); bp = EL.start.values[bi] + (rng.random(len(bi)) * (EL.end.values[bi] - EL.start.values[bi])).astype(int); bc = EL.chr.values[bi]; o = np.lexsort((bp, bc)); bi, bp, bc = bi[o], bp[o], bc[o]
        bpr = np.zeros(len(bp), bool)
        for ch in np.unique(bc): m = bc == ch; bpr[m] = near_tss(ch, bp[m])
        L[name + "__bg"] = dict(chr=bc, pos=bp, prom=bpr)
    for t in MOT:
        S = pd.read_csv(f"{W}/sites_{t}.tsv", sep="\t", header=None, names=["e", "s", "t"]); ei = S.e.values.astype(int); add(f"motif_{t}", ei, EL.start.values[ei] + (S.s.values + S.t.values) // 2 - 1)
    for f in sorted(os.listdir(f"{X}/downstream/tracks/bulk_chipseq")):
        if not f.endswith("_peaks.bed"): continue
        t = f[:-10]; P = pd.read_csv(f"{X}/downstream/tracks/bulk_chipseq/{f}", sep="\t", header=None); sm = (P[1] + P[9]).values; ei = []; ps = []
        for ch, G in EL.groupby("chr"):
            m = P[0].values == ch
            if not m.any(): continue
            st, en, ix = G.start.values, G.end.values, G.index.values; o = np.argsort(st); st, en, ix = st[o], en[o], ix[o]; s = sm[m]; k = np.searchsorted(st, s, side="right") - 1; ok = (k >= 0) & (s < en[np.clip(k, 0, None)])
            ei += list(ix[k[ok]]); ps += list(s[ok])
        add(f"chip_{t}_{'K562' if t == 'NFYA' else 'HEK293'}", np.array(ei, int), np.array(ps, int))
    np.savez(f"{W}/lists.npz", **{f"{k}|{f}": v[f] for k, v in L.items() for f in ("chr", "pos", "prom")}); print({k: len(v["pos"]) for k, v in L.items()})
elif mode == "count":
    t = tfs79[int(sys.argv[2])]; Z = np.load(f"{W}/lists.npz", allow_pickle=True); names = sorted({k.split("|")[0] for k in Z.files})
    F = pd.read_csv(f"/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment/TF.{t}.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 5], names=["c", "s", "e", "st"], dtype={"c": str, "s": np.int64, "e": np.int64, "st": str})
    pos = np.where(F.st.values == "+", F.s.values, F.e.values - 1); POS = {c: np.sort(pos[ix]) for c, ix in F.groupby("c", sort=False).indices.items()}; out = dict(total=len(F))
    def per_site(c, p):
        r = np.zeros(len(p), np.int32)
        for ch in np.unique(c):
            m = c == ch
            if ch in POS: q = POS[ch]; r[m] = np.searchsorted(q, p[m] + 100, side="right") - np.searchsorted(q, p[m] - 100, side="left")
        return r
    def union_count(c, p):  # fragments within +/- 100 bp of any site (no double counting)
        n = 0
        for ch in np.unique(c):
            if ch not in POS: continue
            s = np.sort(p[c == ch]); a, b = s - 100, s + 100; ms = []
            for x, y in zip(a, b):
                if ms and x <= ms[-1][1]: ms[-1][1] = max(ms[-1][1], y)
                else: ms.append([x, y])
            ms = np.array(ms); q = POS[ch]; n += int((np.searchsorted(q, ms[:, 1], side="right") - np.searchsorted(q, ms[:, 0], side="left")).sum())
        return n
    for nm in names:
        c, p, pr = Z[f"{nm}|chr"], Z[f"{nm}|pos"], Z[f"{nm}|prom"]; out[f"{nm}|site"] = per_site(c, p)
        for sub, m in (("all", np.ones(len(p), bool)), ("prom", pr), ("dist", ~pr)): out[f"{nm}|union_{sub}"] = union_count(c[m], p[m])
    # (ii) own-motif promoter counts per gene (fragments in TSS +/- 500 bp within +/- 100 bp of an own-motif promoter site)
    Pm = pd.read_csv(f"{M9}/work/promoters.bed", sep="\t", header=None, names=["c", "s", "e", "gene", "x", "strand"]); g2 = np.zeros(len(Pm))
    if os.path.exists(f"{W}/psites_{t}.tsv") and os.path.getsize(f"{W}/psites_{t}.tsv") > 0:
        S = pd.read_csv(f"{W}/psites_{t}.tsv", sep="\t", header=None, names=["gene", "s", "t"]); gi = pd.Series(np.arange(len(Pm)), index=Pm.gene)
        for g, H in S.groupby("gene"):
            i = gi[g]; c = Pm.c.values[i]; w0 = Pm.s.values[i]
            if c not in POS: continue
            ctr = w0 + (H.s.values + H.t.values) // 2 - 1; q = POS[c]; f = q[(q >= w0) & (q < Pm.e.values[i])]; g2[i] = int(sum(1 for x in f if np.any(np.abs(ctr - x) <= 100)))
    out["prom_motif_counts"] = g2; out["prom_genes"] = Pm.gene.values
    # (i) genes with one of our q <= 0.05 MACS2 summits within 1 kb of the TSS
    pk = f"/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"; gp = np.zeros(len(Pm), bool)
    if os.path.exists(pk):
        K = pd.read_csv(pk, sep="\t", header=None, usecols=[0, 1, 8, 9]); K = K[K[8] >= -np.log10(0.05)]; tss = (Pm.s + 500).values
        for ch, G in K.groupby(0):
            sm = np.sort((G[1] + G[9]).values); m = Pm.c.values == ch
            if m.any(): gp[m] = (np.searchsorted(sm, tss[m] + 1000, side="right") - np.searchsorted(sm, tss[m] - 1000, side="left")) > 0
    out["peak_within_1kb"] = gp
    np.savez(f"{W}/count_{t}.npz", **out); print(t, len(F), flush=True)
