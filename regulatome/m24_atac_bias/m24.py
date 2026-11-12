# m24_atac_bias (SPEC.md): Part A per-TF open-chromatin diagnostics; Part B ranking of peaks by signalValue / TF:ATAC ratio / ATAC residual / ATAC alone vs own motif and bulk ChIP. Diagnostic.
import json, os, subprocess
import numpy as np, pandas as pd
from scipy.stats import rankdata
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m24_atac_bias"; W = f"{O}/work"; OUT = f"{O}/out"; LAB = "diagnostic"
V = "/vast/som/xujie_lab/TAGATG-293T/bigwig"; BT = "/nas/longleaf/rhel9/apps/bedtools/2.31.1/bedtools2/bin/bedtools"; FIMO = "/nas/longleaf/rhel9/apps/meme/5.5.7/bin/fimo"; FA = f"{X}/downstream/cache/hg38.fa"
ATAC = f"{RG}/m2_abc/inputs/omniatac_merged.tn5.tagAlign.gz"; K27 = f"{RG}/m2b_abc_k27ac/inputs/h3k27ac_hek293t_srx18002015_16.tagAlign.gz"; QMIN = -np.log10(0.05); rng = np.random.default_rng(20261112)
MAIN = [f"chr{i}" for i in list(range(1, 23)) + ["X"]]; CI = {c: i for i, c in enumerate(MAIN)}
sizes = pd.read_csv(f"{RG}/m2_abc/inputs/chrom_sizes.tsv", sep="\t", header=None, index_col=0)[1]; sizes[MAIN].to_csv(f"{W}/genome.txt", sep="\t", header=False)
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]
FL = pd.read_csv(f"{RG}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf")
U = pd.read_csv(f"{X}/data/HEK293T_ATAC_union3.bed.gz", sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"]); U = U[U.c.isin(MAIN)].sort_values(["c", "s"]).reset_index(drop=True)
GL = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t", usecols=["chr", "tss"]).dropna(); TSS = {c: np.sort(g.tss.values.astype(int)) for c, g in GL.groupby("chr")}
def merged(c, s, e):
    D = {}
    for ch in np.unique(c):
        m = c == ch; o = np.argsort(s[m]); ss, ee = s[m][o], e[m][o]; ems = np.maximum.accumulate(ee); D[ch] = (ss, ems)
    return D
def inside(D, c, p):
    out = np.zeros(len(p), bool)
    for ch in np.unique(c):
        if ch not in D: continue
        ss, ems = D[ch]; m = c == ch; k = np.searchsorted(ss, p[m], side="right") - 1; ok = k >= 0; ok[ok] = p[m][ok] < ems[k[ok]]; out[m] = ok
    return out
def near_tss(c, p):
    out = np.zeros(len(p), bool)
    for ch in np.unique(c):
        t = TSS.get(ch); m = c == ch
        if t is None: continue
        k = np.searchsorted(t, p[m]); d = np.minimum(np.abs(p[m] - t[np.clip(k - 1, 0, len(t) - 1)]), np.abs(p[m] - t[np.clip(k, 0, len(t) - 1)])); out[m] = d <= 1000
    return out
def win_count(pc, pp, c, a, b):  # counts of points (pc, pp) in windows [a, b)
    out = np.zeros(len(a))
    for ch in np.unique(c):
        m = c == ch; q = np.sort(pp[pc == ch]); out[m] = np.searchsorted(q, b[m]) - np.searchsorted(q, a[m])
    return out
UD = merged(U.c.values, U.s.values, U.e.values)
ROWS = []; PK = []
for t in tfs79:
    F = pd.read_csv(f"{V}/fragment/TF.{t}.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 5], names=["c", "s", "e", "st"]); c5 = F.c.values.astype(str); p5 = np.where(F.st.values == "+", F.s.values, F.e.values - 1); n5 = len(p5)
    A = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 2, 6, 8, 9], names=["c", "s", "e", "sig", "q", "off"]); A = A[A.c.isin(MAIN)]; A["summit"] = A.s + A.off
    Q = A[A.q >= QMIN].reset_index(drop=True); r = dict(label=LAB, tf=t, cells=FL.cells.get(t, np.nan), fragments=n5)
    for nm, P in (("q005", Q), ("ascalled_p01", A)):
        cP = P.c.values.astype(str); sm = P.summit.values; r[f"n_peaks_{nm}"] = len(P)
        r[f"frac_summits_in_atac_{nm}"] = float(inside(UD, cP, sm).mean()) if len(P) else np.nan; r[f"frac_summits_promoter_{nm}"] = float(near_tss(cP, sm).mean()) if len(P) else np.nan
    QD = merged(Q.c.values.astype(str), Q.s.values, Q.e.values); r["FRiP_q005"] = float(inside(QD, c5, p5).mean()) if len(Q) else 0.0; r["frac_reads_in_atac"] = float(inside(UD, c5, p5).mean())
    cQ = Q.c.values.astype(str); ins = inside(UD, cQ, Q.summit.values); Qi = Q[ins].reset_index(drop=True)
    Qi["tf_cpm"] = win_count(c5, p5, Qi.c.values.astype(str), Qi.summit.values - 100, Qi.summit.values + 101) / n5 * 1e6; Qi["tf"] = t; PK.append(Qi[["tf", "c", "summit", "sig", "tf_cpm"]]); ROWS.append(r)
    print(t, len(Q), len(Qi), flush=True)
PK = pd.concat(PK, ignore_index=True); PK["k"] = PK.c.map(CI); PK["ws"] = PK.summit - 100; PK["we"] = PK.summit + 101
TABIX = "/nas/longleaf/rhel9/apps/samtools/1.25/htslib/bin/tabix"
def tag_counts(df, tag, name):  # bedtools -sorted needs the tag file's own chromosome order: take it from the tabix index
    order = subprocess.run([TABIX, "-l", tag], capture_output=True, text=True, check=True).stdout.split(); go = {c: i for i, c in enumerate(order)}
    sizes.reindex(order).dropna().astype(int).to_csv(f"{W}/genome_{name}.txt", sep="\t", header=False)
    o = df[df.c.isin(go)].assign(kk=lambda d: d.c.map(go)).sort_values(["kk", "ws"]); o[["c", "ws", "we"]].to_csv(f"{W}/{name}.bed", sep="\t", header=False, index=False)
    out = subprocess.run(f"set -o pipefail; {BT} intersect -a {W}/{name}.bed -b {tag} -c -sorted -g {W}/genome_{name}.txt", shell=True, capture_output=True, text=True, executable="/bin/bash")
    assert out.returncode == 0, out.stderr[:500]; v = np.array([int(l.rsplit("\t", 1)[1]) for l in out.stdout.strip().splitlines()]); res = pd.Series(v, index=o.index).reindex(df.index).fillna(0).values; return res
def tag_total(tag): return int(subprocess.run(f"set -o pipefail; zcat {tag} | wc -l", shell=True, capture_output=True, text=True, executable="/bin/bash", check=True).stdout)
TA, TK = tag_total(ATAC), tag_total(K27); PK["atac_cpm"] = tag_counts(PK, ATAC, "pk_windows") / TA * 1e6
R = pd.DataFrame(ROWS).set_index("tf")
for t, g in PK.groupby("tf"):
    if len(g) >= 20:
        R.loc[t, "spearman_sig_vs_atac"] = float(np.corrcoef(rankdata(g.sig), rankdata(g.atac_cpm))[0, 1]); R.loc[t, "spearman_tfcpm_vs_atac"] = float(np.corrcoef(rankdata(g.tf_cpm), rankdata(g.atac_cpm))[0, 1]); R.loc[t, "n_peaks_in_atac_q005"] = len(g)
# reference: H3K27ac vs ATAC over union peak centres
Uc = U.copy(); Uc["k"] = Uc.c.map(CI); mid = (Uc.s + Uc.e) // 2; Uc["ws"] = mid - 100; Uc["we"] = mid + 101
ua = tag_counts(Uc, ATAC, "union_atac"); uk = tag_counts(Uc, K27, "union_k27")
Uf = Uc.copy(); Uf["ws"] = Uc.s; Uf["we"] = Uc.e; k_in = tag_counts(Uf, K27, "union_k27_full").sum(); a_in = tag_counts(Uf, ATAC, "union_atac_full").sum()
REF = dict(label=LAB, H3K27ac_frac_reads_in_union_atac=k_in / TK, ATAC_frac_reads_in_union_atac=a_in / TA, spearman_H3K27ac_vs_ATAC_union_centres=float(np.corrcoef(rankdata(uk), rankdata(ua))[0, 1]), n_union_peaks=len(Uc))
json.dump(REF, open(f"{OUT}/partA_reference.json", "w"), indent=1); R.reset_index().to_csv(f"{OUT}/partA_per_tf.tsv", sep="\t", index=False, float_format="%.4g"); PK.to_csv(f"{W}/peaks_in_atac.tsv.gz", sep="\t", index=False)
print(REF, flush=True)
# ---------------- Part B
MOT = sorted(f[6:-5] for f in os.listdir(f"{RG}/m10_promoter_regulome/work/g") if f.startswith("motif_") and f.endswith(".meme")); SWAP = {t: MOT[(i + 1) % len(MOT)] for i, t in enumerate(MOT)}
CHIP = {"CTCF": ["CTCF_HEK293T_GSE103651_peaks.bed", "CTCF_HEK293_peaks.bed"], "NFYA": ["NFYA_K562_peaks.bed"], "MAZ": ["MAZ_HEK293_peaks.bed"], "SETDB1": ["SETDB1_HEK293_peaks.bed"], "ZBTB7A": ["ZBTB7A_HEK293_peaks.bed"], "ZIC2": ["ZIC2_HEK293_peaks.bed"], "ZNF282": ["ZNF282_HEK293_peaks.bed"], "ZNF777": ["ZNF777_HEK293_peaks.bed"]}
# GC background pool: 20,000 union peak centres +/- 50
pool = U.sample(20000, random_state=1).copy(); pm = (pool.s + pool.e) // 2; pool["ws"] = pm - 50; pool["we"] = pm + 51; pool["name"] = [f"bg{i}" for i in range(len(pool))]
def fasta(df, path):
    df[["c", "ws", "we", "name"]].to_csv(f"{path}.bed", sep="\t", header=False, index=False)
    r = subprocess.run(f"{BT} getfasta -fi {FA} -bed {path}.bed -nameOnly -fo {path}", shell=True, capture_output=True, text=True); assert r.returncode == 0, r.stderr[:300]
    seqs = {}; name = None
    for l in open(path):
        l = l.strip()
        if l.startswith(">"): name = l[1:]; seqs[name] = []
        else: seqs[name].append(l.upper())
    return {k: "".join(v) for k, v in seqs.items()}
def gc(s): return (s.count("G") + s.count("C")) / max(len(s), 1)
BGS = fasta(pool, f"{W}/bg.fa"); pool["gc"] = pool.name.map(lambda n: gc(BGS[n])); assert ":" not in "".join(BGS.keys())
def fimo_hits(motif_tf, fa):
    out = subprocess.run(f"set -o pipefail; {FIMO} --thresh 1e-4 --text --skip-matched-sequence {RG}/m10_promoter_regulome/work/g/motif_{motif_tf}.meme {fa} 2> {fa}.{motif_tf}.err | awk -F'\t' 'NR>1{{print $3}}' | sort -u", shell=True, capture_output=True, text=True, executable="/bin/bash")
    assert out.returncode == 0, open(f"{fa}.{motif_tf}.err").read()[:500]; return set(out.stdout.split())
BG_HITS = {}
ROWB = []; CHECK = []
for t in sorted(set(PK.tf)):
    g = PK[PK.tf == t].reset_index(drop=True)
    if len(g) < 200: continue
    lt, la = np.log(g.tf_cpm + .5), np.log(g.atac_cpm + .5); b = np.polyfit(la, lt, 1); g["r1_signal"] = g.sig; g["r2_ratio"] = (g.tf_cpm + .5) / (g.atac_cpm + .5); g["r3_residual"] = lt - np.polyval(b, la); g["r4_atac"] = g.atac_cpm
    tops = {}
    for rk in ("r1_signal", "r2_ratio", "r3_residual", "r4_atac"):
        o = g.index[np.lexsort((rng.random(len(g)), -g[rk].values))]
        for N in (500, 1000, 2000): tops[(rk, N)] = o[:min(N, len(g))]
    need = sorted(set(np.concatenate(list(tops.values())))); S = g.loc[need].copy(); S["ws"] = S.summit - 50; S["we"] = S.summit + 51; S["name"] = [f"p{i}" for i in S.index]
    has_motif = t in MOT; chips = CHIP.get(t, [])
    if has_motif:
        SQ = fasta(S, f"{W}/seq_{t}.fa"); assert ":" not in "".join(SQ.keys()); S["gc"] = S.name.map(lambda n: gc(SQ[n]))
        CHECK.append(dict(tf=t, n_seqs=len(SQ), chromosomes=",".join(sorted(set(S.c))), missing_main=",".join(sorted(set(MAIN) - set(S.c))) or "none"))
        own = fimo_hits(t, f"{W}/seq_{t}.fa"); swp = fimo_hits(SWAP[t], f"{W}/seq_{t}.fa")
        for m_ in (t, SWAP[t]):
            if m_ not in BG_HITS: BG_HITS[m_] = fimo_hits(m_, f"{W}/bg.fa")
    CD = {}
    for f in chips:
        C = pd.read_csv(f"{X}/data/{f}", sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"]); C = C[C.c.isin(MAIN)]; CD[f] = merged(C.c.values, C.s.values, C.e.values)
    for (rk, N), ix in tops.items():
        sel = g.loc[ix]; row = dict(label=LAB, tf=t, ranking=rk, top_n=N, n=len(ix))
        if has_motif:
            nm = [f"p{i}" for i in ix]; row["own_motif_frac"] = np.mean([n in own for n in nm]); row["swapped_motif"] = SWAP[t]; row["swapped_motif_frac"] = np.mean([n in swp for n in nm])
            gq = np.digitize(S.set_index("name").loc[nm, "gc"].values, np.quantile(pool.gc, np.linspace(.1, .9, 9))); pq_ = np.digitize(pool.gc.values, np.quantile(pool.gc, np.linspace(.1, .9, 9)))
            bgn = np.concatenate([rng.choice(np.flatnonzero(pq_ == d), (gq == d).sum(), replace=True) for d in np.unique(gq)]); bnames = pool.name.values[bgn]
            row["own_motif_bg_frac"] = np.mean([n in BG_HITS[t] for n in bnames]); row["swapped_motif_bg_frac"] = np.mean([n in BG_HITS[SWAP[t]] for n in bnames])
        for f, D in CD.items(): row[f"chip_frac::{f}"] = float(inside(D, sel.c.values.astype(str), sel.summit.values).mean())
        ROWB.append(row)
    print("B", t, len(g), flush=True)
B = pd.DataFrame(ROWB); B.to_csv(f"{OUT}/partB_rankings.tsv", sep="\t", index=False, float_format="%.4g"); pd.DataFrame(CHECK).to_csv(f"{OUT}/partB_fimo_checks.tsv", sep="\t", index=False)
def boot(v): v = np.asarray(v, float); v = v[np.isfinite(v)]; b = v[rng.integers(0, len(v), (2000, len(v)))].mean(1); return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(v)
V_ = []
for N in (500, 1000, 2000):
    b1 = B[(B.top_n == N)].pivot_table(index="tf", columns="ranking", values=["own_motif_frac", "swapped_motif_frac"])
    for rk in ("r2_ratio", "r3_residual", "r4_atac"):
        d_own = b1["own_motif_frac"][rk] - b1["own_motif_frac"]["r1_signal"]; d_sw = b1["swapped_motif_frac"][rk] - b1["swapped_motif_frac"]["r1_signal"]
        m, lo, hi, n = boot(d_own); m2, lo2, hi2, _ = boot(d_own - d_sw)
        V_.append(dict(label=LAB, top_n=N, ranking_minus_signal=rk, n_tfs=n, mean_gain_own_motif=m, lo=lo, hi=hi, mean_gain_own_minus_swapped=m2, lo2=lo2, hi2=hi2, verdict_improves=bool(lo > 0 and lo2 > 0)))
    cc = [c for c in B.columns if c.startswith("chip_frac::")]
    for c in cc:
        p = B[(B.top_n == N)].pivot_table(index="tf", columns="ranking", values=c)
        for rk in ("r2_ratio", "r3_residual", "r4_atac"):
            if "r1_signal" in p and rk in p: V_.append(dict(label=LAB, top_n=N, ranking_minus_signal=rk, chip=c.split("::")[1], n_tfs=int(p[rk].notna().sum()), chip_gain=float((p[rk] - p["r1_signal"]).mean())))
VV = pd.DataFrame(V_); VV.to_csv(f"{OUT}/partB_verdict.tsv", sep="\t", index=False, float_format="%.4g"); print(VV.drop(columns="label").round(4).to_string(index=False))
PC = B[(B.top_n == 1000) & (B.ranking == "r1_signal") & B.tf.isin(["CTCF", "NFYA", "NRF1"])][["tf", "own_motif_frac", "own_motif_bg_frac"]]; PC.to_csv(f"{OUT}/partB_positive_controls.tsv", sep="\t", index=False); print(PC.to_string(index=False))
print("done")
