# m25_peak_threshold (SPEC.md): six edge definitions (E0 q005, E1 as-called, E2 top N = 2k/5k/10k/20k), choice rule by own motif and bulk ChIP, Perturb-seq checks. Tables only. Diagnostic.
import json, os, subprocess, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
X_ = "/work/users/d/y/dyy12/XuLab"; RG_ = f"{X_}/regulatome"; O_ = f"{RG_}/m25_peak_threshold"; W_ = f"{O_}/work"; OUT_ = f"{O_}/out"; V_ = "/vast/som/xujie_lab/TAGATG-293T/bigwig"; LAB_ = "diagnostic"; rg = np.random.default_rng(20261112)
BT = "/nas/longleaf/rhel9/apps/bedtools/2.31.1/bedtools2/bin/bedtools"; FIMO = "/nas/longleaf/rhel9/apps/meme/5.5.7/bin/fimo"; FA = f"{X_}/downstream/cache/hg38.fa"; QMIN = -np.log10(0.05)
DEFS = ["E0_q005", "E1_as_called", "E2_top2000", "E2_top5000", "E2_top10000", "E2_top20000"]
tfs79 = [l.strip() for l in open(f"{RG_}/m1_binding/v2/tfs.txt") if l.strip()]; ti79 = {t: i for i, t in enumerate(tfs79)}
UNIV = np.load(f"{RG_}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True)["genes"].astype(str); gi = {g: k for k, g in enumerate(UNIV)}; NG = len(UNIV)
EL = pd.read_csv(f"{RG_}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end"]).reset_index(drop=True); NE = len(EL); ek = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(EL.chr, EL.start, EL.end))}
L = pd.read_csv(f"{RG_}/m2c_abc_canonical/results/HEK293T_hicfixed/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score", "isSelfPromoter"])
L = L[L.TargetGene.isin(gi)].copy(); L["e"] = (L.chr + ":" + L.start.astype(str) + "-" + L.end.astype(str)).map(ek).astype(int); L["g"] = L.TargetGene.map(gi); Le, Lg, La, Ls = L.e.values, L.g.values, L["ABC.Score"].values, L.isSelfPromoter.astype(str).str.lower().eq("true").values
linked = np.zeros(NE, bool); linked[Le] = True
CHI = {}
for ch, G in EL.groupby("chr"): o = np.argsort(G.start.values); CHI[ch] = (G.start.values[o], G.end.values[o], G.index.values[o])
def elem_of(c, p):
    r = np.full(len(p), -1)
    for ch in np.unique(c):
        if ch not in CHI: continue
        st, en, ix = CHI[ch]; m = c == ch; k = np.searchsorted(st, p[m], side="right") - 1; ok = (k >= 0) & (p[m] < en[np.maximum(k, 0)]); rr = np.full(m.sum(), -1); rr[ok] = ix[k[ok]]; r[m] = rr
    return r
# ---- per TF, per definition: strongest summit per element
SIG = {d: np.zeros((NE, 79), np.float32) for d in DEFS}; SUM = {d: np.full((NE, 79), -1, np.int64) for d in DEFS}
for t in tfs79:
    f = f"{V_}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
    if not os.path.exists(f) or os.path.getsize(f) == 0: continue
    P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 6, 8, 9], names=["c", "s", "sig", "q", "off"]); P["summit"] = P.s + P.off; P["rank"] = P.sig.rank(ascending=False, method="first"); P["e"] = elem_of(P.c.values.astype(str), P.summit.values)
    for d in DEFS:
        sel = P.q >= QMIN if d == "E0_q005" else (np.ones(len(P), bool) if d == "E1_as_called" else P["rank"] <= int(d.split("top")[1]))
        Q = P[sel & (P.e >= 0)].sort_values("sig", ascending=False).drop_duplicates("e"); SIG[d][Q.e.values, ti79[t]] = Q.sig.values; SUM[d][Q.e.values, ti79[t]] = Q.summit.values
    print(t, len(P), flush=True)
def gene_mats(d):
    S = SIG[d]; B = np.zeros((79, NG), bool); Pm = np.zeros((79, NG), bool); Sc = np.zeros((79, NG))
    for k in range(79):
        s = S[Le, k]; B[k] = np.bincount(Lg, weights=(s > 0).astype(float), minlength=NG) > 0; Pm[k] = np.bincount(Lg, weights=((s > 0) & Ls).astype(float), minlength=NG) > 0; Sc[k] = np.bincount(Lg, weights=s * La, minlength=NG)
    return B, Pm, Sc
GM = {d: gene_mats(d) for d in DEFS}
FL = pd.read_csv(f"{RG_}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf"); tert = {t: str(FL.tertile.get(t, "na")) for t in tfs79}
# ---- 1. description
D1 = []
for d in DEFS:
    B, Pm, Sc = GM[d]
    for t in tfs79:
        k = ti79[t]; D1.append(dict(label=LAB_, definition=d, tf=t, depth_tertile=tert[t], genes_with_edge=int(B[k].sum()), frac_promoter_edges=float(Pm[k].sum() / max(B[k].sum(), 1)), edge_elements=int((SIG[d][:, k] > 0)[linked].sum())))
D1 = pd.DataFrame(D1); D1.to_csv(f"{OUT_}/definitions_per_tf.tsv", sep="\t", index=False, float_format="%.4g")
JS = []
for d in DEFS:
    B = GM[d][0]; S = {t: set(np.flatnonzero(B[ti79[t]])) for t in tfs79 if B[ti79[t]].any()}; ts = list(S); vals = []
    for i, a in enumerate(ts):
        for b in ts[i + 1:]: vals.append(len(S[a] & S[b]) / len(S[a] | S[b]))
    jj = lambda a, b: len(S[a] & S[b]) / len(S[a] | S[b]) if a in S and b in S else np.nan
    JS.append(dict(label=LAB_, definition=d, n_tfs_with_edges=len(ts), median_jaccard=float(np.median(vals)), NFYA_RBBP4=jj("NFYA", "RBBP4"), RBBP4_SOX4=jj("RBBP4", "SOX4"), NRF1_E4F1=jj("NRF1", "E4F1")))
JS = pd.DataFrame(JS); JS.to_csv(f"{OUT_}/definitions_jaccard.tsv", sep="\t", index=False, float_format="%.4g"); print(JS.drop(columns="label").round(3).to_string(index=False), flush=True)
print(D1.pivot_table(index="depth_tertile", columns="definition", values="genes_with_edge", aggfunc="median").to_string(), flush=True)
# ---- 2. choice rule: motif and ChIP
MOT = sorted(f[6:-5] for f in os.listdir(f"{RG_}/m10_promoter_regulome/work/g") if f.startswith("motif_") and f.endswith(".meme")); SWAP = {t: MOT[(i + 1) % len(MOT)] for i, t in enumerate(MOT)}
U = pd.read_csv(f"{X_}/data/HEK293T_ATAC_union3.bed.gz", sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"]); pool = U.sample(20000, random_state=1).copy(); pm = (pool.s + pool.e) // 2; pool["ws"] = pm - 50; pool["we"] = pm + 51; pool["name"] = [f"bg{i}" for i in range(len(pool))]
def fasta(df, path):
    df[["c", "ws", "we", "name"]].to_csv(f"{path}.bed", sep="\t", header=False, index=False); r = subprocess.run(f"{BT} getfasta -fi {FA} -bed {path}.bed -nameOnly -fo {path}", shell=True, capture_output=True, text=True); assert r.returncode == 0, r.stderr[:300]
    seqs = {}; name = None
    for l in open(path):
        l = l.strip()
        if l.startswith(">"): name = l[1:]; seqs[name] = []
        else: seqs[name].append(l.upper())
    out = {k: "".join(v) for k, v in seqs.items()}; assert ":" not in "".join(out); return out
def gc(s): return (s.count("G") + s.count("C")) / max(len(s), 1)
def fimo_hits(m, fa):
    o = subprocess.run(f"set -o pipefail; {FIMO} --thresh 1e-4 --text --skip-matched-sequence {RG_}/m10_promoter_regulome/work/g/motif_{m}.meme {fa} 2> {fa}.{m}.err | awk -F'\t' 'NR>1{{print $3}}' | sort -u", shell=True, capture_output=True, text=True, executable="/bin/bash")
    assert o.returncode == 0, open(f"{fa}.{m}.err").read()[:300]; return set(o.stdout.split())
BGS = fasta(pool, f"{W_}/bg.fa"); pool["gc"] = pool.name.map(lambda n: gc(BGS[n])); qs = np.quantile(pool.gc, np.linspace(.1, .9, 9)); pool["dec"] = np.digitize(pool.gc, qs); BGH = {}
MOTR = []
for t in MOT:
    if t not in ti79: continue
    k = ti79[t]; rows = []
    for d in DEFS:
        e = np.flatnonzero((SIG[d][:, k] > 0) & linked); rows.append(pd.DataFrame(dict(d=d, e=e, summit=SUM[d][e, k])))
    A = pd.concat(rows); U_ = A.drop_duplicates(["e", "summit"]).reset_index(drop=True)
    if not len(U_): continue
    U_["c"] = EL.chr.values[U_.e.values]; U_["ws"] = U_.summit - 50; U_["we"] = U_.summit + 51; U_["name"] = [f"s{i}" for i in range(len(U_))]; SQ = fasta(U_, f"{W_}/seq_{t}.fa"); U_["gc"] = U_.name.map(lambda n: gc(SQ[n])); U_["dec"] = np.digitize(U_.gc, qs)
    own, swp = fimo_hits(t, f"{W_}/seq_{t}.fa"), fimo_hits(SWAP[t], f"{W_}/seq_{t}.fa")
    for m_ in (t, SWAP[t]):
        if m_ not in BGH: BGH[m_] = fimo_hits(m_, f"{W_}/bg.fa")
    key = {(a, b): n for a, b, n in zip(U_.e, U_.summit, U_.name)}; dec = dict(zip(U_.name, U_.dec))
    for d in DEFS:
        sub = A[A.d == d]
        if not len(sub): MOTR.append(dict(definition=d, tf=t, n_edge_elements=0)); continue
        names = [key[(a, b)] for a, b in zip(sub.e, sub.summit)]; ds = np.array([dec[n] for n in names])
        bgn = np.concatenate([rg.choice(np.flatnonzero(pool.dec.values == x), (ds == x).sum(), replace=True) for x in np.unique(ds)]); bn = pool.name.values[bgn]
        o, s_ = np.mean([n in own for n in names]), np.mean([n in swp for n in names]); ob, sb = np.mean([n in BGH[t] for n in bn]), np.mean([n in BGH[SWAP[t]] for n in bn])
        MOTR.append(dict(definition=d, tf=t, depth_tertile=tert[t], n_edge_elements=len(names), own_motif_frac=o, own_motif_bg=ob, own_excess=o - ob, swapped_motif=SWAP[t], swapped_frac=s_, swapped_bg=sb, swapped_excess=s_ - sb))
    print("motif", t, len(U_), flush=True)
MOTR = pd.DataFrame(MOTR); MOTR.to_csv(f"{OUT_}/choice_motif.tsv", sep="\t", index=False, float_format="%.4g")
CHIP = {"CTCF": ["CTCF_HEK293T_GSE103651_peaks.bed", "CTCF_HEK293_peaks.bed"], "NFYA": ["NFYA_K562_peaks.bed"], "MAZ": ["MAZ_HEK293_peaks.bed"], "ZBTB7A": ["ZBTB7A_HEK293_peaks.bed"], "ZIC2": ["ZIC2_HEK293_peaks.bed"], "ZNF282": ["ZNF282_HEK293_peaks.bed"], "ZNF777": ["ZNF777_HEK293_peaks.bed"]}
def chip_elems(f):
    C = pd.read_csv(f"{X_}/data/{f}", sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"]); hit = np.zeros(NE, bool)
    for ch, G in C.groupby("c"):
        if ch not in CHI: continue
        st, en, ix = CHI[ch]; emax = np.maximum.accumulate(en)
        for a, b in zip(G.s.values, G.e.values):
            k = np.searchsorted(st, b, side="left") - 1
            while k >= 0 and emax[k] > a:
                if en[k] > a: hit[ix[k]] = True
                k -= 1
    return hit
CHR = []
for t, fs in CHIP.items():
    for f in fs:
        H = chip_elems(f)
        for d in DEFS:
            bnd = (SIG[d] > 0) & linked[:, None]; fr = np.array([H[bnd[:, k]].mean() if bnd[:, k].any() else np.nan for k in range(79)])
            own = fr[ti79[t]]; oth = np.nanmedian(np.delete(fr, ti79[t])); CHR.append(dict(definition=d, tf=t, chip=f, depth_tertile=tert[t], own_frac_in_chip=own, other_tf_median=oth, chip_excess=own - oth))
CHR = pd.DataFrame(CHR); CHR.to_csv(f"{OUT_}/choice_chip.tsv", sep="\t", index=False, float_format="%.4g")
def boot(v): v = np.asarray(v, float); v = v[np.isfinite(v)]; b = v[rg.integers(0, len(v), (2000, len(v)))].mean(1) if len(v) else [np.nan]; return (float(v.mean()) if len(v) else np.nan), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(v)
VER = []
for d in DEFS[1:]:
    for grp in ("all", "low", "mid", "high"):
        mm = MOTR[(MOTR.definition == d)].set_index("tf"); m0 = MOTR[MOTR.definition == "E0_q005"].set_index("tf"); cc = CHR[CHR.definition == d].set_index(["tf", "chip"]); c0 = CHR[CHR.definition == "E0_q005"].set_index(["tf", "chip"])
        tm = [t for t in mm.index if grp == "all" or tert.get(t) == grp]; dm = (mm.own_excess - m0.own_excess).reindex(tm); ds_ = (mm.swapped_excess - m0.swapped_excess).reindex(tm)
        tc = [i for i in cc.index if grp == "all" or tert.get(i[0]) == grp]; dc = (cc.chip_excess - c0.chip_excess).reindex(tc)
        a = boot(dm); b = boot(ds_); c = boot(dc); low = D1[(D1.depth_tertile == "low")]; ne = low[low.definition == d].genes_with_edge.median() - low[low.definition == "E0_q005"].genes_with_edge.median()
        VER.append(dict(label=LAB_, definition=d, tertile=grp, motif_excess_diff_vs_E0=a[0], m_lo=a[1], m_hi=a[2], n_motif_tfs=a[3], swapped_motif_diff=b[0], chip_excess_diff_vs_E0=c[0], c_lo=c[1], c_hi=c[2], n_chip=c[3], low_tertile_median_edges_change=ne,
                        better_than_q005=bool((a[1] >= 0 if np.isfinite(a[1]) else False) and (c[1] >= 0 if np.isfinite(c[1]) else True) and ne > 0) if grp == "all" else None))
VER = pd.DataFrame(VER); VER.to_csv(f"{OUT_}/choice_verdict.tsv", sep="\t", index=False, float_format="%.4g"); print(VER.drop(columns="label").round(4).to_string(index=False), flush=True)
# ---- 3. checks (not for choosing)
sys.argv = ["m15.py", "dc"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); _o = (O_, OUT_); exec(src[:src.index("# ---- binding percentiles")]); O_, OUT_ = _o
H1d = pd.DataFrame(Z1, index=TF, columns=UG).reindex(columns=UNIV).values; H2d = pd.DataFrame(Z2, index=TF, columns=UG).reindex(columns=UNIV).values
Rz = np.load(f"{RG_}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True); KT = list(Rz["tfs"].astype(str)); Zr = Rz["r1"].astype(float)
def topk(A, K): a = np.abs(A); flat = np.where(np.isfinite(a), a, -1).ravel(); o = np.argsort(-flat)[:K]; return set(zip(*np.unravel_index(o, A.shape)))
CHK = []; COR = []
TT = [t for t in TF if t in ti79]; trow = {t: TF.index(t) for t in TT}
for d in DEFS:
    B, Pm, Sc = GM[d]
    for K in (500, 2000):
        both = [p for p in topk(H1d, K) & topk(H2d, K) if np.sign(H1d[p]) == np.sign(H2d[p]) and TF[p[0]] in ti79]
        for sub, keepf in (("all", lambda t: True), ("without E4F1, NRF1, DNMT1", lambda t: t not in ("E4F1", "NRF1", "DNMT1"))):
            P_ = [(TF[i], j) for i, j in both if keepf(TF[i])]
            if not P_: continue
            own = np.mean([B[ti79[t], j] for t, j in P_]); sw = np.mean([np.mean([B[ti79[u], j] for u in tfs79 if tert[u] == tert[t] and u != t]) for t, j in P_])
            CHK.append(dict(label=LAB_, definition=d, K=K, subset=sub, n_pairs=len(P_), own_bound=own, swapped_tertile_mean_bound=sw))
    for t in KT:
        if t not in ti79: continue
        z = np.abs(Zr[KT.index(t)]); s = Sc[ti79[t]]; m = np.isfinite(z)
        COR.append(dict(definition=d, tf=t, spearman_score_absz=float(np.corrcoef(rankdata(s[m]), rankdata(z[m]))[0, 1]) if np.std(s[m]) > 0 else np.nan, n_edges=int(B[ti79[t]].sum())))
    S_ = np.vstack([Sc[ti79[t]] for t in KT if t in ti79]); Z_ = np.abs(np.vstack([Zr[KT.index(t)] for t in KT if t in ti79])); m = np.isfinite(Z_)
    raw = float(np.corrcoef(rankdata(S_[m]), rankdata(Z_[m]))[0, 1]); pr = lambda M: np.vstack([rankdata(r) / len(r) for r in np.where(np.isfinite(M), M, 0)])
    pct = float(np.corrcoef(pr(np.where(m, S_, np.nan))[m], pr(np.where(m, Z_, np.nan))[m])[0, 1]); COR.append(dict(definition=d, tf="WHOLE MATRIX", spearman_score_absz=raw, spearman_within_tf_percentile=pct))
CHK = pd.DataFrame(CHK); CHK.to_csv(f"{OUT_}/check_highconf.tsv", sep="\t", index=False, float_format="%.4g"); COR = pd.DataFrame(COR); COR.to_csv(f"{OUT_}/check_simple_correlation.tsv", sep="\t", index=False, float_format="%.4g")
print(CHK.drop(columns="label").round(3).to_string(index=False)); print(COR[COR.tf.isin(["WHOLE MATRIX", "NRF1", "E4F1", "CTCF", "NFYA"])].round(4).to_string(index=False)); print("done")
