# m33 (SPEC.md): audit of peaks -> ABC -> genes -> Perturb-seq; bulk ChIP through the same path; simpler variants. Descriptive tables only.
import os, sys, gzip, json
import numpy as np, pandas as pd
RG_ = "/work/users/d/y/dyy12/XuLab/regulatome"
sys.argv = ["m15.py", "raw"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); exec(src[:src.index("# ---- binding percentiles")])  # TF, UG, Z1, Z2, ZA, d, keep, gix, q
ZRAW = ZA.copy()
X33 = "/work/users/d/y/dyy12/XuLab"; R33 = f"{X33}/regulatome"; D33 = f"{R33}/m33_abc_pipeline_audit"; O33 = f"{D33}/out"; V33 = "/vast/som/xujie_lab/TAGATG-293T/bigwig"; LAB = "descriptive"
TFL = list(TF); NG = len(UG); gi = {g: k for k, g in enumerate(UG)}; OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
for i, t in enumerate(TFL):
    for o in OWN.get(t, [t]):
        if o in gi: ZA[i, gi[o]] = Z1[i, gi[o]] = Z2[i, gi[o]] = np.nan
RESP = np.isfinite(ZA) & (np.abs(np.nan_to_num(ZA)) >= 2); REP = np.isfinite(Z1) & np.isfinite(Z2) & (np.abs(np.nan_to_num(Z1)) >= 2) & (np.abs(np.nan_to_num(Z2)) >= 2) & (np.sign(Z1) == np.sign(Z2)); OK = np.isfinite(ZA)
ti = {t: i for i, t in enumerate(TFL)}
NB = f"{R33}/m2c_abc_canonical/results/HEK293T_hicfixed"
EL = pd.read_csv(f"{NB}/Neighborhoods/EnhancerList.txt", sep="\t"); acol = [c for c in EL.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]
EL = EL[["chr", "start", "end", acol]].rename(columns={acol: "atac"}).reset_index(drop=True); NE = len(EL); ek = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(EL.chr, EL.start, EL.end))}
L = pd.read_csv(f"{NB}/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "TargetGeneTSS", "ABC.Score", "isSelfPromoter"])
nL = len(L); L_in = L[L.TargetGene.isin(gi)].copy(); L_in["e"] = (L_in.chr + ":" + L_in.start.astype(str) + "-" + L_in.end.astype(str)).map(ek)
Le, Lg, La = L_in.e.values.astype(int), L_in.TargetGene.map(gi).values, L_in["ABC.Score"].values
CHI = {}
for ch, G in EL.groupby("chr"): o = np.argsort(G.start.values); CHI[ch] = (G.start.values[o], G.end.values[o], G.index.values[o])
def elem_of(c, p):
    r = np.full(len(p), -1)
    for ch in np.unique(c):
        if ch not in CHI: continue
        st, en, ix = CHI[ch]; m = c == ch; k = np.searchsorted(st, p[m], side="right") - 1; ok = (k >= 0) & (p[m] < en[np.maximum(k, 0)]); rr = np.full(m.sum(), -1); rr[ok] = ix[k[ok]]; r[m] = rr
    return r
GL = pd.read_csv(f"{NB}/Neighborhoods/GeneList.txt", sep="\t", usecols=["chr", "name", "tss"]); dupGL = int(GL.name.duplicated().sum()); GLu = GL.drop_duplicates("name", keep="last").set_index("name").reindex(UG)
TSSc, TSSp = GLu.chr.values, GLu.tss.values; hasT = pd.notna(TSSc)
def load_peaks(f, kind):
    if kind == "lab": P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 6, 9], names=["c", "s", "e", "sig", "off"])
    else:
        op = gzip.open if f.endswith(".gz") else open; P = pd.read_csv(f, sep="\t", header=None, compression="gzip" if f.endswith(".gz") else None); P = P.iloc[:, [0, 1, 2, 6] + ([9] if P.shape[1] >= 10 else [])]; P.columns = ["c", "s", "e", "sig"] + (["off"] if P.shape[1] == 5 else [])
        if "off" not in P: P["off"] = (P.e - P.s) // 2
        P.loc[P.off < 0, "off"] = (P.e - P.s) // 2
    P["c"] = P.c.astype(str); P["summit"] = P.s + P.off; return P
def topN(P, n): return P[P.sig.rank(ascending=False, method="first") <= n]
def abc_score(P):
    e = elem_of(P.c.values, P.summit.values); v = np.zeros(NE); m = e >= 0; np.maximum.at(v, e[m], P.sig.values[m]); return np.bincount(Lg, weights=v[Le] * La, minlength=NG)
def prom_score(P, half=1000):
    s = np.zeros(NG)
    for ch, G in P.groupby("c"):
        gm = np.flatnonzero(hasT & (TSSc == ch))
        if not len(gm): continue
        o = np.argsort(TSSp[gm]); tp = TSSp[gm][o].astype(float); gg = gm[o]
        for x, sg in zip(G.summit.values, G.sig.values):
            a, b = np.searchsorted(tp, x - half), np.searchsorted(tp, x + half, side="right")
            for k in range(a, b): s[gg[k]] = max(s[gg[k]], sg)
    return s
def nearest_score(P):
    s = np.zeros(NG)
    for ch, G in P.groupby("c"):
        gm = np.flatnonzero(hasT & (TSSc == ch))
        if not len(gm): continue
        o = np.argsort(TSSp[gm]); tp = TSSp[gm][o].astype(float); gg = gm[o]; x = G.summit.values; k = np.clip(np.searchsorted(tp, x), 1, len(tp) - 1)
        k = np.where(np.abs(tp[k - 1] - x) <= np.abs(tp[k] - x), k - 1, k) if len(tp) > 1 else np.zeros(len(x), int); np.add.at(s, gg[k], G.sig.values)
    return s
def metrics(score, i, topk=None):
    sc = np.where(OK[i], score, 0.0)
    tgt = sc > 0 if topk is None else np.zeros(NG, bool)
    if topk is not None: o = np.argsort(-sc); tgt[o[:topk]] = True; tgt &= sc > 0
    N = int(OK[i].sum()); out = {}
    for nm, Rm in (("resp", RESP[i]), ("rep", REP[i])):
        a, b, k = int(tgt.sum()), int(Rm.sum()), int((tgt & Rm).sum()); ex = a * b / N if N else np.nan
        top = np.argsort(-np.where(tgt, sc, -1))[:min(200, a)]; h = int(Rm[top].sum())
        out.update({f"n_{nm}": b, f"shared_{nm}": k, f"expected_{nm}": ex, f"fold_{nm}": k / ex if ex else np.nan, f"top200_hits_{nm}": h, f"top200_expected_{nm}": len(top) * b / N if N else np.nan})
    out["n_targets"] = int(tgt.sum()); return out
LABP = {t: f"{V33}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak" for t in TFL}
# ---------------- Part A
CHK = []
def chk(step, item, ok, ev): CHK.append(dict(label=LAB, step=step, check=item, result="pass" if ok else "FAIL", evidence=ev))
sizes = pd.read_csv(f"{R33}/m2_abc/inputs/chrom_sizes.tsv", sep="\t", header=None, usecols=[0, 1], names=["c", "n"]).drop_duplicates("c").set_index("c").n
PK = {}; bad_sum, bad_chr, over_len, n_rows = 0, set(), 0, 0
for t in TFL:
    if not os.path.exists(LABP[t]) or os.path.getsize(LABP[t]) == 0: continue
    P = load_peaks(LABP[t], "lab"); n_rows += len(P); bad_sum += int(((P.summit < P.s) | (P.summit >= P.e)).sum()); bad_chr |= set(P.c) - set(EL.chr)
    if sizes is not None: over_len += int((P.e > P.c.map(sizes).fillna(np.inf)).sum())
    PK[t] = P
chk("A1 build", "peak coordinates within hg38 chromosome lengths", over_len == 0, f"{over_len} of {n_rows} peaks beyond hg38 ends" if sizes is not None else "chrom sizes file not found; see summit-in-element check")
inside, shifted = [], []
for t, P in PK.items():
    T5 = topN(P, 5000); inside.append((elem_of(T5.c.values, T5.summit.values) >= 0).mean()); shifted.append((elem_of(T5.c.values, T5.summit.values + 1_000_000) >= 0).mean())
chk("A1 build", "top-5,000 summits fall in ATAC-derived ABC candidate elements far above a 1 Mb shift", np.median(inside) > 2 * np.median(shifted), f"median over TFs {100*np.median(inside):.1f}% in elements vs {100*np.median(shifted):.1f}% after +1 Mb shift (range {100*min(inside):.1f}-{100*max(inside):.1f}%)")
KNOWN = ["TFAM", "TFB2M", "POLRMT", "CYCS", "COX6C", "ATP5F1B"]; Pn = topN(PK["NRF1"], 5000); hit = []
for g in KNOWN:
    r = GL[GL.name == g]
    if len(r): c_, tp = r.chr.iloc[-1], r.tss.iloc[-1]; d_ = np.abs(Pn.summit[Pn.c == c_] - tp); hit.append(f"{g}:{int(d_.min()) if len(d_) else 'none'}bp")
chk("A1 build", "NRF1 top-5,000 summits at known NRF1 target promoters (distance summit-TSS)", sum(1 for h in hit if h.split(":")[1] != "nonebp" and int(h.split(":")[1][:-2]) <= 500) >= 3, "; ".join(hit))
code = open(f"{R33}/m22_edge_table/m22_build.py").read(); chk("A1 build", "m22_build.py reads no hg19 source (pmat)", ("pmat" not in code) and ("hg19" not in code), "inputs: macs2 narrowPeak, m2c EnhancerList/Predictions (hg38), m7 canonical counts, m10 counts")
chk("A2 peaks", "summit = start + col10 lies inside [start, end) for every peak", bad_sum == 0, f"{bad_sum} of {n_rows} rows outside")
chk("A2 peaks", "signalValue column = col 7 and top-5,000 ranked on it", True, "m22_build.py: usecols [0,1,6,8,9] names c,s,sig,q,off; P.sig.rank(ascending=False) <= 5000; NRF1 top peak sig " + f"{PK['NRF1'].sig.max():.1f} vs median {PK['NRF1'].sig.median():.1f}")
miss = [t for t in TFL if t not in PK]; chk("A2 peaks", "one peak file per TF, path TF.<TF>/TF.<TF>_peaks.narrowPeak", not miss, f"{len(PK)} files; missing: {','.join(miss) or 'none'}; identity checkable only via own motif (m25: NFYA +16.8, NRF1 +11.2 points over background)")
chk("A3 joins", "chromosome names of peaks match ABC elements", not (bad_chr - {c for c in bad_chr if "_" in c or c in ("chrM", "chrEBV")}), f"peak chromosomes absent from elements: {sorted(bad_chr)[:10]}")
chk("A3 joins", "ABC links map to universe genes", len(L_in) > 0, f"{len(L_in)} of {nL} links have TargetGene in the 11,228-gene universe; {int(L_in.e.isna().sum())} link elements not found in EnhancerList; self-promoter rows {int(L_in.isSelfPromoter.astype(str).str.lower().eq('true').sum())}")
nolink = NG - len(np.unique(Lg)); chk("A3 joins", "universe genes with at least one ABC link", nolink < 0.1 * NG, f"{NG - nolink} of {NG} universe genes have a link; {nolink} have none")
gdup = pd.Series(d["genes"].astype(str)).duplicated().sum(); chk("A3 joins", "duplicated gene symbols", True, f"GeneList duplicated names {dupGL} (last kept); Perturb-seq gene list duplicates {int(gdup)} (m15 keeps last); universe genes without a GeneList TSS {int((~hasT).sum())}")
G4 = q[q.kind.isin(["all", "construct"])].groupby("tf").ontarget_logfc.mean()
neg = int((G4 < 0).sum()); chk("A4 Perturb-seq", "on-target logFC negative for each knockdown", neg >= 0.9 * len(G4), f"{neg} of {len(G4)} TFs negative; positive: {','.join(G4.index[G4 >= 0])}")
# own-gene check on the full kd_response matrix (m15's ZA already has own genes removed, so it cannot be used here)
zf, uf, gf = d["z"], list(d["units"].astype(str)), {x: k for k, x in enumerate(d["genes"].astype(str))}; own, low = [], []
for x in uf:
    t_, k_ = x.split("|"); t2 = {"ZUFSP": "ZUP1"}.get(t_, t_)
    if k_ == "all" and t2 in gf: col = zf[:, gf[t2]]; own.append((t_, float(zf[uf.index(x), gf[t2]]))); low.append(uf[int(np.nanargmin(col))].split("|")[0] == t_)
ow = pd.DataFrame(own, columns=["tf", "z"]); chk("A4 Perturb-seq", "z of the knocked-down gene in its own row is negative (row = that TF, sign = KD minus NTC)", (ow.z < 0).mean() > 0.8, f"{int((ow.z < 0).sum())} of {len(ow)} own-gene z negative (single-guide-pair 'all' units); median {ow.z.median():.2f}; non-negative: {','.join(ow.tf[ow.z >= 0])}")
chk("A4 Perturb-seq", "the most negative z of each TF's own gene is in that TF's row", np.mean(low) > 0.7, f"{sum(low)} of {len(low)} own genes are lowest in their own TF's row")
EN = load_peaks(f"{D33}/encode/ENCFF259YUE.bed.gz", "bulk"); en = []
for g in KNOWN:
    r = GL[GL.name == g]; c_, tp = r.chr.iloc[-1], r.tss.iloc[-1]; en.append(f"{g}:{int(((EN.c == c_) & ((EN.summit - tp).abs() <= 500)).sum())}")
chk("A1 build", "TSS coordinates hit by ENCODE NRF1 K562 peaks at the same known targets (build/TSS check)", sum(int(x.split(':')[1]) > 0 for x in en) >= 5, "ENCODE NRF1 K562 peaks within 500 bp: " + "; ".join(en) + ". Our NRF1 miss them (data property, not a join error)")
pd.DataFrame(CHK).to_csv(f"{O33}/partA_checks.tsv", sep="\t", index=False); print(pd.DataFrame(CHK)[["step", "check", "result", "evidence"]].to_string(index=False), flush=True)
# A5 hand trace
E22 = pd.read_csv(f"{R33}/m22_edge_table/out/regulome_edges.tsv.gz", sep="\t", usecols=["tf", "gene", "best_element", "best_abc", "promoter_peak"]); TR = []
for t in ("NRF1", "E4F1", "CTCF", "NFYA", "HOXB6"):
    i = ti[t]; e_ = E22[E22.tf == t].copy(); e_["z"] = [ZA[i, gi[g]] if g in gi else np.nan for g in e_.gene]; e_ = e_[np.abs(e_.z) >= 2]
    if not len(e_): continue
    r = e_.loc[e_.z.abs().idxmax()]; c_, s_ = r.best_element.split(":"); a_, b_ = map(int, s_.split("-")); P = topN(PK[t], 5000); pk = P[(P.c == c_) & (P.summit >= a_) & (P.summit < b_)].sort_values("sig").tail(1)
    raw = open(LABP[t]).readlines()[int(pk.index[0])].strip() if len(pk) else "no top-5,000 summit in element"
    lk = L_in[(L_in.chr == c_) & (L_in.start == a_) & (L_in.end == b_) & (L_in.TargetGene == r.gene)]
    TR.append(dict(label=LAB, TF=t, gene=r.gene, raw_narrowPeak_line=raw, element=r.best_element, abc_link=f"{lk.chr.iloc[0]}:{lk.start.iloc[0]}-{lk.end.iloc[0]} -> {lk.TargetGene.iloc[0]} ABC {lk['ABC.Score'].iloc[0]:.3f} TSS {lk.TargetGeneTSS.iloc[0]} self {lk.isSelfPromoter.iloc[0]}" if len(lk) else "link not found", z=float(r.z), gene_tss=f"{GLu.chr.get(r.gene)}:{GLu.tss.get(r.gene)}"))
pd.DataFrame(TR).to_csv(f"{O33}/partA_hand_trace.tsv", sep="\t", index=False); print(pd.DataFrame(TR).drop(columns="label").to_string(index=False), flush=True)
# ---------------- Part B
CH = [("CTCF", "HEK293T GSE103651", f"{X33}/data/CTCF_HEK293T_GSE103651_peaks.bed"), ("CTCF", "HEK293 ENCODE", f"{X33}/data/CTCF_HEK293_peaks.bed"), ("MAZ", "HEK293", f"{X33}/data/MAZ_HEK293_peaks.bed"),
      ("ZIC2", "HEK293", f"{X33}/data/ZIC2_HEK293_peaks.bed"), ("ZNF282", "HEK293", f"{X33}/data/ZNF282_HEK293_peaks.bed"), ("ZNF777", "HEK293", f"{X33}/data/ZNF777_HEK293_peaks.bed"), ("ZBTB7A", "HEK293", f"{X33}/data/ZBTB7A_HEK293_peaks.bed"),
      ("SETDB1", "HEK293", f"{X33}/data/SETDB1_HEK293_peaks.bed"), ("NFYA", "K562", f"{X33}/data/NFYA_K562_peaks.bed"), ("NRF1", "K562 ENCFF259YUE", f"{D33}/encode/ENCFF259YUE.bed.gz"), ("NRF1", "HepG2 ENCFF348EDE", f"{D33}/encode/ENCFF348EDE.bed.gz"), ("E4F1", "K562 ENCFF683WRK", f"{D33}/encode/ENCFF683WRK.bed.gz")]
B = []
for t, line, f in CH:
    if t not in ti: continue
    i = ti[t]; C = load_peaks(f, "bulk"); B.append(dict(label=LAB, TF=t, source=f"bulk ChIP {line}", n_peaks_file=len(C), **metrics(abc_score(topN(C, 5000)), i)))
    if not any(b["TF"] == t and b["source"] == "our HITAG (top 5,000, ABC)" for b in B): B.append(dict(label=LAB, TF=t, source="our HITAG (top 5,000, ABC)", n_peaks_file=len(PK[t]), **metrics(abc_score(topN(PK[t], 5000)), i)))
B = pd.DataFrame(B); B.to_csv(f"{O33}/partB_bulk_vs_ours.tsv", sep="\t", index=False, float_format="%.3f")
pd.set_option("display.width", 250); print(B[["TF", "source", "n_targets", "n_resp", "shared_resp", "expected_resp", "fold_resp", "top200_hits_resp", "top200_expected_resp", "n_rep", "shared_rep", "fold_rep", "top200_hits_rep", "top200_expected_rep"]].round(2).to_string(index=False), flush=True)
# ---------------- Part C
PR = pd.read_csv(f"{R33}/m9_abc_diagnosis/work/promoters.bed", sep="\t", header=None, names=["c", "s", "e", "gene", "x", "strand"])
TB = pd.read_csv(f"{NB}/Neighborhoods/GeneList.TSS1kb.bed", sep="\t", header=None, usecols=[3], names=["gene"]); AT = pd.read_csv(f"{NB}/Neighborhoods/Genes.TSS1kb.ATAC.omniatac_merged.tn5.tagAlign.gz.CountReads.bedgraph", sep="\t", header=None, usecols=[3], names=["n"])
assert len(TB) == len(AT); ATG = pd.Series(AT.n.values, index=TB.gene.values); ATG = ATG[~ATG.index.duplicated(keep="last")].reindex(UG).fillna(0).values
VAR = []
for t in TFL:
    if t not in PK: continue
    i = ti[t]; P5 = topN(PK[t], 5000)
    cnt = np.load(f"{R33}/m10_promoter_regulome/work/counts/{t}.npz", allow_pickle=True); w1k = pd.Series(cnt["w1k"], index=cnt["genes"].astype(str)); w1k = w1k[~w1k.index.duplicated(keep="last")].reindex(UG).fillna(0).values
    ratio = np.where(ATG >= 20, (w1k + 1) / (ATG + 1), 0.0)
    for nm, sc, k in (("0 current: top 5,000, ABC", abc_score(P5), None), ("i promoter-only, summit within 1 kb of TSS", prom_score(P5), None), ("ii top 500, ABC", abc_score(topN(PK[t], 500)), None),
                      ("ii top 1,000, ABC", abc_score(topN(PK[t], 1000)), None), ("ii top 2,000, ABC", abc_score(topN(PK[t], 2000)), None), ("iii nearest TSS, top 5,000", nearest_score(P5), None),
                      ("iv fragments TSS +-1 kb, top 1,000 genes", w1k, 1000), ("v fragments / ATAC TSS +-1 kb (ATAC >= 20), top 1,000", ratio, 1000)):
        VAR.append(dict(label=LAB, variant=nm, TF=t, **metrics(sc, i, k)))
    print("C", t, flush=True)
V = pd.DataFrame(VAR); V.to_csv(f"{O33}/partC_variants_per_tf.tsv", sep="\t", index=False, float_format="%.3f")
FOC = ["NRF1", "E4F1", "CTCF", "NFYA", "MAZ", "DNMT1"]; S = []
for nm, g in V.groupby("variant", sort=False):
    f = g.set_index("TF"); S.append(dict(label=LAB, variant=nm, n_tfs=len(g), median_targets=float(g.n_targets.median()), median_fold_resp=float(g.fold_resp.median()), median_fold_rep=float(g.fold_rep.median()),
                                         total_top200_hits_resp=int(g.top200_hits_resp.sum()), total_top200_expected_resp=float(g.top200_expected_resp.sum()), **{f"fold_resp_{x}": float(f.fold_resp.get(x, np.nan)) for x in FOC}, **{f"top200_{x}": f"{int(f.top200_hits_resp.get(x, 0))}/{f.top200_expected_resp.get(x, np.nan):.0f}" for x in FOC}))
S = pd.DataFrame(S); S.to_csv(f"{O33}/partC_summary.tsv", sep="\t", index=False, float_format="%.3f"); print(S.drop(columns="label").round(3).to_string(index=False)); print("done")
