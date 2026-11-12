# m20 Part 3: where the signal is lost, per TF with a bulk ChIP set and a knockdown; ours vs swapped TF vs ATAC-only at every row. Diagnostic.
import json, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
X_ = "/work/users/d/y/dyy12/XuLab"; RG_ = f"{X_}/regulatome"; O_ = f"{RG_}/m20_raw_binding_audit"; OUT_ = f"{O_}/out"; V_ = "/vast/som/xujie_lab/TAGATG-293T/bigwig"; LAB_ = "diagnostic"; QMIN_ = -np.log10(0.05)
sys.argv = ["m15.py", "raw"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); exec(src[:src.index("# ---- binding percentiles")])  # TF, UG, Z1, Z2, ZA (raw per-TF z halves)
tfs79 = [l.strip() for l in open(f"{RG_}/m1_binding/v2/tfs.txt") if l.strip()]; MIS = dict(zip(tfs79, json.load(open(f"{RG_}/m7_step2/features/raw_hicfixed/feature_qc.json"))["mis_paired_map"]["mis0"]))
CHIP = [("CTCF", "HEK293T (GSE103651, ChIP-nexus)", "CTCF_HEK293T_GSE103651_peaks.bed"), ("CTCF", "HEK293 (ENCODE)", "CTCF_HEK293_peaks.bed"), ("MAZ", "HEK293", "MAZ_HEK293_peaks.bed"), ("NFYA", "K562", "NFYA_K562_peaks.bed"),
        ("SETDB1", "HEK293", "SETDB1_HEK293_peaks.bed"), ("ZBTB7A", "HEK293", "ZBTB7A_HEK293_peaks.bed"), ("ZIC2", "HEK293", "ZIC2_HEK293_peaks.bed"), ("ZNF282", "HEK293", "ZNF282_HEK293_peaks.bed"), ("ZNF777", "HEK293", "ZNF777_HEK293_peaks.bed")]
EL = pd.read_csv(f"{RG_}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/EnhancerList.txt", sep="\t"); acol = [c for c in EL.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]
EL = EL[["chr", "start", "end", acol]].rename(columns={"chr": "c", "start": "s", "end": "e", acol: "atac"}).reset_index(drop=True); NE = len(EL); ek = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(EL.c, EL.s, EL.e))}
gi = {g: k for k, g in enumerate(UG)}; NG = len(UG)
L = pd.read_csv(f"{RG_}/m2c_abc_canonical/results/HEK293T_hicfixed/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score"])
L = L[L.TargetGene.isin(gi)]; Le = (L.chr + ":" + L.start.astype(str) + "-" + L.end.astype(str)).map(ek).values.astype(int); Lg = L.TargetGene.map(gi).values; La = L["ABC.Score"].values
linked = np.zeros(NE, bool); linked[Le] = True
CHI = {}
for ch, G in EL.groupby("c"): o = np.argsort(G.s.values); CHI[ch] = (G.s.values[o], G.e.values[o], G.index.values[o])
def elem_point(c, p):
    r = np.full(len(p), -1)
    for ch in np.unique(c):
        if ch not in CHI: continue
        st, en, ix = CHI[ch]; m = c == ch; k = np.searchsorted(st, p[m], side="right") - 1; ok = (k >= 0) & (p[m] < en[np.maximum(k, 0)]); rr = np.full(m.sum(), -1); rr[ok] = ix[k[ok]]; r[m] = rr
    return r
def elem_interval(C):  # max signal of overlapping intervals per element
    v = np.zeros(NE)
    for ch, G in C.groupby("c"):
        if ch not in CHI: continue
        st, en, ix = CHI[ch]; emax = np.maximum.accumulate(en)
        for a, b, sg in zip(G.s.values, G.e.values, G.sig.values):
            k = np.searchsorted(st, b, side="left") - 1
            while k >= 0 and emax[k] > a:
                if en[k] > a: v[ix[k]] = max(v[ix[k]], sg)
                k -= 1
    return v
def our_elem(t):
    P = pd.read_csv(f"{V_}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 8, 9], names=["c", "s", "sig", "q", "off"]); P = P[P.q >= QMIN_]
    e = elem_point(P.c.values.astype(str), (P.s + P.off).values); v = np.zeros(NE); m = e >= 0; np.maximum.at(v, e[m], P.sig.values[m]); return v
def gscore(v): return np.bincount(Lg, weights=v[Le] * La, minlength=NG)
def gany(v): out = np.zeros(NG, bool); np.logical_or.at(out, Lg, v[Le] > 0); return out
def orj(a, b, m=None):
    m = np.ones(len(a), bool) if m is None else m; a, b = a[m], b[m]; n11 = (a & b).sum(); n10 = (a & ~b).sum(); n01 = (~a & b).sum(); n00 = (~a & ~b).sum()
    return float((n11 + .5) * (n00 + .5) / ((n10 + .5) * (n01 + .5))), float(n11 / max((a | b).sum(), 1)), int(a.sum()), int(n11)
def sp(a, b): m = np.isfinite(a) & np.isfinite(b); return np.nan if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0 else float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])
ti = {t: k for k, t in enumerate(TF)}; ROWS = []; SH = []
for tf, line, f in CHIP:
    if tf not in ti: print("no knockdown", tf); continue
    C = pd.read_csv(f"{X_}/data/{f}", sep="\t", header=None, usecols=[0, 1, 2, 6], names=["c", "s", "e", "sig"]); ce = elem_interval(C)
    k = ti[tf]; za = ZA[k]; ok = np.isfinite(za); top5 = np.zeros(NG, bool); idx = np.flatnonzero(ok); top5[idx[np.argsort(-np.abs(za[idx]))[:int(round(ok.sum() * .05))]]] = True
    rep = (np.abs(Z1[k]) >= 2) & (np.abs(Z2[k]) >= 2) & (np.sign(Z1[k]) == np.sign(Z2[k])) & ok
    chip_gene = gany(ce); chip_score = gscore(ce)
    for who, t in (("ours", tf), ("swapped TF " + MIS[tf], MIS[tf]), ("ATAC-only", None)):
        if t is None:
            ne = int((our_elem(tf)[linked] > 0).sum()); v = np.zeros(NE); o = np.flatnonzero(linked)[np.argsort(-EL.atac.values[linked])[:ne]]; v[o] = EL.atac.values[o]
        else: v = our_elem(t)
        r = dict(label=LAB_, tf=tf, chip=f"{tf} {line}", source=who)
        r["elem_OR"], r["elem_jaccard"], r["elem_n_called"], r["elem_n_both"] = orj(v > 0, ce > 0, linked)
        g = gany(v); r["gene_OR_vs_chip_genes"], r["gene_jaccard"], r["n_genes"], r["n_genes_both"] = orj(g, chip_gene); r["n_chip_genes"] = int(chip_gene.sum())
        r["OR_vs_top5pct_responders"], _, _, r["n_list_and_top5"] = orj(g, top5, ok); r["OR_vs_reproducible_responders"], _, _, r["n_list_and_reproducible"] = orj(g, rep, ok); r["n_reproducible"] = int(rep.sum())
        ROWS.append(r)
    r = dict(label=LAB_, tf=tf, chip=f"{tf} {line}", source="bulk ChIP gene list"); r["n_genes"] = int(chip_gene.sum())
    r["OR_vs_top5pct_responders"], _, _, r["n_list_and_top5"] = orj(chip_gene, top5, ok); r["OR_vs_reproducible_responders"], _, _, r["n_list_and_reproducible"] = orj(chip_gene, rep, ok); r["n_reproducible"] = int(rep.sum()); ROWS.append(r)
    our = gscore(our_elem(tf)); both = (our > 0) & chip_gene
    SH.append(dict(label=LAB_, tf=tf, chip=f"{tf} {line}", n_genes_both=int(both.sum()), spearman_ours_vs_chip=sp(our[both], chip_score[both]), spearman_chip_vs_absz=sp(chip_score[both & ok], np.abs(za[both & ok])),
                   spearman_ours_vs_absz=sp(our[both & ok], np.abs(za[both & ok])), spearman_chip_vs_absz_all_chip_genes=sp(chip_score[chip_gene & ok], np.abs(za[chip_gene & ok]))))
    print(tf, line, flush=True)
R = pd.DataFrame(ROWS); R.to_csv(f"{OUT_}/part3_signal_path.tsv", sep="\t", index=False, float_format="%.4f"); S = pd.DataFrame(SH); S.to_csv(f"{OUT_}/part3_within_shared_genes.tsv", sep="\t", index=False, float_format="%.4f")
print(R.drop(columns="label").round(3).to_string(index=False)); print(S.drop(columns="label").round(3).to_string(index=False))
