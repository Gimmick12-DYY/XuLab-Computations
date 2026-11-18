# m37 (SPEC.md): list sizes and capped gene lists vs Perturb-seq, plain overlap. Descriptive.
import sys, os
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; O = f"{R}/m37_target_cap/out"; V = "/vast/som/xujie_lab/TAGATG-293T/bigwig"; NB = f"{R}/m2c_abc_canonical/results/HEK293T_hicfixed"
UG = np.load(f"{R}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True)["genes"].astype(str); NG = len(UG); gi = {g: k for k, g in enumerate(UG)}
EL = pd.read_csv(f"{NB}/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end"]); NE = len(EL); ek = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(EL.chr, EL.start, EL.end))}
L = pd.read_csv(f"{NB}/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score", "isSelfPromoter"]); L = L[L.TargetGene.isin(gi)].copy()
L["e"] = (L.chr + ":" + L.start.astype(str) + "-" + L.end.astype(str)).map(ek).astype(int); L["g"] = L.TargetGene.map(gi); L["self"] = L.isSelfPromoter.astype(str).str.lower().eq("true")
Le, Lg, La, Ls = L.e.values, L.g.values, L["ABC.Score"].values, L.self.values
prom_el = np.zeros(NE, bool); prom_el[Le[Ls]] = True; linked_el = np.zeros(NE, bool); linked_el[Le] = True; ngenes_el = np.bincount(Le, minlength=NE)
own_prom = {}  # gene -> its self-promoter elements
for e, g in zip(Le[Ls], Lg[Ls]): own_prom.setdefault(g, []).append(e)
CHI = {}
for ch, G in EL.groupby("chr"): o = np.argsort(G.start.values); CHI[ch] = (G.start.values[o], G.end.values[o], G.index.values[o])
def elem_of(c, p):
    r = np.full(len(p), -1)
    for ch in np.unique(c):
        if ch not in CHI: continue
        st, en, ix = CHI[ch]; m = c == ch; k = np.searchsorted(st, p[m], side="right") - 1; ok = (k >= 0) & (p[m] < en[np.maximum(k, 0)]); rr = np.full(m.sum(), -1); rr[ok] = ix[k[ok]]; r[m] = rr
    return r
tfs79 = [l.strip() for l in open(f"{R}/m1_binary/v2/tfs.txt") if l.strip()] if os.path.exists(f"{R}/m1_binary/v2/tfs.txt") else [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]
SIZE = []; SCORE = {}; PROMO = {}
for t in tfs79:
    f = f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
    if not os.path.exists(f) or os.path.getsize(f) == 0: continue
    P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= 5000]
    e = elem_of(P.c.values.astype(str), (P.s + P.off).values); v = np.zeros(NE); m = e >= 0; np.maximum.at(v, e[m], P.sig.values[m])
    inprom = m.copy(); inprom[m] = prom_el[e[m]]; indist = m.copy(); indist[m] = (~prom_el[e[m]]) & linked_el[e[m]]
    sc = np.zeros(NG); np.maximum.at(sc, Lg, v[Le] * La); SCORE[t] = sc
    viaP = np.zeros(NG, bool); viaP[Lg[(v[Le] > 0) & prom_el[Le]]] = True; viaD = np.zeros(NG, bool); viaD[Lg[(v[Le] > 0) & ~prom_el[Le]]] = True
    pr = np.zeros(NG, bool)
    for g, es in own_prom.items():
        if (v[es] > 0).any(): pr[g] = True
    PROMO[t] = pr; ng = int((sc > 0).sum()); dist_el = np.unique(e[indist])
    SIZE.append(dict(TF=t, peaks=len(P), peaks_in_promoter_element=int(inprom.sum()), peaks_in_linked_distal_element=int(indist.sum()), peaks_in_neither=int(len(P) - inprom.sum() - indist.sum()),
                     pct_peaks_promoter=100 * inprom.mean(), our_genes=ng, genes_via_promoter_elements=int(viaP.sum()), genes_via_distal_elements=int(viaD.sum()), genes_only_via_distal=int((viaD & ~viaP).sum()),
                     genes_per_peak=ng / max(int(inprom.sum() + indist.sum()), 1), mean_genes_per_distal_element_with_peak=float(ngenes_el[dist_el].mean()) if len(dist_el) else np.nan))
    print(t, ng, flush=True)
S = pd.DataFrame(SIZE).sort_values("our_genes", ascending=False); S.to_csv(f"{O}/list_sizes_per_tf.tsv", sep="\t", index=False, float_format="%.2f")
T25 = pd.read_csv(f"{R}/m23_sgrna_top25/out/observed_kd_top25.tsv.gz", sep="\t", usecols=["tf", "gene", "z_top25"]); TFS = [t for t in sorted(set(T25.tf)) if t in SCORE]
Z = T25.pivot(index="tf", columns="gene", values="z_top25").reindex(index=TFS, columns=UG).values.astype(float); OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
for i, t in enumerate(TFS):
    for o in OWN.get(t, [t]):
        if o in gi: Z[i, gi[o]] = np.nan
VERS = ["uncapped (current Regulome)", "top 2,000 genes", "top 1,000 genes", "top 500 genes", "top 250 genes", "promoter-only"]
def ours(t, ver):
    sc = SCORE[t]
    if ver.startswith("uncapped"): return sc > 0
    if ver == "promoter-only": return PROMO[t]
    n = int(ver.split()[1].replace(",", "")); m = np.zeros(NG, bool); idx = np.flatnonzero(sc > 0); m[idx[np.argsort(-sc[idx])[:n]]] = True; return m
ROWS = []
for ver in VERS:
    for cut in (2, 3):
        for i, t in enumerate(TFS):
            ok = np.isfinite(Z[i]); b = ours(t, ver) & ok; p = ok & (np.abs(np.nan_to_num(Z[i])) >= cut); k = int((b & p).sum())
            ROWS.append({"version": ver, "cutoff": f"|z| >= {cut}", "TF": t, "our genes": int(b.sum()), "Perturb-seq genes": int(p.sum()), "in both": k,
                         "share of Perturb-seq genes found in our Regulome (%)": 100 * k / p.sum() if p.sum() else np.nan, "share of our Regulome genes found in Perturb-seq (%)": 100 * k / b.sum() if b.sum() else np.nan})
T = pd.DataFrame(ROWS); T.to_csv(f"{O}/capped_overlap_per_tf.tsv", sep="\t", index=False, float_format="%.2f")
SM = []
for (ver, cut), g in T.groupby(["version", "cutoff"], sort=False):
    SM.append({"version": ver, "cutoff": cut, "TFs": len(g), "our genes (pooled)": int(g["our genes"].sum()), "Perturb-seq genes (pooled)": int(g["Perturb-seq genes"].sum()), "in both (pooled)": int(g["in both"].sum()),
               "share of Perturb-seq genes found in our Regulome (%), pooled": 100 * g["in both"].sum() / g["Perturb-seq genes"].sum(), "share of our Regulome genes found in Perturb-seq (%), pooled": 100 * g["in both"].sum() / g["our genes"].sum(),
               "median per TF, share of Perturb-seq genes found (%)": float(g["share of Perturb-seq genes found in our Regulome (%)"].median()), "median per TF, share of our genes found (%)": float(g["share of our Regulome genes found in Perturb-seq (%)"].median()),
               "median our genes per TF": float(g["our genes"].median())})
SM = pd.DataFrame(SM); SM.to_csv(f"{O}/capped_overlap_summary.tsv", sep="\t", index=False, float_format="%.2f")
sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
for n, d in (("list_sizes_per_tf", S), ("capped_overlap_per_tf", T), ("capped_overlap_summary", SM)): d.to_excel(f"{O}/{n}.xlsx", index=False)
pd.set_option("display.width", 300)
print(S.head(12).round(2).to_string(index=False)); print("median:", S.drop(columns="TF").median().round(2).to_dict())
print(SM.round(1).to_string(index=False))
FOC = ["RBBP7", "XPA", "NFYA", "NME2", "CTCF", "NRF1", "E4F1", "DNMT1", "ADNP", "MAZ"]
F = T[(T.cutoff == "|z| >= 2") & T.TF.isin(FOC)]
for col in ("our genes", "in both", "share of Perturb-seq genes found in our Regulome (%)", "share of our Regulome genes found in Perturb-seq (%)"):
    print("==", col); print(F.pivot(index="TF", columns="version", values=col)[VERS].round(1).to_string())
print("done")
