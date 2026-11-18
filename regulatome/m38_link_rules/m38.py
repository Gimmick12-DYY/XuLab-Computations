# m38 (SPEC.md): link rules vs Perturb-seq, plain overlap. Descriptive.
import sys, os
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; X = "/work/users/d/y/dyy12/XuLab"; O = f"{R}/m38_link_rules/out"; V = "/vast/som/xujie_lab/TAGATG-293T/bigwig"; NB = f"{R}/m2c_abc_canonical/results/HEK293T_hicfixed"
UG = np.load(f"{R}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True)["genes"].astype(str); NG = len(UG); gi = {g: k for k, g in enumerate(UG)}
EL = pd.read_csv(f"{NB}/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end"]); NE = len(EL); ek = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(EL.chr, EL.start, EL.end))}
L = pd.read_csv(f"{NB}/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "isSelfPromoter"]); L = L[L.TargetGene.isin(gi)].copy()
Le = (L.chr + ":" + L.start.astype(str) + "-" + L.end.astype(str)).map(ek).values.astype(int); Lg = L.TargetGene.map(gi).values; Ls = L.isSelfPromoter.astype(str).str.lower().eq("true").values
prom_el = np.zeros(NE, bool); prom_el[Le[Ls]] = True; linked_el = np.zeros(NE, bool); linked_el[Le] = True
GL = pd.read_csv(f"{NB}/Neighborhoods/GeneList.txt", sep="\t", usecols=["chr", "name", "tss"]).drop_duplicates("name", keep="last").set_index("name").reindex(UG)
TSSc, TSSp = GL.chr.values.astype(str), GL.tss.values.astype(float)
TB = {}
for ch in np.unique(TSSc):
    idx = np.flatnonzero(TSSc == ch); o = np.argsort(TSSp[idx]); TB[ch] = (TSSp[idx][o], idx[o])
CHI = {}
for ch, G in EL.groupby("chr"): o = np.argsort(G.start.values); CHI[ch] = (G.start.values[o], G.end.values[o], G.index.values[o])
def elem_of(c, p):
    r = np.full(len(p), -1)
    for ch in np.unique(c):
        if ch not in CHI: continue
        st, en, ix = CHI[ch]; m = c == ch; k = np.searchsorted(st, p[m], side="right") - 1; ok = (k >= 0) & (p[m] < en[np.maximum(k, 0)]); rr = np.full(m.sum(), -1); rr[ok] = ix[k[ok]]; r[m] = rr
    return r
def rule_abc(c, p):
    e = elem_of(c, p); v = np.zeros(NE, bool); v[e[e >= 0]] = True; m = np.zeros(NG, bool); m[Lg[v[Le]]] = True
    used = np.zeros(len(p), bool); used[e >= 0] = linked_el[e[e >= 0]]; return m, used
def window(c, p, w):
    m = np.zeros(NG, bool)
    for ch in np.unique(c):
        if ch not in TB: continue
        tp, ix = TB[ch]; x = p[c == ch]; a = np.searchsorted(tp, x - w); b = np.searchsorted(tp, x + w, side="right")
        for i, j in zip(a, b): m[ix[i:j]] = True
    return m
def nearest(c, p, cap):
    m = np.zeros(NG, bool)
    for ch in np.unique(c):
        if ch not in TB: continue
        tp, ix = TB[ch]; x = p[c == ch]; k = np.clip(np.searchsorted(tp, x), 1, len(tp) - 1) if len(tp) > 1 else np.zeros(len(x), int)
        if len(tp) > 1: k = np.where(np.abs(tp[k - 1] - x) <= np.abs(tp[k] - x), k - 1, k)
        d = np.abs(tp[k] - x); ok = d <= cap; m[ix[k[ok]]] = True
    return m
def cicero(t):
    f = f"{X}/cobinding/work/ourrun_k50/{t}/{t}.peak_edges.tsv"; m = np.zeros(NG, bool)
    if not os.path.exists(f): return m, False
    E = pd.read_csv(f, sep="\t")
    def parse(s): c, se = s.str.split(":", n=1).str[0], s.str.split(":", n=1).str[1]; a = se.str.split("-").str[0].astype(int); b = se.str.split("-").str[1].astype(int); return c.values, ((a + b) // 2).values
    c1, m1 = parse(E.peak1); c2, m2 = parse(E.peak2); far = (c1 == c2) & (np.abs(m1 - m2) > 25000)
    for cc, mm in ((c1[far], m1[far]), (c2[far], m2[far])): m |= window(cc, mm, 1000)
    return m, True
T25 = pd.read_csv(f"{R}/m23_sgrna_top25/out/observed_kd_top25.tsv.gz", sep="\t", usecols=["tf", "gene", "z_top25"]); TFS = sorted(set(T25.tf))
Z = T25.pivot(index="tf", columns="gene", values="z_top25").reindex(index=TFS, columns=UG).values.astype(float); OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
for i, t in enumerate(TFS):
    for o in OWN.get(t, [t]):
        if o in gi: Z[i, gi[o]] = np.nan
RULES = ["A current", "B A + 10 kb", "C A + 50 kb", "D100 A + nearest TSS (<= 100 kb)", "Dall A + nearest TSS (no cap)", "E A + coordinated binding", "F10k top 10,000 peaks", "F20k top 20,000 peaks", "Fq all q<0.05 peaks", "G B + E"]
LIST = {r: np.zeros((len(TFS), NG), bool) for r in RULES}; INFO = []
for i, t in enumerate(TFS):
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 8, 9], names=["c", "s", "sig", "q", "off"]); P["rk"] = P.sig.rank(ascending=False, method="first"); P["sm"] = P.s + P.off
    def sub(n): x = P[P.rk <= n]; return x.c.values.astype(str), x.sm.values
    c5, p5 = sub(5000); A, used = rule_abc(c5, p5); LIST["A current"][i] = A
    LIST["B A + 10 kb"][i] = A | window(c5, p5, 10000); LIST["C A + 50 kb"][i] = A | window(c5, p5, 50000)
    LIST["D100 A + nearest TSS (<= 100 kb)"][i] = A | nearest(c5[~used], p5[~used], 100000); LIST["Dall A + nearest TSS (no cap)"][i] = A | nearest(c5[~used], p5[~used], np.inf)
    Ec, has = cicero(t); LIST["E A + coordinated binding"][i] = A | Ec; LIST["G B + E"][i] = LIST["B A + 10 kb"][i] | Ec
    for r, n in (("F10k top 10,000 peaks", 10000), ("F20k top 20,000 peaks", 20000)): LIST[r][i] = rule_abc(*sub(n))[0]
    q = P[P.q >= -np.log10(0.05)]; LIST["Fq all q<0.05 peaks"][i] = rule_abc(q.c.values.astype(str), q.sm.values)[0]
    INFO.append(dict(TF=t, peaks_total=len(P), peaks_q005=len(q), cicero_edge_file=has, genes_added_by_E_alone=int((Ec & ~A).sum())))
    print(t, flush=True)
OK = np.isfinite(Z); A_ = np.abs(np.nan_to_num(Z))
CUTS = {"|z| >= 2": OK & (A_ >= 2), "|z| >= 3": OK & (A_ >= 3), "|z| >= 4": OK & (A_ >= 4), "|z| >= 5": OK & (A_ >= 5), "down only (z <= -2)": OK & (np.nan_to_num(Z) <= -2), "up only (z >= 2)": OK & (np.nan_to_num(Z) >= 2)}
PT, PO = [], []
for r in RULES:
    B = LIST[r] & OK
    for cn, Pm in CUTS.items():
        k = (B & Pm).sum(1); b = B.sum(1); p = Pm.sum(1)
        PO.append({"rule": r, "Perturb-seq genes": cn, "TFs with >= 1 Perturb-seq gene": int((p > 0).sum()), "our genes (pooled)": int(b.sum()), "Perturb-seq genes (pooled)": int(p.sum()), "in both (pooled)": int(k.sum()),
                   "share of Perturb-seq genes found in our Regulome (%)": 100 * k.sum() / p.sum(), "share of our Regulome genes found in Perturb-seq (%)": 100 * k.sum() / b.sum(),
                   "median per TF, share of Perturb-seq genes found (%)": float(np.median(100 * k[p > 0] / p[p > 0])), "median per TF, share of our genes found (%)": float(np.median(100 * k / np.maximum(b, 1))), "median our genes per TF": float(np.median(b))})
        if cn in ("|z| >= 2", "|z| >= 3"):
            for i, t in enumerate(TFS): PT.append({"rule": r, "Perturb-seq genes": cn, "TF": t, "our genes": int(b[i]), "Perturb-seq genes (n)": int(p[i]), "in both": int(k[i]),
                                                   "share of Perturb-seq genes found in our Regulome (%)": 100 * k[i] / p[i] if p[i] else np.nan, "share of our Regulome genes found in Perturb-seq (%)": 100 * k[i] / b[i] if b[i] else np.nan})
PO = pd.DataFrame(PO); PT = pd.DataFrame(PT); INFO = pd.DataFrame(INFO)
PO.to_csv(f"{O}/link_rules_pooled.tsv", sep="\t", index=False, float_format="%.2f"); PT.to_csv(f"{O}/link_rules_per_tf.tsv", sep="\t", index=False, float_format="%.2f"); INFO.to_csv(f"{O}/link_rules_info.tsv", sep="\t", index=False)
sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
for n, d in (("link_rules_pooled", PO), ("link_rules_per_tf", PT), ("link_rules_info", INFO)): d.to_excel(f"{O}/{n}.xlsx", index=False)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 20)
a = PO[(PO.rule == "A current") & (PO["Perturb-seq genes"] == "|z| >= 2")].iloc[0]; print("SANITY rule A:", a["our genes (pooled)"], a["in both (pooled)"], "(m36: 117845, 11337)")
for cn in CUTS: print("==", cn); print(PO[PO["Perturb-seq genes"] == cn].drop(columns=["Perturb-seq genes"]).round(1).to_string(index=False))
g2 = PT[PT["Perturb-seq genes"] == "|z| >= 2"].pivot(index="TF", columns="rule", values="in both")
for r in ("B A + 10 kb", "E A + coordinated binding"):
    d = (g2[r] - g2["A current"]).sort_values(ascending=False).head(5); print("== top gains in 'in both' under", r)
    for t, v in d.items():
        row = PT[(PT["Perturb-seq genes"] == "|z| >= 2") & (PT.TF == t)].set_index("rule"); print(t, "+%d" % v, "| A:", int(row.loc["A current", "our genes"]), int(row.loc["A current", "in both"]), round(row.loc["A current", "share of Perturb-seq genes found in our Regulome (%)"], 1), round(row.loc["A current", "share of our Regulome genes found in Perturb-seq (%)"], 1), "|", r[:1], ":", int(row.loc[r, "our genes"]), int(row.loc[r, "in both"]), round(row.loc[r, "share of Perturb-seq genes found in our Regulome (%)"], 1), round(row.loc[r, "share of our Regulome genes found in Perturb-seq (%)"], 1))
print("E: TFs with an edge file", int(INFO.cicero_edge_file.sum()), "| genes added by E alone, total", int(INFO.genes_added_by_E_alone.sum()), "median", INFO.genes_added_by_E_alone.median())
print("done")
