# m39 (SPEC.md): Regulome v3 (rule G) and the partner rule. Descriptive.
import sys, os
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"
src = open(f"{R}/m38_link_rules/m38.py").read(); exec(src[:src.index("T25 = pd.read_csv")])  # UG, gi, NG, rule_abc, window, cicero, V, X
O39 = f"{R}/m39_partner_rule/out"; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
tfs79 = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]
LA, LG, ADD10, ADDE = {}, {}, {}, {}
for t in tfs79:
    f = f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
    if not os.path.exists(f) or os.path.getsize(f) == 0: continue
    P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= 5000]; c5, p5 = P.c.values.astype(str), (P.s + P.off).values
    A, _ = rule_abc(c5, p5); W = window(c5, p5, 10000); Ec, _ = cicero(t)
    LA[t] = A; LG[t] = A | W | Ec; ADD10[t] = W & ~A; ADDE[t] = Ec & ~A & ~W
print("lists built for", len(LA), "TFs", flush=True)
# Part 1: regulome_v3
V2 = pd.read_csv(f"{R}/m34_fig5_panels_v3/out_v2/regulome_v2.tsv.gz", sep="\t", low_memory=False)
pp = V2.promoter_peak.astype(str).str.lower().eq("true"); V2["link_type"] = np.where(pp, "promoter", "ABC")
add = []
for t in LA:
    for lab, m in (("within_10kb", ADD10[t]), ("coordinated_binding", ADDE[t])):
        for j in np.flatnonzero(m): add.append((t, UG[j], lab))
AD = pd.DataFrame(add, columns=["tf", "gene", "link_type"])
T25 = pd.read_csv(f"{R}/m23_sgrna_top25/out/observed_kd_top25.tsv.gz", sep="\t", usecols=["tf", "gene", "z_top25"]); TFS = sorted(set(T25.tf))
zmap = T25.set_index(["tf", "gene"]).z_top25; OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
z = zmap.reindex(list(zip(AD.tf, AD.gene))).values.astype(float)
for k, (t, g) in enumerate(zip(AD.tf, AD.gene)):
    if g in OWN.get(t, [t]): z[k] = np.nan
a = np.abs(np.nan_to_num(z)); AD["knockdown_available"] = AD.tf.isin(TFS); AD["knockdown_z"] = z; AD["supported"] = a >= 2
AD["direction"] = np.where(a >= 2, np.where(z < 0, "activates", "represses"), "none"); AD["confidence"] = np.where(a >= 4, "|z| >= 4", np.where(a >= 3, "|z| >= 3", np.where(a >= 2, "|z| >= 2", "none")))
V3 = pd.concat([V2, AD], ignore_index=True); V3 = V3.rename(columns={"supported": "in_both (|z| >= 2)"}); V3["label"] = "Regulome v3 (rule G: promoter/ABC + 10 kb + coordinated binding); Perturb-seq columns from top-25% sgRNA cells, external, descriptive"
V3.to_csv(f"{O39}/regulome_v3.tsv.gz", sep="\t", index=False, float_format="%.4g"); V3.to_excel(f"{O39}/regulome_v3.xlsx", index=False)
tpg = V3.groupby("gene").tf.nunique(); gpt = V3.groupby("tf").gene.nunique()
print("V3: edges", len(V3), "| TFs", V3.tf.nunique(), "| genes", V3.gene.nunique(), "| median TFs per gene", tpg.median(), "| median genes per TF", gpt.median(), "| by link_type", V3.link_type.value_counts().to_dict(), flush=True)
# Part 2: partner rule
Z = T25.pivot(index="tf", columns="gene", values="z_top25").reindex(index=TFS, columns=UG).values.astype(float)
for i, t in enumerate(TFS):
    for o in OWN.get(t, [t]):
        if o in gi: Z[i, gi[o]] = np.nan
OK = np.isfinite(Z); AZ = np.abs(np.nan_to_num(Z)); ti = {t: i for i, t in enumerate(TFS)}
CF = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_frozen_v1.tsv", sep="\t"); CA = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_all3081.tsv", sep="\t")
def sym(pairs):
    d = {}
    for a_, b_ in pairs: d.setdefault(a_, set()).add(b_); d.setdefault(b_, set()).add(a_)
    return d
PS = {"P1 Figure 4 trio": sym([("RBBP4", "ZNF644"), ("ADNP", "MTF2"), ("DR1", "NFYA")]), "P2 AF3-supported (strong)": sym(zip(CF.tf_a[CF.af3_5seed_tier == "strong"], CF.tf_b[CF.af3_5seed_tier == "strong"])),
      "P3 frozen candidate list": sym(zip(CF.tf_a[CF.candidate], CF.tf_b[CF.candidate]))}
for k in (1, 3, 5):
    d = {}
    for t in set(CA.tf_a) | set(CA.tf_b):
        s = CA[(CA.tf_a == t) | (CA.tf_b == t)].copy(); s["p"] = np.where(s.tf_a == t, s.tf_b, s.tf_a); s = s.sort_values(["best_pct", "pct_all"], ascending=False).head(k); d[t] = set(s.p)
    PS[f"P4 top {k} by co-binding"] = d
print({k: len(v) for k, v in PS.items()}, flush=True)
CUTS = {"|z| >= 2": 2, "|z| >= 3": 3, "|z| >= 4": 4, "|z| >= 5": 5}
POOL, PER, ONLY = [], [], []
for base, LL in (("A current", LA), ("G (A + 10 kb + coordinated binding)", LG)):
    for pname, pdct in [("no partners", {})] + list(PS.items()):
        lists = np.zeros((len(TFS), NG), bool); npart = np.zeros(len(TFS), int); own = np.zeros((len(TFS), NG), bool)
        for i, t in enumerate(TFS):
            own[i] = LL.get(t, np.zeros(NG, bool)); m = own[i].copy(); ps = [p for p in pdct.get(t, set()) if p in LL]; npart[i] = len(ps)
            for p in ps: m |= LL[p]
            lists[i] = m
        haspart = npart > 0
        for cn, c in CUTS.items():
            Pm = OK & (AZ >= c); B = lists & OK; k_ = (B & Pm).sum(1); b_ = B.sum(1); p_ = Pm.sum(1); kb = (own & OK & Pm).sum(1); bb = (own & OK).sum(1)
            for sub, sel in (("all 63 TFs", np.ones(len(TFS), bool)), ("only TFs with a partner", haspart)):
                if sel.sum() == 0: continue
                POOL.append({"base rule": base, "partner set": pname, "Perturb-seq genes": cn, "TFs": sub, "n TFs": int(sel.sum()), "TFs with >= 1 partner": int(haspart.sum()), "median partners per TF (TFs with a partner)": float(np.median(npart[haspart])) if haspart.any() else 0,
                             "our genes (TF or partner, pooled)": int(b_[sel].sum()), "Perturb-seq genes (pooled)": int(p_[sel].sum()), "in both (pooled)": int(k_[sel].sum()),
                             "share of Perturb-seq genes found in our Regulome (%)": 100 * k_[sel].sum() / max(p_[sel].sum(), 1), "share of our Regulome genes found in Perturb-seq (%)": 100 * k_[sel].sum() / max(b_[sel].sum(), 1),
                             "same TFs, no partners: in both": int(kb[sel].sum()), "same TFs, no partners: share of Perturb-seq genes found (%)": 100 * kb[sel].sum() / max(p_[sel].sum(), 1), "same TFs, no partners: share of our genes found (%)": 100 * kb[sel].sum() / max(bb[sel].sum(), 1)})
            if cn in ("|z| >= 2", "|z| >= 3"):
                for i, t in enumerate(TFS):
                    PER.append({"base rule": base, "partner set": pname, "Perturb-seq genes": cn, "TF": t, "partners": ",".join(sorted(p for p in pdct.get(t, set()) if p in LL)), "our genes (TF or partner)": int(b_[i]), "our genes (TF alone)": int(bb[i]),
                                "Perturb-seq genes (n)": int(p_[i]), "in both": int(k_[i]), "in both (TF alone)": int(kb[i]), "share of Perturb-seq genes found in our Regulome (%)": 100 * k_[i] / p_[i] if p_[i] else np.nan, "share of our Regulome genes found in Perturb-seq (%)": 100 * k_[i] / b_[i] if b_[i] else np.nan})
            if cn == "|z| >= 2" and pname != "no partners":
                tot = chk = also = 0
                for i, t in enumerate(TFS):
                    only = lists[i] & ~own[i] & OK[i] & Pm[i]
                    for j in np.flatnonzero(only):
                        tot += 1; bps = [p for p in pdct.get(t, set()) if p in LL and LL[p][j]]; usable = [p for p in bps if p in ti and np.isfinite(Z[ti[p], j])]
                        if usable: chk += 1; also += any(AZ[ti[p], j] >= 2 for p in usable)
                ONLY.append({"base rule": base, "partner set": pname, "genes in both found only through a partner": tot, "of these, a binding partner has a usable knockdown": chk, "partner's own knockdown also changes the gene (|z| >= 2)": also, "share (%)": 100 * also / chk if chk else np.nan})
    # size reference: any of the 79 TFs
    anyb = np.zeros(NG, bool)
    for t in LL: anyb |= LL[t]
    for cn, c in CUTS.items():
        Pm = OK & (AZ >= c); B = np.tile(anyb, (len(TFS), 1)) & OK; k_ = (B & Pm).sum(); POOL.append({"base rule": base, "partner set": "size reference: any of the 79 TFs bound", "Perturb-seq genes": cn, "TFs": "all 63 TFs", "n TFs": len(TFS),
                    "our genes (TF or partner, pooled)": int(B.sum()), "Perturb-seq genes (pooled)": int(Pm.sum()), "in both (pooled)": int(k_), "share of Perturb-seq genes found in our Regulome (%)": 100 * k_ / Pm.sum(), "share of our Regulome genes found in Perturb-seq (%)": 100 * k_ / B.sum()})
PO = pd.DataFrame(POOL); PT = pd.DataFrame(PER); ON = pd.DataFrame(ONLY)
ref = PO["partner set"].str.startswith("size reference")
PO[~ref].to_csv(f"{O39}/partner_rule_pooled.tsv", sep="\t", index=False, float_format="%.2f"); PO[ref].dropna(axis=1, how="all").to_csv(f"{O39}/partner_rule_size_reference.tsv", sep="\t", index=False, float_format="%.2f")
PT.to_csv(f"{O39}/partner_rule_per_tf.tsv", sep="\t", index=False, float_format="%.2f"); ON.to_csv(f"{O39}/partner_only_genes_check.tsv", sep="\t", index=False, float_format="%.2f")
with pd.ExcelWriter(f"{O39}/partner_rule_pooled.xlsx") as w:
    PO[~ref].to_excel(w, sheet_name="pooled", index=False); PO[ref].dropna(axis=1, how="all").to_excel(w, sheet_name="size_reference", index=False); ON.to_excel(w, sheet_name="partner_only_genes", index=False)
PT.to_excel(f"{O39}/partner_rule_per_tf.xlsx", index=False)
pd.set_option("display.width", 330); pd.set_option("display.max_columns", 30)
s = PO[(PO["partner set"] == "no partners") & (PO["Perturb-seq genes"] == "|z| >= 2") & (PO.TFs == "all 63 TFs")]; print("SANITY:", s[["base rule", "share of Perturb-seq genes found in our Regulome (%)", "share of our Regulome genes found in Perturb-seq (%)"]].round(1).to_string(index=False))
cols = ["base rule", "partner set", "TFs", "n TFs", "TFs with >= 1 partner", "median partners per TF (TFs with a partner)", "our genes (TF or partner, pooled)", "in both (pooled)", "share of Perturb-seq genes found in our Regulome (%)", "share of our Regulome genes found in Perturb-seq (%)", "same TFs, no partners: in both", "same TFs, no partners: share of Perturb-seq genes found (%)", "same TFs, no partners: share of our genes found (%)"]
for cn in CUTS: print("==", cn); print(PO[(~ref) & (PO["Perturb-seq genes"] == cn)][cols].round(1).to_string(index=False))
print("== size reference"); print(PO[ref][["base rule", "Perturb-seq genes", "our genes (TF or partner, pooled)", "in both (pooled)", "share of Perturb-seq genes found in our Regulome (%)", "share of our Regulome genes found in Perturb-seq (%)"]].round(1).to_string(index=False))
print("== partner-only genes"); print(ON.round(1).to_string(index=False))
for base in ("A current", "G (A + 10 kb + coordinated binding)"):
    for pname in ("P2 AF3-supported (strong)", "P4 top 3 by co-binding"):
        g = PT[(PT["base rule"] == base) & (PT["partner set"] == pname) & (PT["Perturb-seq genes"] == "|z| >= 2")].copy(); g["gain"] = g["in both"] - g["in both (TF alone)"]
        print("== top gains", base, "|", pname); print(g.nlargest(5, "gain")[["TF", "partners", "our genes (TF alone)", "our genes (TF or partner)", "in both (TF alone)", "in both", "gain", "share of Perturb-seq genes found in our Regulome (%)", "share of our Regulome genes found in Perturb-seq (%)"]].round(1).to_string(index=False))
print("done")
