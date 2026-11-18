# m40 Amendment 1: permutation tests (Perm 1 genes within strata, Perm 2 TF labels, Perm 3 partner labels) with positive and negative controls.
import sys, os
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"
src = open(f"{R}/m38_link_rules/m38.py").read(); exec(src[:src.index("T25 = pd.read_csv")])  # UG, gi, NG, rule_abc, window, cicero, V, X, NB
O = f"{R}/m40_overlap_reference/out"; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps"); rng = np.random.default_rng(20261118); NP = 1000
pval = lambda obs, perm: (1 + int((np.asarray(perm) >= obs).sum())) / (len(perm) + 1)
tfs79 = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]; LISTS = {"A": {}, "G": {}}
for t in tfs79:
    f = f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
    if not os.path.exists(f) or os.path.getsize(f) == 0: continue
    P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= 5000]; c5, p5 = P.c.values.astype(str), (P.s + P.off).values
    A, _ = rule_abc(c5, p5); LISTS["A"][t] = A; LISTS["G"][t] = A | window(c5, p5, 10000) | cicero(t)[0]
ALLT = sorted(LISTS["A"])
D33 = f"{R}/m33_abc_pipeline_audit"; CH = [("CTCF", f"{X}/data/CTCF_HEK293T_GSE103651_peaks.bed"), ("MAZ", f"{X}/data/MAZ_HEK293_peaks.bed"), ("ZIC2", f"{X}/data/ZIC2_HEK293_peaks.bed"), ("ZNF282", f"{X}/data/ZNF282_HEK293_peaks.bed"),
      ("ZNF777", f"{X}/data/ZNF777_HEK293_peaks.bed"), ("ZBTB7A", f"{X}/data/ZBTB7A_HEK293_peaks.bed"), ("SETDB1", f"{X}/data/SETDB1_HEK293_peaks.bed"), ("NFYA", f"{X}/data/NFYA_K562_peaks.bed"), ("NRF1", f"{D33}/encode/ENCFF259YUE.bed.gz"), ("E4F1", f"{D33}/encode/ENCFF683WRK.bed.gz")]
BULK = {"A": {}, "G": {}}
for t, f in CH:
    P = pd.read_csv(f, sep="\t", header=None, compression="gzip" if f.endswith(".gz") else None); has = P.shape[1] >= 10
    P = P.iloc[:, [0, 1, 2, 6] + ([9] if has else [])]; P.columns = ["c", "s", "e", "sig"] + (["off"] if has else []); P["off"] = P.off if has else (P.e - P.s) // 2; P.loc[P.off < 0, "off"] = (P.e - P.s) // 2
    P = P[P.sig.rank(ascending=False, method="first") <= 5000]; c5, p5 = P.c.values.astype(str), (P.s + P.off).values; A, _ = rule_abc(c5, p5); BULK["A"][t] = A; BULK["G"][t] = A | window(c5, p5, 10000)
T25 = pd.read_csv(f"{R}/m23_sgrna_top25/out/observed_kd_top25.tsv.gz", sep="\t", usecols=["tf", "gene", "z_top25"]); TFS = sorted(set(T25.tf) & set(ALLT)); NT = len(TFS); ti = {t: i for i, t in enumerate(TFS)}
Z = T25.pivot(index="tf", columns="gene", values="z_top25").reindex(index=TFS, columns=UG).values.astype(float); OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
for i, t in enumerate(TFS):
    for o in OWN.get(t, [t]):
        if o in gi: Z[i, gi[o]] = np.nan
OK = np.isfinite(Z); AZ = np.abs(np.nan_to_num(Z))
kd = np.load(f"{R}/m4_validation/kd_response/out_v2/arm2_all_cell/kd_response.npz", allow_pickle=True); keep = kd["ntc_detect"] >= .1
ex = pd.Series(kd["ntc_mean"][keep], index=kd["genes"][keep].astype(str)); ex = ex[~ex.index.duplicated(keep="last")].reindex(UG).fillna(0).values
TB_ = pd.read_csv(f"{NB}/Neighborhoods/GeneList.TSS1kb.bed", sep="\t", header=None, usecols=[3], names=["g"]); AT_ = pd.read_csv(f"{NB}/Neighborhoods/Genes.TSS1kb.ATAC.omniatac_merged.tn5.tagAlign.gz.CountReads.bedgraph", sep="\t", header=None, usecols=[3], names=["n"])
at = pd.Series(AT_.n.values, index=TB_.g.values); at = at[~at.index.duplicated(keep="last")].reindex(UG).fillna(0).values
dec = lambda v: pd.qcut(pd.Series(v).rank(method="first"), 10, labels=False).values; STRAT = dec(ex) * 10 + dec(at)
EX3 = {"NRF1", "E4F1", "DNMT1"}
def perm1_draws(Bi, Pi, OKi, strat=True):
    s = STRAT[OKi] if strat else np.zeros(OKi.sum(), int); b = Bi[OKi]; p = Pi[OKi]; tot = np.zeros(NP)
    for st in np.unique(s):
        m = s == st; n = int(m.sum()); nb = int(b[m].sum()); k = int(p[m].sum())
        if k and nb: tot += rng.hypergeometric(nb, n - nb, k, size=NP)
    return tot
def perm1(Blist, Pm, tfs, strat=True):
    obs, draws, den = [], [], []
    for t in tfs:
        i = ti[t]; obs.append(int((Blist[t] & Pm[i] & OK[i]).sum())); den.append(int((Pm[i] & OK[i]).sum())); draws.append(perm1_draws(Blist[t], Pm[i], OK[i], strat))
    return np.array(obs), np.array(draws), np.array(den)
def derange(groups):
    perm = np.arange(NT)
    for g in groups:
        g = np.array(g)
        if len(g) < 2: return None
        while True:
            q = rng.permutation(g)
            if not (q == g).any(): break
        perm[g] = q
    return perm
P1, P1T, P1U, P2, P2T, P3, POS, NEG = [], [], [], [], [], [], [], []
def pooled_rows(store, base, obs, draws, den, tfs, extra):
    for en, ex_ in (("all TFs", set()), ("without NRF1, E4F1, DNMT1", EX3)):
        sel = np.array([t not in ex_ for t in tfs]) & (den > 0)
        if not sel.any(): continue
        o = 100 * obs[sel].sum() / den[sel].sum(); dd = 100 * draws[sel].sum(0) / den[sel].sum()
        store.append({**base, "TFs": en, "n TFs": int(sel.sum()), "Perturb-seq genes": int(den[sel].sum()), "in both": int(obs[sel].sum()), "share found, observed (%)": o, "permutation mean (%)": dd.mean(), "permutation 2.5-97.5 (%)": f"{np.percentile(dd,2.5):.1f}-{np.percentile(dd,97.5):.1f}", "p (one-sided)": pval(o, dd), **extra})
for rule in ("A", "G"):
    L = LISTS[rule]
    for c in (2, 3, 4, 5):
        Pm = OK & (AZ >= c); base = {"rule": rule, "Perturb-seq genes cutoff": f"|z| >= {c}"}
        obs, draws, den = perm1(L, Pm, TFS); pooled_rows(P1, base, obs, draws, den, TFS, {})
        for k, t in enumerate(TFS):
            if den[k]: P1T.append({**base, "TF": t, "Perturb-seq genes": int(den[k]), "in both": int(obs[k]), "share found (%)": 100 * obs[k] / den[k], "permutation mean (%)": 100 * draws[k].mean() / den[k], "permutation 2.5-97.5 (%)": f"{100*np.percentile(draws[k],2.5)/den[k]:.1f}-{100*np.percentile(draws[k],97.5)/den[k]:.1f}", "p (one-sided)": pval(obs[k], draws[k])})
        ou, du, deu = perm1(L, Pm, TFS, strat=False); pooled_rows(P1U, {**base, "note": "reference only (unstratified gene shuffle)"}, ou, du, deu, TFS, {})
        # Perm 2
        M = np.array([[int((L[TFS[j]] & Pm[i] & OK[i]).sum()) for i in range(NT)] for j in range(NT)]); den2 = Pm.sum(1); own = np.diag(M).astype(float)
        sz = np.array([L[t].sum() for t in TFS]); qn = pd.qcut(pd.Series(sz).rank(method="first"), 5, labels=False).values
        for version, groups in (("all TFs (derangement)", [list(range(NT))]), ("same list-size quintile", [list(np.flatnonzero(qn == q)) for q in range(5)])):
            pt = np.zeros((NP, NT))
            for r_ in range(NP): pi = derange(groups); pt[r_] = M[pi, np.arange(NT)]
            pooled_rows(P2, {**base, "version": version}, own, pt.T, den2, TFS, {})
            for k, t in enumerate(TFS):
                if den2[k]: P2T.append({**base, "version": version, "TF": t, "share found, own list (%)": 100 * own[k] / den2[k], "permutation mean (%)": 100 * pt[:, k].mean() / den2[k], "permutation 2.5-97.5 (%)": f"{100*np.percentile(pt[:,k],2.5)/den2[k]:.1f}-{100*np.percentile(pt[:,k],97.5)/den2[k]:.1f}", "p (one-sided)": pval(own[k], pt[:, k])})
        # positive control: bulk ChIP
        bt = [t for t in BULK[rule] if t in ti]; bo, bd, bden = perm1(BULK[rule], Pm, bt)
        for k, t in enumerate(bt):
            if bden[k]: POS.append({**base, "test": "Perm 1 (genes within strata)", "list": f"bulk ChIP {t}", "share found (%)": 100 * bo[k] / bden[k], "permutation mean (%)": 100 * bd[k].mean() / bden[k], "p (one-sided)": pval(bo[k], bd[k])})
        o = 100 * bo.sum() / bden.sum(); dd = 100 * bd.sum(0) / bden.sum(); POS.append({**base, "test": "Perm 1 (genes within strata)", "list": "bulk ChIP, 10 TFs pooled", "share found (%)": o, "permutation mean (%)": dd.mean(), "p (one-sided)": pval(o, dd)})
        oo, od, oden = perm1(L, Pm, bt); o = 100 * oo.sum() / oden.sum(); dd = 100 * od.sum(0) / oden.sum(); POS.append({**base, "test": "Perm 1 (genes within strata)", "list": "our HITAG, same 10 TFs pooled", "share found (%)": o, "permutation mean (%)": dd.mean(), "p (one-sided)": pval(o, dd)})
        bi = [ti[t] for t in bt]; MB = np.array([[int((BULK[rule][bt[j]] & Pm[i] & OK[i]).sum()) for i in bi] for j in range(len(bt))]); dB = Pm[bi].sum(1); pt = np.zeros((NP, len(bt)))
        for r_ in range(NP):
            while True:
                q = rng.permutation(len(bt))
                if not (q == np.arange(len(bt))).any(): break
            pt[r_] = MB[q, np.arange(len(bt))]
        o = 100 * np.trace(MB) / dB.sum(); dd = 100 * pt.sum(1) / dB.sum(); POS.append({**base, "test": "Perm 2 (TF labels, among the 10)", "list": "bulk ChIP, 10 TFs pooled", "share found (%)": o, "permutation mean (%)": dd.mean(), "p (one-sided)": pval(o, dd)})
        # negative control (|z| >= 2 and 3 only)
        if c in (2, 3):
            Ps = np.zeros_like(Pm)
            for i in range(NT):
                idx = np.flatnonzero(OK[i]); s = STRAT[idx]; pv = Pm[i, idx].copy()
                for st in np.unique(s): m = s == st; pv[m] = rng.permutation(pv[m])
                Ps[i, idx] = pv
            no, nd, nden = perm1(L, Ps, TFS); p1 = [pval(no[k], nd[k]) for k in range(NT) if nden[k]]
            Mn = np.array([[int((L[TFS[j]] & Ps[i] & OK[i]).sum()) for i in range(NT)] for j in range(NT)]); ptn = np.zeros((NP, NT))
            for r_ in range(NP): pi = derange([list(range(NT))]); ptn[r_] = Mn[pi, np.arange(NT)]
            p2 = [pval(Mn[k, k], ptn[:, k]) for k in range(NT) if nden[k]]
            for nm, pp in (("Perm 1", p1), ("Perm 2", p2)): NEG.append({**base, "test": nm, "TFs": len(pp), "median p": float(np.median(pp)), "share of TFs with p < 0.05": float(np.mean(np.array(pp) < .05)), "share with p < 0.5": float(np.mean(np.array(pp) < .5))})
        print(rule, c, flush=True)
# Perm 3
CF = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_frozen_v1.tsv", sep="\t"); CA = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_all3081.tsv", sep="\t")
def sym(pairs):
    d = {}
    for a_, b_ in pairs: d.setdefault(a_, set()).add(b_); d.setdefault(b_, set()).add(a_)
    return d
PS = {"P1 Figure 4 trio": sym([("RBBP4", "ZNF644"), ("ADNP", "MTF2"), ("DR1", "NFYA")]), "P2 AF3-supported (strong)": sym(zip(CF.tf_a[CF.af3_5seed_tier == "strong"], CF.tf_b[CF.af3_5seed_tier == "strong"])), "P3 frozen candidate list": sym(zip(CF.tf_a[CF.candidate], CF.tf_b[CF.candidate]))}
for k in (1, 3, 5):
    d = {}
    for t in set(CA.tf_a) | set(CA.tf_b):
        s = CA[(CA.tf_a == t) | (CA.tf_b == t)].copy(); s["p"] = np.where(s.tf_a == t, s.tf_b, s.tf_a); d[t] = set(s.sort_values(["best_pct", "pct_all"], ascending=False).head(k).p)
    PS[f"P4 top {k} by co-binding"] = d
for rule in ("A", "G"):
    L = LISTS[rule]; szs = pd.Series({t: L[t].sum() for t in ALLT}); q79 = pd.qcut(szs.rank(method="first"), 5, labels=False)
    for pname, pdct in PS.items():
        real = [(t, [p for p in pdct.get(t, set()) if p in L and p != t]) for t in TFS]; real = [(t, ps) for t, ps in real if ps]
        if not real: continue
        for c in (2, 3, 4, 5):
            Pm = OK & (AZ >= c); den = np.array([int((Pm[ti[t]] & OK[ti[t]]).sum()) for t, _ in real])
            def found(t, ps):
                m = L[t].copy()
                for p in ps: m |= L[p]
                return int((m & Pm[ti[t]] & OK[ti[t]]).sum()), int((m & OK[ti[t]]).sum())
            ob = np.array([found(t, ps)[0] for t, ps in real]); alone = np.array([found(t, [])[0] for t, _ in real]); dr = np.zeros((NP, len(real)))
            for r_ in range(NP):
                for k, (t, ps) in enumerate(real):
                    pick = []
                    for p in ps:
                        cand = [u for u in ALLT if q79[u] == q79[p] and u != t and u not in pdct.get(t, set()) and u not in pick]
                        if cand: pick.append(cand[rng.integers(len(cand))])
                    dr[r_, k] = found(t, pick)[0]
            for en, ex_ in (("all TFs with a partner", set()), ("without NRF1, E4F1, DNMT1", EX3)):
                sel = np.array([t not in ex_ for t, _ in real]) & (den > 0)
                if not sel.any(): continue
                o = 100 * ob[sel].sum() / den[sel].sum(); dd = 100 * dr[:, sel].sum(1) / den[sel].sum()
                P3.append({"rule": rule, "partner set": pname, "Perturb-seq genes cutoff": f"|z| >= {c}", "TFs": en, "n TFs": int(sel.sum()), "share found, TF alone (%)": 100 * alone[sel].sum() / den[sel].sum(), "share found, TF + real partners (%)": o,
                           "TF + same-size non-partners, permutation mean (%)": dd.mean(), "permutation 2.5-97.5 (%)": f"{np.percentile(dd,2.5):.1f}-{np.percentile(dd,97.5):.1f}", "p (one-sided)": pval(o, dd)})
    print("perm3", rule, flush=True)
T = {"perm1_genes_within_strata_pooled": pd.DataFrame(P1), "perm1_per_tf": pd.DataFrame(P1T), "perm1_unstratified_reference_only": pd.DataFrame(P1U), "perm2_tf_labels_pooled": pd.DataFrame(P2), "perm2_per_tf": pd.DataFrame(P2T),
     "perm3_partner_labels_pooled": pd.DataFrame(P3), "control_positive_bulk_chip": pd.DataFrame(POS), "control_negative_shuffled_data": pd.DataFrame(NEG)}
SIG = []
for rule in ("A", "G"):
    for c in ("|z| >= 2", "|z| >= 3"):
        a = T["perm1_per_tf"]; a = a[(a.rule == rule) & (a["Perturb-seq genes cutoff"] == c) & (a["p (one-sided)"] < .05)]
        b = T["perm2_per_tf"]; b = b[(b.rule == rule) & (b["Perturb-seq genes cutoff"] == c) & (b.version == "all TFs (derangement)") & (b["p (one-sided)"] < .05)]
        b2 = T["perm2_per_tf"]; b2 = b2[(b2.rule == rule) & (b2["Perturb-seq genes cutoff"] == c) & (b2.version == "same list-size quintile") & (b2["p (one-sided)"] < .05)]
        SIG.append({"rule": rule, "cutoff": c, "expected by chance among 63": 3.15, "Perm 1 p < 0.05 (n)": len(a), "Perm 1 TFs": ",".join(a.TF), "Perm 2 p < 0.05 (n)": len(b), "Perm 2 TFs": ",".join(b.TF), "Perm 2 same-quintile p < 0.05 (n)": len(b2), "Perm 2 same-quintile TFs": ",".join(b2.TF)})
T["per_tf_p_below_005"] = pd.DataFrame(SIG)
for n, d in T.items(): d.to_csv(f"{O}/{n}.tsv", sep="\t", index=False, float_format="%.4g")
with pd.ExcelWriter(f"{O}/permutation_tests.xlsx") as w:
    for n, d in T.items(): d.to_excel(w, sheet_name=n[:31], index=False)
pd.set_option("display.width", 330); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 120)
for n in ("perm1_genes_within_strata_pooled", "perm2_tf_labels_pooled", "perm3_partner_labels_pooled", "control_positive_bulk_chip", "control_negative_shuffled_data", "per_tf_p_below_005"): print("==", n); print(T[n].round(3).to_string(index=False))
print("done")
