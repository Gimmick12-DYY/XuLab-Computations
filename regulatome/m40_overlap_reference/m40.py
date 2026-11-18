# m40 (SPEC.md): matched-gene, other-TF and partner comparisons for the overlap. Descriptive.
import sys, os
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"
src = open(f"{R}/m38_link_rules/m38.py").read(); exec(src[:src.index("T25 = pd.read_csv")])  # UG, gi, NG, rule_abc, window, cicero, V, X, NB
O40 = f"{R}/m40_overlap_reference/out"; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps"); rng = np.random.default_rng(20261118)
tfs79 = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]; LISTS = {"A": {}, "G": {}}
for t in tfs79:
    f = f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
    if not os.path.exists(f) or os.path.getsize(f) == 0: continue
    P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= 5000]; c5, p5 = P.c.values.astype(str), (P.s + P.off).values
    A, _ = rule_abc(c5, p5); LISTS["A"][t] = A; LISTS["G"][t] = A | window(c5, p5, 10000) | cicero(t)[0]
ALLT = sorted(LISTS["A"]); print("lists", len(ALLT), flush=True)
T25 = pd.read_csv(f"{R}/m23_sgrna_top25/out/observed_kd_top25.tsv.gz", sep="\t", usecols=["tf", "gene", "z_top25"]); TFS = sorted(set(T25.tf) & set(ALLT))
Z = T25.pivot(index="tf", columns="gene", values="z_top25").reindex(index=TFS, columns=UG).values.astype(float); OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
for i, t in enumerate(TFS):
    for o in OWN.get(t, [t]):
        if o in gi: Z[i, gi[o]] = np.nan
OK = np.isfinite(Z); AZ = np.abs(np.nan_to_num(Z)); NT = len(TFS)
# strata
kd = np.load(f"{R}/m4_validation/kd_response/out_v2/arm2_all_cell/kd_response.npz", allow_pickle=True); keep = kd["ntc_detect"] >= .1
ex = pd.Series(kd["ntc_mean"][keep], index=kd["genes"][keep].astype(str)); ex = ex[~ex.index.duplicated(keep="last")].reindex(UG).fillna(0).values
TB_ = pd.read_csv(f"{NB}/Neighborhoods/GeneList.TSS1kb.bed", sep="\t", header=None, usecols=[3], names=["g"]); AT_ = pd.read_csv(f"{NB}/Neighborhoods/Genes.TSS1kb.ATAC.omniatac_merged.tn5.tagAlign.gz.CountReads.bedgraph", sep="\t", header=None, usecols=[3], names=["n"])
at = pd.Series(AT_.n.values, index=TB_.g.values); at = at[~at.index.duplicated(keep="last")].reindex(UG).fillna(0).values
dec = lambda v: pd.qcut(pd.Series(v).rank(method="first"), 10, labels=False).values
STRAT = dec(ex) * 10 + dec(at)
EXCL = {"all TFs": set(), "without NRF1, E4F1, DNMT1": {"NRF1", "E4F1", "DNMT1"}}
def boot(num, den, sel, n=2000):
    idx = np.flatnonzero(sel); v = []
    for _ in range(n):
        b = rng.choice(idx, len(idx)); v.append(100 * num[b].sum() / max(den[b].sum(), 1e-9))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)
R1, R1T, DOSE, R2, R2T = [], [], [], [], []
for rule in ("A", "G"):
    B = np.vstack([LISTS[rule][t] for t in TFS])
    for c in (2, 3, 4, 5):
        ch = OK & (AZ >= c); nc = OK & (AZ < 1)
        nC, bC, mT, nU, bU = np.zeros(NT), np.zeros(NT), np.zeros(NT), np.zeros(NT), np.zeros(NT)
        for i in range(NT):
            s_ch = STRAT[ch[i]]; b_ch = B[i][ch[i]]
            nb = pd.Series(B[i][nc[i]]).groupby(STRAT[nc[i]]).mean(); okm = pd.Series(s_ch).isin(nb.index).values
            nC[i] = okm.sum(); bC[i] = b_ch[okm].sum(); mT[i] = nb.reindex(s_ch[okm]).sum(); nU[i] = nc[i].sum(); bU[i] = B[i][nc[i]].sum()
            if c in (2, 3): R1T.append({"rule": rule, "Perturb-seq genes": f"|z| >= {c}", "TF": TFS[i], "changed genes (matched)": int(nC[i]), "share bound, changed (%)": 100 * bC[i] / nC[i] if nC[i] else np.nan,
                                         "share bound, matched not-changed (%)": 100 * mT[i] / nC[i] if nC[i] else np.nan, "not-changed genes": int(nU[i]), "share bound, all not-changed (%)": 100 * bU[i] / nU[i] if nU[i] else np.nan})
        for en, ex_ in EXCL.items():
            sel = np.array([t not in ex_ for t in TFS]) & (nC > 0); lo, hi = boot(bC, nC, sel); lo2, hi2 = boot(mT, nC, sel)
            R1.append({"rule": rule, "Perturb-seq genes": f"|z| >= {c}", "TFs": en, "n TFs": int(sel.sum()), "changed genes": int(nC[sel].sum()), "share bound, changed (%)": 100 * bC[sel].sum() / nC[sel].sum(), "changed 95% interval over TFs": f"{lo:.1f}-{hi:.1f}",
                       "share bound, matched not-changed (%)": 100 * mT[sel].sum() / nC[sel].sum(), "matched 95% interval over TFs": f"{lo2:.1f}-{hi2:.1f}", "share bound, all not-changed (%)": 100 * bU[sel].sum() / nU[sel].sum()})
    bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, np.inf)]
    for lo_, hi_ in bins:
        m = OK & (AZ >= lo_) & (AZ < hi_); nb_ = m.sum(1); bb_ = (m & B).sum(1)
        for en, ex_ in EXCL.items():
            sel = np.array([t not in ex_ for t in TFS]) & (nb_ > 0); l, h = boot(bb_, nb_, sel)
            DOSE.append({"rule": rule, "|z| bin": f"{lo_}-{hi_}" if np.isfinite(hi_) else f">= {lo_}", "TFs": en, "TFs with genes in bin": int(sel.sum()), "genes": int(nb_[sel].sum()), "share bound (%)": 100 * bb_[sel].sum() / nb_[sel].sum(), "95% interval over TFs": f"{l:.1f}-{h:.1f}"})
    # reference 2
    ALLB = np.vstack([LISTS[rule][t] for t in ALLT]); sizes = ALLB.sum(1); ai = {t: k for k, t in enumerate(ALLT)}
    for c in (2, 3, 4, 5):
        ch = OK & (AZ >= c); own_f = np.zeros(NT); near_f = np.zeros(NT); den = ch.sum(1).astype(float)
        for i, t in enumerate(TFS):
            if den[i] == 0: continue
            f_all = (ALLB[:, ch[i]]).sum(1).astype(float); own = f_all[ai[t]]; oth = np.delete(f_all, ai[t]); osz = np.delete(sizes, ai[t]); near = np.argsort(np.abs(osz - sizes[ai[t]]))[:5]
            own_f[i] = own; near_f[i] = oth[near].mean(); rank = 1 + int((oth > own).sum())
            R2T.append({"rule": rule, "Perturb-seq genes": f"|z| >= {c}", "TF": t, "our genes": int(sizes[ai[t]]), "Perturb-seq genes (n)": int(den[i]), "share found, own list (%)": 100 * own / den[i],
                        "share found, 5 size-nearest TFs (mean, %)": 100 * oth[near].mean() / den[i], "share found, other 78 TFs median (%)": 100 * np.median(oth) / den[i], "other 78 TFs 5th-95th (%)": f"{100*np.percentile(oth,5)/den[i]:.1f}-{100*np.percentile(oth,95)/den[i]:.1f}", "rank of own list (1 = best of 79)": rank})
        for en, ex_ in EXCL.items():
            sel = np.array([t not in ex_ for t in TFS]) & (den > 0); l1, h1 = boot(own_f, den, sel); l2, h2 = boot(near_f, den, sel); g = pd.DataFrame(R2T); g = g[(g.rule == rule) & (g["Perturb-seq genes"] == f"|z| >= {c}") & ~g.TF.isin(ex_)]
            R2.append({"rule": rule, "Perturb-seq genes": f"|z| >= {c}", "TFs": en, "n TFs": int(sel.sum()), "share found, own lists (%)": 100 * own_f[sel].sum() / den[sel].sum(), "own 95% interval over TFs": f"{l1:.1f}-{h1:.1f}",
                       "share found, 5 size-nearest TFs (%)": 100 * near_f[sel].sum() / den[sel].sum(), "size-nearest 95% interval over TFs": f"{l2:.1f}-{h2:.1f}", "median rank of own list": float(g["rank of own list (1 = best of 79)"].median()), "TFs whose own list ranks in the top 5 of 79": int((g["rank of own list (1 = best of 79)"] <= 5).sum())})
    print("rule", rule, "refs 1-2 done", flush=True)
# reference 3
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
R3 = []
for rule in ("A", "G"):
    L = LISTS[rule]; sizes = {t: int(L[t].sum()) for t in ALLT}
    for pname, pd_ in PS.items():
        real = [(i, t, [p for p in pd_.get(t, set()) if p in L and p != t]) for i, t in enumerate(TFS)]; real = [x for x in real if x[2]]
        if not real: continue
        for c in (2, 3):
            ch = OK & (AZ >= c)
            def score(sets):
                ours = found = resp = 0
                for (i, t, _), ps in zip(real, sets):
                    m = L[t].copy()
                    for p in ps: m |= L[p]
                    mm = m & OK[i]; ours += mm.sum(); found += (mm & ch[i]).sum(); resp += ch[i].sum()
                return ours, found, resp
            o_r, f_r, rsp = score([x[2] for x in real]); o_b, f_b, _ = score([[] for _ in real]); draws = []
            for _ in range(100):
                sets = []
                for i, t, ps in real:
                    cand_all = [u for u in ALLT if u != t and u not in pd_.get(t, set())]; pick = []
                    for p in ps:
                        cand = [u for u in cand_all if abs(sizes[u] - sizes[p]) <= 0.2 * sizes[p] and u not in pick]
                        if not cand: cand = sorted([u for u in cand_all if u not in pick], key=lambda u: abs(sizes[u] - sizes[p]))[:3]
                        if cand: pick.append(cand[rng.integers(len(cand))])
                    sets.append(pick)
                o_d, f_d, _ = score(sets); draws.append((o_d, f_d))
            dd = np.array(draws, float)
            R3.append({"rule": rule, "partner set": pname, "Perturb-seq genes": f"|z| >= {c}", "TFs with a partner": len(real), "Perturb-seq genes (these TFs)": int(rsp),
                       "TF alone: our genes": int(o_b), "TF alone: in both": int(f_b), "TF alone: share of Perturb-seq genes found (%)": 100 * f_b / rsp, "TF alone: share of our genes found (%)": 100 * f_b / o_b,
                       "TF + real partners: our genes": int(o_r), "TF + real partners: in both": int(f_r), "TF + real partners: share of Perturb-seq genes found (%)": 100 * f_r / rsp, "TF + real partners: share of our genes found (%)": 100 * f_r / o_r,
                       "TF + size-matched non-partners (mean of 100): our genes": float(dd[:, 0].mean()), "... in both": float(dd[:, 1].mean()), "... share of Perturb-seq genes found (%)": 100 * dd[:, 1].mean() / rsp,
                       "... 5th-95th over draws (%)": f"{100*np.percentile(dd[:,1],5)/rsp:.1f}-{100*np.percentile(dd[:,1],95)/rsp:.1f}", "... share of our genes found (%)": 100 * (dd[:, 1] / dd[:, 0]).mean()})
    print("rule", rule, "ref 3 done", flush=True)
TABS = {"ref1_matched_genes_pooled": pd.DataFrame(R1), "ref1_matched_genes_per_tf": pd.DataFrame(R1T), "ref1_dose_response": pd.DataFrame(DOSE), "ref2_other_tfs_pooled": pd.DataFrame(R2), "ref2_other_tfs_per_tf": pd.DataFrame(R2T), "ref3_partner_vs_nonpartner": pd.DataFrame(R3)}
for n, d in TABS.items(): d.to_csv(f"{O40}/{n}.tsv", sep="\t", index=False, float_format="%.2f"); d.to_excel(f"{O40}/{n}.xlsx", index=False)
# per-TF reading
t1 = TABS["ref1_matched_genes_per_tf"]; t2 = TABS["ref2_other_tfs_per_tf"]; RD = []
for rule in ("A", "G"):
    for c in ("|z| >= 2", "|z| >= 3"):
        a = t1[(t1.rule == rule) & (t1["Perturb-seq genes"] == c)].set_index("TF"); b = t2[(t2.rule == rule) & (t2["Perturb-seq genes"] == c)].set_index("TF")
        for t in a.index:
            good = (a.loc[t, "share bound, changed (%)"] > a.loc[t, "share bound, matched not-changed (%)"]) and (b.loc[t, "rank of own list (1 = best of 79)"] <= 5) if t in b.index else False
            RD.append({"rule": rule, "Perturb-seq genes": c, "TF": t, "changed (%)": a.loc[t, "share bound, changed (%)"], "matched not-changed (%)": a.loc[t, "share bound, matched not-changed (%)"], "rank of own list of 79": b.loc[t, "rank of own list (1 = best of 79)"] if t in b.index else np.nan, "own list clearly better": bool(good)})
RD = pd.DataFrame(RD); RD.to_csv(f"{O40}/per_tf_reading.tsv", sep="\t", index=False, float_format="%.2f"); RD.to_excel(f"{O40}/per_tf_reading.xlsx", index=False)
pd.set_option("display.width", 330); pd.set_option("display.max_columns", 30)
for n in ("ref1_matched_genes_pooled", "ref1_dose_response", "ref2_other_tfs_pooled", "ref3_partner_vs_nonpartner"): print("==", n); print(TABS[n].round(1).to_string(index=False))
for rule in ("A", "G"):
    for c in ("|z| >= 2", "|z| >= 3"):
        g = RD[(RD.rule == rule) & (RD["Perturb-seq genes"] == c)]; print("==", rule, c, "clearly better:", ",".join(g.TF[g["own list clearly better"]]), "| n =", int(g["own list clearly better"].sum()), "of", len(g))
print("done")
