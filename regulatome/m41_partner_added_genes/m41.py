# m41_partner_added_genes (SPEC.md): real partners vs non-partners, per added gene.
import os, sys
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"
src = open(f"{R}/m38_link_rules/m38.py").read(); exec(src[:src.index("T25 = pd.read_csv")])
O = f"{R}/m41_partner_added_genes/out"; rng = np.random.default_rng(20261118); ND = 100
tfs79 = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]; LISTS = {"A": {}, "G": {}}
for t in tfs79:
    f = f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
    if not os.path.exists(f) or os.path.getsize(f) == 0: continue
    P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= 5000]; c5, p5 = P.c.values.astype(str), (P.s + P.off).values
    A, _ = rule_abc(c5, p5); LISTS["A"][t] = A; LISTS["G"][t] = A | window(c5, p5, 10000) | cicero(t)[0]
ALLT = sorted(LISTS["A"])
T25 = pd.read_csv(f"{R}/m23_sgrna_top25/out/observed_kd_top25.tsv.gz", sep="\t", usecols=["tf", "gene", "z_top25"]); TFS = sorted(set(T25.tf) & set(ALLT)); ti = {t: i for i, t in enumerate(TFS)}
Z = T25.pivot(index="tf", columns="gene", values="z_top25").reindex(index=TFS, columns=UG).values.astype(float); OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
for i, t in enumerate(TFS):
    for o in OWN.get(t, [t]):
        if o in gi: Z[i, gi[o]] = np.nan
OK = np.isfinite(Z); AZ = np.abs(np.nan_to_num(Z)); SG = np.sign(np.nan_to_num(Z))
kd = np.load(f"{R}/m4_validation/kd_response/out_v2/arm2_all_cell/kd_response.npz", allow_pickle=True); keep = kd["ntc_detect"] >= .1
ex = pd.Series(kd["ntc_mean"][keep], index=kd["genes"][keep].astype(str)); ex = ex[~ex.index.duplicated(keep="last")].reindex(UG).fillna(0).values
TB_ = pd.read_csv(f"{NB}/Neighborhoods/GeneList.TSS1kb.bed", sep="\t", header=None, usecols=[3], names=["g"]); AT_ = pd.read_csv(f"{NB}/Neighborhoods/Genes.TSS1kb.ATAC.omniatac_merged.tn5.tagAlign.gz.CountReads.bedgraph", sep="\t", header=None, usecols=[3], names=["n"])
at = pd.Series(AT_.n.values, index=TB_.g.values); at = at[~at.index.duplicated(keep="last")].reindex(UG).fillna(0).values
dec = lambda v: pd.qcut(pd.Series(v).rank(method="first"), 10, labels=False).values; STRAT = dec(ex) * 10 + dec(at)
BROAD = sorted(t for t in ALLT if LISTS["A"][t].sum() > 4000); print("broad TFs (rule A > 4,000 genes):", len(BROAD), BROAD, flush=True)
CF = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_frozen_v1.tsv", sep="\t"); CA = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_all3081.tsv", sep="\t")
def sym(pairs):
    d = {}
    for a_, b_ in pairs: d.setdefault(a_, set()).add(b_); d.setdefault(b_, set()).add(a_)
    return d
PS = {"P1 Figure 4 trio": sym([("RBBP4", "ZNF644"), ("ADNP", "MTF2"), ("DR1", "NFYA")]), "P2 AF3-supported (strong)": sym(zip(CF.tf_a[CF.af3_5seed_tier == "strong"], CF.tf_b[CF.af3_5seed_tier == "strong"]))}
for k in (1, 3):
    d = {}
    for t in set(CA.tf_a) | set(CA.tf_b):
        s = CA[(CA.tf_a == t) | (CA.tf_b == t)].copy(); s["p"] = np.where(s.tf_a == t, s.tf_b, s.tf_a); d[t] = set(s.sort_values(["best_pct", "pct_all"], ascending=False).head(k).p)
    PS[f"P4 top {k} by co-binding"] = d
def pick(pool, val, target):
    v = np.array([val[u] for u in pool]); ok = np.flatnonzero(np.abs(v - target) <= .2 * max(target, 1))
    if len(ok) == 0: ok = np.argsort(np.abs(v - target))[:5]
    return pool[ok[rng.integers(len(ok))]]
def stats(L, x, p, Pm, c):
    i = ti[x]; ok = OK[i]; lx = L[x] & ok; lp = L[p] & ok; add = lp & ~lx; sh = lx & lp; xo = lx & ~lp; nb = ok & ~lx & ~lp; pm = Pm[i]
    # matched expectation for added genes: rate among neither-bound genes per stratum
    s_nb = STRAT[nb]; rate = np.bincount(s_nb, weights=pm[nb], minlength=100) / np.maximum(np.bincount(s_nb, minlength=100), 1); expct = rate[STRAT[add]].sum()
    out = dict(jac=(L[x] & L[p]).sum() / max((L[x] | L[p]).sum(), 1), inx=(L[x] & L[p]).sum() / max(L[p].sum(), 1), n_add=add.sum(), add_hit=(add & pm).sum(), add_exp=expct,
               n_sh=sh.sum(), sh_hit=(sh & pm).sum(), n_xo=xo.sum(), xo_hit=(xo & pm).sum(), n_own=lx.sum(), own_hit=(lx & pm).sum(), n_nb=nb.sum(), nb_hit=(nb & pm).sum())
    if p in ti:
        j = ti[p]; a2 = add & OK[j]; pj = OK[j] & (AZ[j] >= c); both = a2 & pm & pj
        out.update(co_n=a2.sum(), co_p=(a2 & pj).sum(), co_both=both.sum(), co_same=(both & (SG[i] == SG[j])).sum())
    return out
ROWS, PAIR = [], []
for rule in ("A", "G"):
    L = LISTS[rule]; size = {u: int(L[u].sum()) for u in ALLT}
    for c in (2, 3):
        Pm = OK & (AZ >= c)
        for pname, pd_ in PS.items():
            pairs = [(x, p) for x in TFS for p in sorted(pd_.get(x, set())) if p in L and p != x]
            if not pairs: continue
            real = [stats(L, x, p, Pm, c) for x, p in pairs]
            for (x, p), r in zip(pairs, real):
                if pname.startswith(("P1", "P2")): PAIR.append({"rule": rule, "cutoff": f"|z| >= {c}", "partner set": pname, "TF": x, "partner": p, "partner is broad": p in BROAD, **r})
            draws = {k: [[] for _ in range(ND)] for k in ("a", "b", "c", "d", "co")}
            for (x, p), r in zip(pairs, real):
                pool = np.array([u for u in ALLT if u != x and u not in pd_.get(x, set())]); poolc = np.array([u for u in pool if u not in BROAD]); poolk = np.array([u for u in pool if u in ti])
                lx = L[x] & OK[ti[x]]; addn = {u: int((L[u] & OK[ti[x]] & ~lx).sum()) for u in pool}; shn = {u: int((L[u] & lx).sum()) for u in pool}
                for d_ in range(ND):
                    draws["a"][d_].append(stats(L, x, pick(pool, size, size[p]), Pm, c)); draws["b"][d_].append(stats(L, x, pick(pool, addn, r["n_add"]), Pm, c))
                    draws["c"][d_].append(stats(L, x, pick(poolc, addn, r["n_add"]), Pm, c)); draws["d"][d_].append(stats(L, x, pick(pool, shn, r["n_sh"]), Pm, c))
                    if "co_n" in r: draws["co"][d_].append(stats(L, x, pick(poolk, {u: addn[u] for u in poolk}, r["n_add"]), Pm, c))
            def agg(rs, sel=None):
                df = pd.DataFrame(rs) if sel is None else pd.DataFrame(rs)[sel]
                if len(df) == 0: return {}
                s = df.sum(numeric_only=True); o = dict(pairs=len(df), mean_jaccard=df.jac.mean(), mean_share_of_P_in_X=df.inx.mean(), added=s.n_add, added_hit=100 * s.add_hit / max(s.n_add, 1), added_matched=100 * s.add_exp / max(s.n_add, 1),
                    shared_hit=100 * s.sh_hit / max(s.n_sh, 1), xonly_hit=100 * s.xo_hit / max(s.n_xo, 1), own_hit=100 * s.own_hit / max(s.n_own, 1), neither_hit=100 * s.nb_hit / max(s.n_nb, 1))
                if "co_n" in df and df.co_n.notna().any():
                    dd = df[df.co_n.notna()]; s2 = dd.sum(numeric_only=True); o.update(co_pairs=len(dd), co_P=100 * s2.co_p / max(s2.co_n, 1), co_both=100 * s2.co_both / max(s2.co_n, 1), co_sign=100 * s2.co_same / max(s2.co_both, 1))
                return o
            base = {"rule": rule, "cutoff": f"|z| >= {c}", "partner set": pname}
            isb = np.array([p in BROAD for _, p in pairs])
            for lab, sel in (("all pairs", None), ("partner is a broad TF", isb), ("partner is not a broad TF", ~isb)):
                if sel is not None and not sel.any(): continue
                ROWS.append({**base, "who": "real partners", "pair group": lab, **agg(real, sel)})
            for k, lab in (("a", "non-partners, same list size"), ("b", "non-partners, same number of added genes"), ("c", "non-partners, same number of added genes, no broad TFs"), ("d", "non-partners, same number of shared genes"), ("co", "non-partners with a knockdown, same number of added genes")):
                A_ = [agg(dr) for dr in draws[k] if dr]
                if not A_: continue
                A_ = pd.DataFrame(A_); row = {**base, "who": lab, "pair group": "all pairs"}
                for col in A_.columns:
                    row[col] = A_[col].mean()
                    if col in ("added_hit", "added_matched", "shared_hit", "co_P", "co_both", "co_sign", "mean_jaccard", "added"): row[col + " 5-95"] = f"{np.percentile(A_[col],5):.3g}-{np.percentile(A_[col],95):.3g}"
                ROWS.append(row)
            print(rule, c, pname, len(pairs), flush=True)
REN = {"mean_jaccard": "mean Jaccard of lists", "mean_share_of_P_in_X": "mean share of partner's list already in TF's list", "added": "genes added (total)", "added_hit": "added genes: share that are TF's Perturb-seq genes (%)",
       "added_matched": "matched genes (same expression x openness): share (%)", "shared_hit": "genes bound by both: share that are TF's Perturb-seq genes (%)", "xonly_hit": "genes bound by TF only: share (%)",
       "own_hit": "TF's own genes: share (%)", "neither_hit": "genes bound by neither: share (%)", "co_pairs": "pairs where partner has a knockdown", "co_P": "added genes: share changed on partner knockdown (%)",
       "co_both": "added genes: share changed on both knockdowns (%)", "co_sign": "changed on both: same direction (%)"}
T = pd.DataFrame(ROWS); T["added minus matched (points)"] = T.added_hit - T.added_matched
T.columns = [REN.get(c_.replace(" 5-95", ""), c_.replace(" 5-95", "")) + (" 5th-95th" if c_.endswith(" 5-95") else "") for c_ in T.columns]
PP = pd.DataFrame(PAIR).rename(columns={"jac": "Jaccard", "inx": "share of partner's list already in TF's list", "n_add": "genes added", "add_hit": "added and TF's Perturb-seq gene", "add_exp": "matched expectation (genes)", "n_sh": "bound by both", "sh_hit": "bound by both and changed",
    "n_xo": "TF only", "xo_hit": "TF only and changed", "n_own": "TF's own genes", "own_hit": "own and changed", "n_nb": "neither", "nb_hit": "neither and changed", "co_n": "added, partner knockdown usable", "co_p": "added, changed on partner knockdown", "co_both": "added, changed on both", "co_same": "changed on both, same direction"})
PP["added: share changed (%)"] = 100 * PP["added and TF's Perturb-seq gene"] / PP["genes added"].clip(lower=1); PP["matched (%)"] = 100 * PP["matched expectation (genes)"] / PP["genes added"].clip(lower=1)
T.to_csv(f"{O}/partner_added_genes_pooled.tsv", sep="\t", index=False, float_format="%.4g"); PP.to_csv(f"{O}/partner_added_genes_per_pair.tsv", sep="\t", index=False, float_format="%.4g")
sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
with pd.ExcelWriter(f"{O}/partner_added_genes.xlsx") as w: T.to_excel(w, sheet_name="pooled", index=False); PP.to_excel(w, sheet_name="per_pair_P1_P2", index=False)
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 50); pd.set_option("display.max_colwidth", 40)
print(T.round(2).to_string(index=False)); print("done")
