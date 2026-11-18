# m41 Amendment 1: co-regulation of partner-added genes, non-partners matched on Perturb-seq gene count, broad class and added genes.
import sys
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"
s = open(f"{R}/m41_partner_added_genes/m41.py").read(); exec(s[:s.index("def pick(")])
rng = np.random.default_rng(20261118); ND = 100; PS = {k: v for k, v in PS.items() if k.startswith(("P1", "P2"))}
def co(L, x, p, Pm):
    i, j = ti[x], ti[p]; a = L[p] & ~L[x] & OK[i] & OK[j]; n = a.sum(); hx = (a & Pm[i]).sum(); hp = (a & Pm[j]).sum(); b = a & Pm[i] & Pm[j]
    return dict(n=n, hx=hx, hp=hp, both=b.sum(), same=(b & (SG[i] == SG[j])).sum(), exp=hx * hp / n if n else 0.0)
POOL, PAIR, LISTP = [], [], []
for rule in ("A", "G"):
    L = LISTS[rule]
    for c in (2, 3):
        Pm = OK & (AZ >= c); npm = {u: int(Pm[ti[u]].sum()) for u in TFS}
        for pname, pd_ in PS.items():
            pairs = [(x, p) for x in TFS for p in sorted(pd_.get(x, set())) if p in ti and p != x]
            if rule == "A" and c == 2:
                for x, p in pairs: LISTP.append({"partner set": pname, "TF": x, "partner with a knockdown": p, "partner is broad": p in BROAD, "partner's Perturb-seq genes (|z| >= 2)": npm[p]})
            real = [co(L, x, p, Pm) for x, p in pairs]; draws = [[None] * len(pairs) for _ in range(ND)]; nfb = 0
            for k, (x, p) in enumerate(pairs):
                pool = [u for u in TFS if u != x and u not in pd_.get(x, set()) and ((u in BROAD) == (p in BROAD))]
                lx = L[x] & OK[ti[x]]; ad = {u: int((L[u] & ~lx & OK[ti[x]] & OK[ti[u]]).sum()) for u in pool}; ra = real[k]["n"]
                c1 = [u for u in pool if abs(npm[u] - npm[p]) <= .2 * max(npm[p], 1) and abs(ad[u] - ra) <= .2 * max(ra, 1)]
                if not c1: c1 = [u for u in pool if abs(npm[u] - npm[p]) <= .2 * max(npm[p], 1)]; nfb += 1
                if not c1: c1 = sorted(pool, key=lambda u: abs(npm[u] - npm[p]))[:5]
                for d_ in range(ND): draws[d_][k] = co(L, x, c1[rng.integers(len(c1))], Pm)
                r = real[k]; PAIR.append({"rule": rule, "cutoff": f"|z| >= {c}", "partner set": pname, "TF": x, "partner": p, "partner is broad": p in BROAD, "partner's Perturb-seq genes": npm[p], "genes added": r["n"],
                    "added: changed on TF knockdown": r["hx"], "added: changed on partner knockdown": r["hp"], "changed on both (observed)": r["both"], "changed on both (expected from the two shares)": r["exp"],
                    "observed / expected": r["both"] / r["exp"] if r["exp"] else np.nan, "non-partners: changed on both (mean)": np.mean([dr[k]["both"] for dr in draws]), "non-partner candidates": len(c1)})
            def agg(rs, sel):
                rs = [r for r, s_ in zip(rs, sel) if s_]; n = sum(r["n"] for r in rs); b = sum(r["both"] for r in rs); e = sum(r["exp"] for r in rs)
                return dict(n=n, P=100 * sum(r["hp"] for r in rs) / max(n, 1), B=100 * b / max(n, 1), S=100 * sum(r["same"] for r in rs) / max(b, 1), OE=b / e if e else np.nan)
            for lab, sel in (("all pairs", [True] * len(pairs)), ("pairs involving RBBP7 removed", [("RBBP7" not in (x, p)) for x, p in pairs]), ("pairs with a broad partner removed", [p not in BROAD for x, p in pairs])):
                if not any(sel): continue
                rr = agg(real, sel); dd = pd.DataFrame([agg(dr, sel) for dr in draws])
                POOL.append({"rule": rule, "cutoff": f"|z| >= {c}", "partner set": pname, "pairs": lab, "n pairs": int(sum(sel)), "genes added": int(rr["n"]),
                    "real: share changed on partner knockdown (%)": rr["P"], "non-partners: same (%)": dd.P.mean(), "non-partners 5th-95th": f"{np.percentile(dd.P,5):.2f}-{np.percentile(dd.P,95):.2f}",
                    "real: share changed on both (%)": rr["B"], "non-partners: changed on both (%)": dd.B.mean(), "non-partners 5th-95th (both)": f"{np.percentile(dd.B,5):.2f}-{np.percentile(dd.B,95):.2f}",
                    "real: same direction (%)": rr["S"], "non-partners: same direction (%)": dd.S.mean(), "real: observed / expected (both)": rr["OE"], "non-partners: observed / expected (mean)": dd.OE.mean(),
                    "non-partners: observed / expected 5th-95th": f"{np.nanpercentile(dd.OE,5):.2f}-{np.nanpercentile(dd.OE,95):.2f}", "pairs where matching fell back": nfb})
            print(rule, c, pname, len(pairs), "fallbacks", nfb, flush=True)
T = {"coreg_pooled": pd.DataFrame(POOL), "coreg_per_pair": pd.DataFrame(PAIR), "coreg_pairs_with_partner_knockdown": pd.DataFrame(LISTP)}
O = f"{R}/m41_partner_added_genes/out"
for n, d in T.items(): d.to_csv(f"{O}/{n}.tsv", sep="\t", index=False, float_format="%.4g")
sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
with pd.ExcelWriter(f"{O}/partner_coregulation_matched.xlsx") as w:
    for n, d in T.items(): d.to_excel(w, sheet_name=n[:31], index=False)
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 40)
for n, d in T.items(): print("==", n); print(d.round(2).to_string(index=False))
print("done")
