# Merge batch score tables into screen/scores.tsv with ipTM length residuals (fit on finished heterodimers)
import glob, sys
import numpy as np, pandas as pd
root = sys.argv[1]
fs = sorted(glob.glob(f"{root}/screen/batch_scores/*.tsv"))
s = pd.concat([pd.read_csv(f, sep="\t").assign(batch=f.split("/")[-1][:-4]) for f in fs])
j = pd.read_csv(f"{root}/jobs/joblist.tsv", sep="\t")
s = j.merge(s, on="pair_id", how="inner")
ok = (s.status == "ok") & (s.set == "het")
s["iptm_len_resid"] = np.nan
if ok.sum() > 20:
    b = np.polyfit(np.log(s.loc[ok, "n_tokens"]), s.loc[ok, "iptm_top"].astype(float), 1)
    m = s.status == "ok"
    s.loc[m, "iptm_len_resid"] = s.loc[m, "iptm_top"].astype(float) - np.polyval(b, np.log(s.loc[m, "n_tokens"]))
s["hit"] = np.where((s.status == "ok") & (s.ipsae_max.astype(float) >= 0.3) & (s.iptm_top.astype(float) >= 0.4), "strong",
           np.where((s.status == "ok") & (s.ipsae_max.astype(float) >= 0.1), "weak", "."))
# STAG2 hits: partner chains wrap the STAG2 HEAT surface (~10% of the interface in the CES pocket; stag2_pocket.tsv),
# so they are reported as a separate class, not as complex hits
s["stag2_class"] = np.where(((s.chain_a == "STAG2") | (s.chain_b == "STAG2")) & (s.hit != "."), "STAG2-wrap", ".")
# hub_partner (annotation, not a filter): a TF's hit degree (strong + weak, scored heterodimers) vs a null that permutes hit
# labels across pairs within token-length deciles (1,000 permutations, seed 1); flagged if degree > null 95th percentile
het = s[(s.status == "ok") & (s.set == "het")].copy()
hub_rows = []
if len(het) > 50:
    isht = (het.hit != ".").values; dec = pd.qcut(het.n_tokens, 10, labels=False, duplicates="drop").values
    tfs = sorted(set(het.chain_a) | set(het.chain_b)); ti = {t: i for i, t in enumerate(tfs)}
    ia, ib = het.chain_a.map(ti).values, het.chain_b.map(ti).values
    def degree(h): return np.bincount(ia[h], minlength=len(tfs)) + np.bincount(ib[h], minlength=len(tfs))
    obs = degree(isht); rng = np.random.default_rng(1); null = np.empty((1000, len(tfs)))
    for k in range(1000):
        hp = isht.copy()
        for d in np.unique(dec):
            ix = np.flatnonzero(dec == d); hp[ix] = rng.permutation(hp[ix])
        null[k] = degree(hp)
    npairs = np.bincount(ia, minlength=len(tfs)) + np.bincount(ib, minlength=len(tfs))
    p95 = np.percentile(null, 95, axis=0)
    hub_rows = pd.DataFrame({"tf": tfs, "pairs_scored": npairs, "hit_degree": obs, "null_mean": null.mean(0), "null_p95": p95,
                             "p": ((null >= obs).sum(0) + 1) / 1001, "hub_partner": obs > p95})
    hub_rows.sort_values("hit_degree", ascending=False).to_csv(f"{root}/screen/hub_partner.tsv", sep="\t", index=False, float_format="%.3f")
    hubs = set(hub_rows[hub_rows.hub_partner].tf)
    s["hub_partner"] = [",".join(sorted({a, b} & hubs)) or "." for a, b in zip(s.chain_a, s.chain_b)]
else:
    s["hub_partner"] = "."
s.sort_values("ipsae_max", ascending=False).to_csv(f"{root}/screen/scores.tsv", sep="\t", index=False, float_format="%.4f")
print(len(s), "scored;", s.status.value_counts().to_dict(), s.hit.value_counts().to_dict())
