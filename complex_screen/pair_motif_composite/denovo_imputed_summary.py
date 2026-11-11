# Shared de novo motifs on imputed data for the correlation candidate pairs. A family is present in a TF when denovo_share.py put one of the TF's HOMER de novo
# motifs (fold >= 1.5, p <= 1e-10, odd chromosomes) in it. Held-out support: FIMO presence of the family representative in the TF's foreground vs background on
# even chromosomes + chrX (fold > 1, one-sided Fisher p < 0.05). Generic = family present in > 10 TFs (as denovo_share.py).
# Pair columns: shared families (non-generic, and all), those held-out in both TFs, novel = no Tomtom match (q < 0.05), and the empirical p of the held-out
# non-generic count vs 200 random panel pairs matched on depth tertiles and correlation-pct bin. Own-motif ranks from HOMER known are the sanity check.
# Output per_tf_matched/: shared_motif_candidates.tsv (novel, pct >= 0.60), shared_motif_all_pairs.tsv, shared_motif_controls.tsv, shared_motif_rates.tsv, own_motif_check.tsv, family_val.tsv
import os, re, glob, gzip
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = os.environ.get("TFBASE", f"{O}/per_tf_matched"); rng = np.random.default_rng(20261111); MAXTF = 10
F = pd.read_csv(f"{G}/denovo_families.tsv", sep="\t").set_index("family"); hit = {}
for f in glob.glob(f"{G}/denovo_cmp/fimo/c*.tsv.gz"):
    for l in gzip.open(f, "rt"): m, s = l.rstrip("\n").split("\t"); hit.setdefault(m, set()).add(s)
tfs = sorted(d for d in os.listdir(f"{G}/pools")); val = []
ids = lambda f, t: [x if x.startswith("b") else f"{t}_{x}" for x in (l.split("\t")[3] for l in open(f))]
for t in tfs:
    fg, bg = set(ids(f"{G}/pools/{t}/val/fg.bed", t)), set(ids(f"{G}/pools/{t}/val/bg.bed", t))
    for k in F.index:
        h = hit.get(f"fam{k}", set()); a, b = len(h & fg), len(h & bg); fo = ((a + .5) / len(fg)) / ((b + .5) / len(bg)); p = fisher_exact([[a, len(fg) - a], [b, len(bg) - b]], alternative="greater")[1]
        val.append((t, k, a, b, round(fo, 3), p))
V = pd.DataFrame(val, columns=["tf", "family", "hits_fg", "hits_bg", "fold", "p"]); V.to_csv(f"{G}/family_val.tsv", sep="\t", index=False); ok = {(t, k) for t, k, fo, p in zip(V.tf, V.family, V.fold, V.p) if fo > 1 and p < .05}
mem = {t: set(F.index[F.tfs.str.split(",").apply(lambda L: t in L)]) for t in tfs}; gen = set(F.index[F.n_tf > MAXTF])
R1 = pd.read_csv(f"{X}/regulatome/r1_tftf/v2/r1v2_pairs_provisional.tsv", sep="\t"); R1 = R1[R1.a.isin(tfs) & R1.b.isin(tfs)].copy()
R1["pct"] = R1[["pearson_domain_pct", "spqn_noquies_pct"]].max(1); R1["pair"] = R1.a + "-" + R1.b; R1["novel"] = R1.string_physical.isna() & ~R1.capselex.astype(str).str.startswith("called")
cells = pd.read_csv(f"{X}/complex_screen/pair_motif_lookup/tf_cells.tsv", sep="\t").set_index("tf").n_cells; tert = np.quantile(cells.values, [1 / 3, 2 / 3]); cb = {t: int(np.searchsorted(tert, cells[t])) for t in tfs}
R1["dbin"] = [tuple(sorted((cb[a], cb[b]))) for a, b in zip(R1.a, R1.b)]; R1["pbin"] = pd.cut(R1.pct, [-1, .6, .8, .9, 1.01], labels=False)
af3 = {}
for f in glob.glob(f"{X}/complex_screen/af3_corr060_20261104/scores_s1/*.tsv"):
    x = open(f).read().split("\t"); a_, b_ = x[0].split("__"); af3[tuple(sorted((a_, b_)))] = (x[4], float(x[1]), float(x[2]))
for f in [f"{X}/complex_screen/af3_novel_20261103/af3_scores_weak.tsv", f"{X}/complex_screen/af3_novel_20261103/af3_scores.tsv", f"{X}/complex_screen/af3_novel_20261103/af3_scores_srcap.tsv"]:
    for l in open(f):
        x = l.rstrip("\n").split("\t")
        if x[0] != "pair": a_, b_ = [y.replace("CBX3", "HP1").replace("CBX1", "HP1").replace("ZUP1", "ZUFSP") for y in x[0].split("__")]; af3[tuple(sorted((a_, b_)))] = (x[9], float(x[1]), float(x[7]))
R1["af3_tier"], R1["af3_iptm"], R1["af3_ipsae"] = zip(*[af3.get(tuple(sorted((a, b))), ("not_run", np.nan, np.nan)) for a, b in zip(R1.a, R1.b)])
def lab(k): return f"{F.cons[k]}({F.n_tf[k]}TF," + ("novel" if F.novel[k] else str(F.guess[k]).split("/")[0]) + ")"
sh = [mem[a] & mem[b] for a, b in zip(R1.a, R1.b)]; hv = [{k for k in s if (a, k) in ok and (b, k) in ok} for s, a, b in zip(sh, R1.a, R1.b)]
R1["n_shared_all"] = [len(s) for s in sh]; R1["n_shared_nongeneric"] = [len(s - gen) for s in sh]; R1["n_heldout_all"] = [len(s) for s in hv]; R1["n_heldout_nongeneric"] = [len(s - gen) for s in hv]; R1["n_heldout_novel_nongeneric"] = [sum(F.novel[k] for k in s - gen) for s in hv]
R1["heldout_nongeneric"] = [";".join(lab(k) for k in sorted(s - gen)) for s in hv]; R1["heldout_generic"] = [";".join(lab(k) for k in sorted(s & gen)) for s in hv]; R1["shared_nongeneric_not_heldout"] = [";".join(lab(k) for k in sorted((s - gen) - v)) for s, v in zip(sh, hv)]
def emp(r):
    pool = R1[(R1.dbin == r.dbin) & (R1.pbin == r.pbin) & (R1.pair != r.pair)]
    if len(pool) < 20: pool = R1[(R1.dbin == r.dbin) & (R1.pair != r.pair)]
    return (1 + int(np.sum(rng.choice(pool.n_heldout_nongeneric.values, 200) >= r.n_heldout_nongeneric))) / 201 if r.n_heldout_nongeneric else 1.0
R1["emp_p"] = [emp(r) for _, r in R1.iterrows()]; cols = ["pair", "novel", "pct", "pearson_domain_pct", "spqn_noquies_pct", "af3_tier", "af3_iptm", "af3_ipsae", "n_shared_all", "n_shared_nongeneric", "n_heldout_all", "n_heldout_nongeneric", "n_heldout_novel_nongeneric", "emp_p", "heldout_nongeneric", "heldout_generic", "shared_nongeneric_not_heldout"]
rk = {"strong": 0, "weak": 1}; R1["_a"] = R1.af3_tier.map(rk).fillna(2); R1 = R1.sort_values(["_a", "emp_p", "pct"], ascending=[True, True, False]); R1[cols].to_csv(f"{G}/shared_motif_all_pairs.tsv", sep="\t", index=False); C = R1[R1.novel & (R1.pct >= .6)]; C[cols].to_csv(f"{G}/shared_motif_candidates.tsv", sep="\t", index=False)
CON = ["CTCF-STAG2", "MAZ-ZNF143", "NFYA-ZNF644", "EZH2-MTF2", "RBBP7-SRCAP"]; Cn = R1[R1.pair.isin(CON) | (R1.b + "-" + R1.a).isin(CON)][cols]; Cn.to_csv(f"{G}/shared_motif_controls.tsv", sep="\t", index=False); nv = R1[R1.novel]; rows = []
for L, s in [("novel pct>=0.90", nv[nv.pct >= .9]), ("novel 0.80-0.90", nv[(nv.pct >= .8) & (nv.pct < .9)]), ("novel 0.60-0.80", nv[(nv.pct >= .6) & (nv.pct < .8)]), ("novel <0.60", nv[nv.pct < .6]), ("candidates AF3 strong", C[C.af3_tier == "strong"]), ("candidates AF3 weak", C[C.af3_tier == "weak"]), ("candidates AF3 none", C[C.af3_tier == "none"]), ("all pairs", R1)]:
    rows.append(dict(set=L, n=len(s), any_shared_nongeneric=round((s.n_shared_nongeneric > 0).mean(), 3), any_heldout_nongeneric=round((s.n_heldout_nongeneric > 0).mean(), 3), any_heldout_novel=round((s.n_heldout_novel_nongeneric > 0).mean(), 3), emp_le05=round((s.emp_p <= .05).mean(), 3)))
Rt = pd.DataFrame(rows); Rt.to_csv(f"{G}/shared_motif_rates.tsv", sep="\t", index=False)
OWN = {"CTCF": r"^CTCF\(", "NFYA": r"^NF[Yy]\(", "NRF1": r"^NRF1?\(", "ZNF143": r"ZNF143", "MAZ": r"^Maz\(", "E2F3": r"^E2F3\(", "SOX4": r"^Sox4\(", "MLX": r"^(Usf|Max|c-Myc)", "RELA": r"^p65", "FOXK1": r"^FOXK1\(", "TBX3": r"^Tbx[35]\(", "HES1": r"^(HEY|Hes)", "MEF2A": r"^Mef2a\(", "SREBF2": r"^SREBF?2\(", "ZBTB7A": r"^ZBTB7A", "ZIC2": r"^Zic", "RBPJ": r"^RBPJ"}; own = []
for t, rx in OWN.items():
    r = dict(tf=t)
    for md in ["known", "val_known"]:
        f = f"{G}/results/{t}/{md}/knownResults.txt"
        if not os.path.exists(f): continue
        k = pd.read_csv(f, sep="\t"); k.columns = ["name", "cons", "p", "logp", "q", "nt", "pt", "nb", "pb"]; k["rank"] = np.arange(1, len(k) + 1); s = k[k.name.str.contains(rx, regex=True)]
        if len(s): b = s.iloc[0]; r[f"{md}_motif"] = b["name"].split("/")[0]; r[f"{md}_rank"] = int(b["rank"]); r[f"{md}_of"] = len(k); r[f"{md}_fold"] = round(float(str(b.pt).rstrip("%")) / max(float(str(b.pb).rstrip("%")), .01), 2); r[f"{md}_log10p"] = round(b.logp / np.log(10), 1)
    own.append(r)
Ow = pd.DataFrame(own); Ow.to_csv(f"{G}/own_motif_check.tsv", sep="\t", index=False); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 140)
print(len(F), "families;", int((F.n_tf <= MAXTF).sum()), "non-generic;", int(F.novel.sum()), "novel"); print(Rt.to_string(index=False)); print(Ow.to_string(index=False)); print(Cn[["pair", "af3_tier", "n_shared_all", "n_shared_nongeneric", "n_heldout_all", "n_heldout_nongeneric", "emp_p", "heldout_nongeneric"]].to_string(index=False))
print(C[C.af3_tier.isin(["strong", "weak"]) & (C.n_heldout_nongeneric > 0)][["pair", "pct", "af3_tier", "af3_ipsae", "n_heldout_nongeneric", "n_heldout_novel_nongeneric", "emp_p", "heldout_nongeneric"]].head(40).to_string(index=False))
