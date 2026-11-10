# Track B2-B4: novel co-binding pairs ranked by motifs enriched in BOTH TFs' matched-background HOMER tables (per_tf_matched/).
# Per TF a known motif passes at strict (p <= 1e-5, fold >= 1.5) or loose (p <= 1e-3, fold >= 1.3) on the discovery chromosomes.
# Pair statistics: shared count; hypergeometric P(X >= shared) over the known library given each TF's passing count; empirical p
# against 200 random panel pairs matched on depth (cell-count tertiles of both TFs) and correlation-percentile bin.
# Shared motifs carry: discovery p in each TF, cross-TF percentile (rank of -log10 p among 79 TFs), number of TFs passing, even-chromosome p.
# De novo: each TF's de novo motifs with fold >= 1.5 and p <= 1e-10 are compared between the two TFs (HOMER-style similarity >= 0.6);
# a shared word whose HOMER BestGuess score is < 0.6 in both is marked novel.
# Novel pair = not STRING-physical and not CAP-SELEX called. Tiers: max(pearson_domain_pct, spqn_noquies_pct) >= 0.90 / 0.80 / 0.60.
import os, sys, re, glob, math
import numpy as np, pandas as pd
from scipy.stats import hypergeom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import summarize as S
X = "/work/users/d/y/dyy12/XuLab"; O = os.path.dirname(os.path.abspath(__file__)); G = os.environ.get("TFBASE", f"{O}/per_tf_matched"); rng = np.random.default_rng(20261110); NNULL = 200
LV = {"strict": (1.5, -5), "loose": (1.3, -3)}; CONTROLS = ["CTCF-STAG2", "MAZ-ZNF143", "NFYA-ZNF644", "EZH2-MTF2", "RBBP7-SRCAP"]
tfs = sorted(os.listdir(f"{G}/results")); K, V, DN = {}, {}, {}
for t in tfs:
    k = S.known(f"{G}/results/{t}/known/knownResults.txt")
    if k is None: continue
    k.index = [S.short(n) for n in k.index]; K[t] = k[~k.index.duplicated()]
    v = S.known(f"{G}/results/{t}/val_known/knownResults.txt")
    if v is not None: v.index = [S.short(n) for n in v.index]; V[t] = v[~v.index.duplicated()]
    DN[t] = [m for f in sorted(glob.glob(f"{G}/results/{t}/L*/homerMotifs.all.motifs")) for m in S.denovo(f) if m["fold"] >= 1.5 and m["log10p"] <= -10]
tfs = sorted(K); lib = sorted(set.intersection(*[set(K[t].index) for t in tfs])); NL = len(lib)
P = pd.DataFrame({t: K[t].reindex(lib).log10p for t in tfs}); F = pd.DataFrame({t: K[t].reindex(lib).fold for t in tfs})
pct = (-P).rank(axis=1, pct=True)  # cross-TF percentile of each motif's -log10 p
passing = {lv: (F >= f) & (P <= lp) for lv, (f, lp) in LV.items()}; npass = {lv: m.sum(1) for lv, m in passing.items()}
own = {t: S.own_of(t) for t in tfs}
rows = []
for t in tfs:
    o = [m for m in own[t] if m in K[t].index]
    best = min(o, key=lambda m: K[t].loc[m].log10p) if o else None; rk = int((K[t].log10p.rank(method="min")).loc[best]) if best else None
    rows.append(dict(tf=t, own_motif=best or "none", own_rank=rk, own_log10p=round(K[t].loc[best].log10p, 1) if best else np.nan, own_fold=round(K[t].loc[best].fold, 2) if best else np.nan,
                     own_val_log10p=round(V[t].loc[best].log10p, 1) if best and t in V and best in V[t].index else np.nan, top_motif=K[t].index[0], top_log10p=round(K[t].log10p.iloc[0], 1),
                     n_strict=int(passing["strict"][t].sum()), n_loose=int(passing["loose"][t].sum()), n_denovo=len(DN[t])))
TFT = pd.DataFrame(rows); TFT.to_csv(f"{G}/per_tf_summary.tsv", sep="\t", index=False)
def shared(a, b, lv): return [m for m in lib if passing[lv].loc[m, a] and passing[lv].loc[m, b]]
def dn_shared(a, b):
    out = []
    for x in DN[a]:
        ys = [(S.sim(x["m"], y["m"]), y) for y in DN[b]]
        if not ys: continue
        s, y = max(ys, key=lambda q: q[0])
        if s >= 0.6:
            sc = [float(re.search(r"\(([\d.]+)\)$", z["name"]).group(1)) if re.search(r"\(([\d.]+)\)$", z["name"]) else 0 for z in (x, y)]
            out.append(f"{x['cons']}~{y['cons']}({s:.2f}{',novel' if max(sc) < 0.6 else ',' + x['name'].split('BestGuess:')[-1].split('/')[0]})")
    return out
R1 = pd.read_csv(f"{X}/regulatome/r1_tftf/v2/r1v2_pairs_provisional.tsv", sep="\t"); R1 = R1[R1.a.isin(tfs) & R1.b.isin(tfs)].copy()
R1["pct"] = R1[["pearson_domain_pct", "spqn_noquies_pct"]].max(1); R1["pair"] = R1.a + "-" + R1.b
R1["novel"] = R1.string_physical.isna() & ~R1.capselex.astype(str).str.startswith("called")
cells = pd.read_csv(f"{X}/complex_screen/pair_motif_lookup/tf_cells.tsv", sep="\t").set_index("tf").n_cells
tert = np.quantile(cells.values, [1 / 3, 2 / 3]); cb = {t: int(np.searchsorted(tert, cells.get(t, np.nan))) if t in cells.index else 3 for t in tfs}
R1["dbin"] = [tuple(sorted((cb[a], cb[b]))) for a, b in zip(R1.a, R1.b)]; R1["pbin"] = pd.cut(R1.pct, [-1, .6, .8, .9, 1.01], labels=False)
for lv in LV:
    R1[f"n_shared_{lv}"] = [len(shared(a, b, lv)) for a, b in zip(R1.a, R1.b)]
    R1[f"hyper_p_{lv}"] = [hypergeom.sf(x - 1, NL, int(passing[lv][a].sum()), int(passing[lv][b].sum())) for x, a, b in zip(R1[f"n_shared_{lv}"], R1.a, R1.b)]
def emp(r, lv):
    pool = R1[(R1.dbin == r.dbin) & (R1.pbin == r.pbin) & (R1.pair != r.pair)]
    if len(pool) < 20: pool = R1[(R1.dbin == r.dbin) & (R1.pair != r.pair)]
    nul = rng.choice(pool[f"n_shared_{lv}"].values, NNULL, replace=True); return (1 + np.sum(nul >= r[f"n_shared_{lv}"])) / (1 + NNULL), len(pool)
af3 = {}
def add_af3(f, src):
    for l in open(f):
        x = l.rstrip("\n").split("\t")
        if x[0] == "pair": continue
        a_, b_ = x[0].split("__"); key = tuple(sorted((a_.replace("CBX3", "HP1").replace("CBX1", "HP1").replace("ZUP1", "ZUFSP"), b_.replace("CBX3", "HP1").replace("CBX1", "HP1").replace("ZUP1", "ZUFSP"))))
        rec = dict(af3_tier=x[4] if src == "corr060_s1" else x[9], af3_iptm=float(x[1]), af3_ipsae=float(x[2]) if src == "corr060_s1" else float(x[7]), af3_source=src)
        rk = {"strong": 2, "weak": 1, "none": 0}
        if key not in af3 or rk.get(rec["af3_tier"], 0) > rk.get(af3[key]["af3_tier"], 0): af3[key] = rec
for f in glob.glob(f"{X}/complex_screen/af3_corr060_20261104/scores_s1/*.tsv"): add_af3(f, "corr060_s1")
add_af3(f"{X}/complex_screen/af3_novel_20261103/af3_scores_weak.tsv", "novel_1seed"); add_af3(f"{X}/complex_screen/af3_novel_20261103/af3_scores.tsv", "novel_5seed")
A_ = {}
for fn in [f"{O}/gc_matched/gate_levels.tsv", f"{O}/triple_support/gate_levels.tsv"]:
    if os.path.exists(fn):
        for _, r in pd.read_csv(fn, sep="\t").iterrows():
            A_[tuple(sorted(r.pair.split("-")))] = dict(cobound_ownA=f"{r.own_A}:{r.A_fold}", cobound_ownB=f"{r.own_B}:{r.B_fold}", cobound_strict=r.strict_pass, cobound_relaxed=r.relaxed_pass, cobound_composite_validated=r.composite_n_validated)
def fmt(a, b, ms):
    return "; ".join(f"{m} [A p=1e{P.loc[m, a]:.1f} pct={pct.loc[m, a]:.2f} val=1e{(V[a].log10p.get(m, np.nan) if a in V else np.nan):.1f} | B p=1e{P.loc[m, b]:.1f} pct={pct.loc[m, b]:.2f} val=1e{(V[b].log10p.get(m, np.nan) if b in V else np.nan):.1f} | nTF={int(npass['strict'][m])}]" for m in ms)
def build(sel):
    out = []
    for _, r in sel.iterrows():
        a, b = r.a, r.b; k = tuple(sorted((a, b))); e_s, npool = emp(r, "strict"); e_l, _ = emp(r, "loose")
        ss, sl = shared(a, b, "strict"), shared(a, b, "loose")
        val_both = [m for m in sl if a in V and b in V and V[a].log10p.get(m, 0) <= -2 and V[b].log10p.get(m, 0) <= -2]
        d = dict(pair=r.pair, pct=round(r.pct, 3), pearson_domain_pct=r.pearson_domain_pct, spqn_noquies_pct=r.spqn_noquies_pct, novel=r.novel, **af3.get(k, dict(af3_tier="not_run", af3_iptm=np.nan, af3_ipsae=np.nan, af3_source="")),
                 n_strict_A=int(passing["strict"][a].sum()), n_strict_B=int(passing["strict"][b].sum()), n_shared_strict=len(ss), n_shared_loose=len(sl), hyper_p_strict=r.hyper_p_strict, hyper_p_loose=r.hyper_p_loose,
                 emp_p_strict=e_s, emp_p_loose=e_l, null_pool=npool, shared_loose_val_even_both=";".join(val_both), shared_strict=fmt(a, b, ss), shared_loose_only=fmt(a, b, [m for m in sl if m not in ss]),
                 denovo_shared=" ; ".join(dn_shared(a, b)[:5]), **A_.get(k, {}))
        out.append(d)
    return pd.DataFrame(out)
nov = R1[R1.novel & (R1.pct >= 0.60)].copy(); nov["tier"] = np.select([nov.pct >= .9, nov.pct >= .8], ["pct>=0.90", "pct>=0.80"], "pct>=0.60")
T = build(nov); T["corr_tier"] = nov.tier.values
rk = {"strong": 0, "weak": 1}; T["_a"] = T.af3_tier.map(rk).fillna(2); T = T.sort_values(["_a", "emp_p_strict", "emp_p_loose", "pct"], ascending=[True, True, True, False]).drop(columns="_a")
T.to_csv(f"{G}/novel_pairs_ranked.tsv", sep="\t", index=False)
C = R1[R1.pair.isin(CONTROLS) | (R1.b + "-" + R1.a).isin(CONTROLS)]; CT = build(C); CT.to_csv(f"{G}/controls.tsv", sep="\t", index=False)
print("library", NL, "motifs; TFs", len(tfs)); print(TFT.to_string())
print("novel pairs per tier:", nov.tier.value_counts().to_dict())
for tr in ["pct>=0.90", "pct>=0.80", "pct>=0.60"]:
    s = T[T.corr_tier == tr]; print(tr, len(s), "emp_p_strict<=0.05:", int((s.emp_p_strict <= .05).sum()), "emp_p_loose<=0.05:", int((s.emp_p_loose <= .05).sum()), "with shared loose motif validated on even:", int((s.shared_loose_val_even_both != "").sum()))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(CT[["pair", "pct", "af3_tier", "n_shared_strict", "n_shared_loose", "hyper_p_strict", "emp_p_strict", "emp_p_loose", "shared_strict"]].to_string())
print(T.head(25)[["pair", "corr_tier", "pct", "af3_tier", "af3_ipsae", "n_shared_strict", "n_shared_loose", "hyper_p_strict", "emp_p_strict", "emp_p_loose", "shared_loose_val_even_both"]].to_string())
