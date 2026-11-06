# Pair-level shared motif calls from SEA (known, q < 0.05) and de novo Tomtom matches (q < 0.05), all panel pairs; background rates with and without generic motifs; own-motif recovery.
import itertools, json, re
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/motif_shared_20261103"; tfs = [t.strip() for t in open(f"{O}/tfs.txt")]; Own = json.load(open(f"{O}/db/own.json"))
Hs = {}; A = {}
for t in tfs:
    d = pd.read_csv(f"{O}/sea/{t}/sea.tsv", sep="\t", comment="#"); d = d[d.ID.notna()]; d["ALT_ID"] = d.ALT_ID.fillna(d.ID).astype(str); A[t] = d; Hs[t] = d[(d.QVALUE < .05) & (d.ENR_RATIO > 1)]
name = {r.ID: r.ALT_ID for d in A.values() for r in d.itertuples()}; ids = sorted(name); TT = pd.read_csv(f"{O}/tomtom_db/tomtom.tsv", sep="\t", comment="#"); TT = TT[TT["q-value"] < .05]; sim = {}
for r in TT.itertuples(): sim.setdefault(r.Query_ID, set()).add(r.Target_ID); sim.setdefault(r.Target_ID, set()).add(r.Query_ID)
GEN = re.compile(r"^(SP\d|KLF\d|EGR\d|WT1|ZBTB7|PATZ|NRF1|ETS\d|ELK\d|ELF\d|ETV\d|ERG|FLI1|FEV|GABPA|ERF|SPI|SPDEF|EHF|MBD3)", re.I); seed = {m for m in ids if GEN.match(name[m].replace("x_", ""))}; gen = set(seed)
for m in seed: gen |= sim.get(m, set()) & set(ids)
ub = pd.Series({m: sum(m in set(Hs[t].ID) for t in tfs) / 79 for m in ids}); UB = set(ub.index[ub >= .5]); print("known motifs", len(ids), "significant per TF median", np.median([len(Hs[t]) for t in tfs]), "generic set", len(gen), "ubiquitous >=50% TFs", len(UB), [name[m] for m in ub.sort_values(ascending=False).index[:15]])
Q = {t: dict(zip(Hs[t].ID, Hs[t].QVALUE)) for t in tfs}; RK = {t: dict(zip(A[t].ID, A[t].RANK)) for t in tfs}
def known(a, b, drop=frozenset()):
    best = None
    for m, qa in Q[a].items():
        if m in drop: continue
        for k in {m} | (sim.get(m, set()) & set(Q[b])):
            if k in Q[b] and k not in drop:
                s = max(qa, Q[b][k])
                if best is None or s < best[0]: best = (s, m, k, qa, Q[b][k])
    return best
Dn = pd.read_csv(f"{O}/tomtom_denovo2/tomtom.tsv", sep="\t", comment="#"); Dn = Dn[Dn["q-value"] < .05]; Dj = pd.read_csv(f"{O}/tomtom_denovo2_db/tomtom.tsv", sep="\t", comment="#"); Dj = Dj[Dj["q-value"] < .05]; dg = set(Dj.Query_ID[Dj.Target_ID.isin(gen)]); bj = Dj.sort_values("q-value").groupby("Query_ID").first(); dn = {}
for r in Dn.itertuples():
    q, t = r.Query_ID.split("|"), r.Target_ID.split("|")
    if q[1] == t[1]: continue
    k = tuple(sorted((q[1], t[1]))); v = (r._6 if False else r[6], r.Query_ID, r.Target_ID); g = r.Query_ID in dg or r.Target_ID in dg; dn.setdefault(k, []).append((float(r[6]), r.Query_ID, r.Target_ID, g))
rows = []
for a, b in itertools.combinations(tfs, 2):
    ka = known(a, b); kn = known(a, b, gen); kt = known(a, b, gen | UB); d = sorted(dn.get((a, b), [])); dng = [x for x in d if not x[3]]; dnew = [x for x in d if x[1].startswith("new|") and x[2].startswith("new|")]; r = dict(a=a, b=b, known_any=ka is not None, known_nongeneric=kn is not None, known_specific=kt is not None, denovo_any=len(d) > 0, denovo_new=len(dnew) > 0, denovo_nongeneric=len(dng) > 0)
    if ka: r.update(known_motif=f"{name[ka[1]]}/{name[ka[2]]}", known_q=ka[0])
    if kn: r.update(ng_motif=f"{name[kn[1]]}/{name[kn[2]]}", ng_q=kn[0], ng_qa=kn[3], ng_qb=kn[4], ng_ubiq=round(ub[kn[1]], 2))
    if kt: r.update(sp_motif=f"{name[kt[1]]}/{name[kt[2]]}", sp_q=kt[0], sp_ubiq=round(ub[kt[1]], 2))
    if d: x = d[0]; r.update(dn_q=x[0], dn_a=x[1], dn_b=x[2], dn_match=name.get(bj.Target_ID.get(x[1]), "."), dn_generic=x[3])
    if dng: x = dng[0]; r.update(dng_q=x[0], dng_a=x[1], dng_b=x[2], dng_match=name.get(bj.Target_ID.get(x[1]), "."))
    rows.append(r)
P = pd.DataFrame(rows); P.to_csv(f"{O}/pairs_all.tsv", sep="\t", index=False, float_format="%.3g"); P["any"] = P.known_any | P.denovo_any; P["ng"] = P.known_nongeneric | P.denovo_nongeneric; P["sp"] = P.known_specific | P.denovo_nongeneric; R = dict(n_pairs=len(P), any=round(P["any"].mean(), 3), known_any=round(P.known_any.mean(), 3), denovo_any=round(P.denovo_any.mean(), 3), denovo_new_only=round(P.denovo_new.mean(), 3), without_generic=round(P.ng.mean(), 3), known_without_generic=round(P.known_nongeneric.mean(), 3), denovo_without_generic=round(P.denovo_nongeneric.mean(), 3), known_without_generic_or_ubiquitous=round(P.known_specific.mean(), 3)); print(R); json.dump(R, open(f"{O}/background_rates.json", "w"), indent=1)
Rc = []
for t in tfs:
    o = Own["own"].get(t.upper(), []); sd = A[t][A[t].ID.isin(o)]; bq = sd.QVALUE.min() if len(sd) else np.nan; dd = Dj[Dj.Query_ID.str.startswith(f"new|{t}|") & Dj.Target_ID.isin(o)]; dq = dd["q-value"].min() if len(dd) else np.nan; Rc.append(dict(tf=t, own_source=Own["src"].get(t.upper()), own_ids=",".join(o), sea_best_q=bq, sea_best_rank=int(sd.RANK.min()) if len(sd) else -1, sea_recovered=bool(bq < .05), denovo_q=dq, denovo_recovered=bool(dq < .05), n_sig_known=len(Hs[t]), top_known=name.get(A[t].ID.iloc[0]) if len(A[t]) else ".", top_nongeneric=next((name[m] for m in A[t].ID if m not in gen), ".")))
Rc = pd.DataFrame(Rc); Rc.to_csv(f"{O}/own_motif_recovery.tsv", sep="\t", index=False, float_format="%.3g"); W = Rc[Rc.own_ids != ""]; print("own motif testable", len(W), "SEA recovered", int(W.sea_recovered.sum()), "de novo recovered", int(W.denovo_recovered.sum())); pd.set_option("display.width", 240); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 44); print(Rc[Rc.tf.isin(["ctcf", "nrf1", "nfya", "znf143", "maz"])].to_string(index=False))
TAB = ["mbd3|rbbp4", "mbd3|rbbp7", "wiz|znf644", "rbbp4|znf512b", "dnmt1|hp1", "dnmt1|ezh2", "ezh2|mtf2", "mef2a|mef2d", "maz|znf143", "hp1|pcgf6", "adnp|hp1", "ctcf|stag2", "rbbp4|smarca1", "hoxd10|srebf2", "foxk1|mlx", "ddit3|rbbp4", "mlx|thap6", "ddit3|znf367", "hoxb8|hp1", "thap6|znf746", "hp1|znf143", "hes1|znf367", "mlx|srebf2", "hoxa5|znf581", "hoxc9|rbbp4", "rbbp7|srcap"]; idx = P.set_index(["a", "b"]); keep = []
for k in TAB:
    a, b = sorted(k.split("|")); r = idx.loc[(a, b)]; keep.append(dict(pair=f"{a}-{b}", known_any=r.known_any, known_motif=r.get("known_motif"), known_q=r.get("known_q"), nongeneric=r.known_nongeneric, ng_motif=r.get("ng_motif"), ng_q=r.get("ng_q"), ng_ubiq=r.get("ng_ubiq"), specific=r.known_specific, sp_motif=r.get("sp_motif"), sp_q=r.get("sp_q"), denovo_any=r.denovo_any, dn_q=r.get("dn_q"), dn_match=r.get("dn_match"), dn_generic=r.get("dn_generic"), denovo_ng=r.denovo_nongeneric, dng_q=r.get("dng_q"), dng_match=r.get("dng_match"), dng_a=r.get("dng_a"), dng_b=r.get("dng_b")))
Tb = pd.DataFrame(keep); Tb.to_csv(f"{O}/pairs_table.tsv", sep="\t", index=False, float_format="%.3g"); print(Tb.drop(columns=["dng_a", "dng_b"]).to_string(index=False)); print(Tb[["pair", "dng_a", "dng_b"]].dropna().to_string(index=False))
for t in ("mlx", "srebf2"):
    d = A[t][A[t].ID.isin(["MA0663.1", "MA0828.3", "MA0596.1"])]; print(t, d[["ID", "ALT_ID", "RANK", "TP%", "FP%", "ENR_RATIO", "QVALUE"]].to_string(index=False)); print(Dj[Dj.Query_ID.str.contains(rf"\|{t}\|") & Dj.Target_ID.isin(["MA0663.1", "MA0828.3", "MA0596.1"])][["Query_ID", "Target_ID", "q-value"]].to_string(index=False)); print(A[t].head(5)[["ID", "ALT_ID", "ENR_RATIO", "QVALUE"]].to_string(index=False))
