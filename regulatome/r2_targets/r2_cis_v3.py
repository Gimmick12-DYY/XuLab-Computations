# R1/propagation v3: adds the cis-coordinated classes; as v2, plus the qualification rule z > 0 and >= 1 shared edge, because the stratified percentile admitted negative-z pairs. Reads frozen v1 tables and never changes them; opens no knockdown data.
# Candidates: frozen co-localized pairs with Cicero shared-edge excess percentile >= 0.90, >= 1 other view-1 metric >= 0.90, excess z > 0 and >= 1 shared edge; STAG2-wrap and NEUROD4 excluded (pairs already in a w > 0 class keep their class).
# Loop test: shared edges = merged-universe Cicero pairs both TFs carry (the matrix behind cicero_excess_z) against Peakachu 0.7 loops, 5 kb and 10 kb, anchors +/- 1 bin, both ends. Null: 1,000 draws of distance-matched edges (10 log bins, 25 kb to 1 Mb) from the two TFs' own > 25 kb edges.
# Loop-supported: p <= 0.05 and >= 3 looped edges, w 0.5; else cis-coordinated, w 0.25. Low-low pairs enter only as loop-supported. Cap: each TF keeps its 5 best by excess z, greedy down the z ranking, a pair needing room at both TFs.
# Sources: ATAC-filtered specific TF lists, or a module pool when the partner is a pool member and the recipient is not (via_module). Local support: elements carrying >= 50% of S(p,g), largest positive contributions first; recipient Btilde > 0 on >= 1, or >= 2 when the set has >= 2.
# Baselines for the in-recipient-top-1,000 rate: all genes passing the rule against the source list (same recipient), and the same filtered-list test on 5 depth-matched non-partner recipients.
import hashlib, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r2_scores.py")).read().split("y = Y[i];")[0]
sys.argv = ["r2_scores.py", "tf", "MBD3"]; exec(src)
R1 = f"{X}/regulatome/r1_tftf"; OR = f"{R1}/v2_final_v3"; OP = f"{O}/propagation_ciscoord_v3"; LP = f"{X}/hic/work_wt293t/loops"
for d in (OR, OP, f"{OP}/inferred_lists"): os.makedirs(d, exist_ok=True)
F = pd.read_csv(f"{R1}/v2_final/r1_pairs_frozen.tsv", sep="\t", keep_default_na=False, na_values=[""]); M_ = pd.read_csv(f"{R1}/v2/r1v2_pairs_provisional.tsv", sep="\t", usecols=["a", "b", "cicero_excess_z", "pearson_domain_pct", "spqn_noquies_pct", "cicero_excess_z_pct", "compartment_spqn_pct"])
D = F.merge(M_, on=["a", "b"], validate="one_to_one"); oth = D[["pearson_domain_pct", "spqn_noquies_pct", "compartment_spqn_pct"]]
D["cis_prequal"] = D["cis_candidate"] = (D.cicero_excess_z_pct >= 0.90) & (oth >= 0.90).any(axis=1) & ~D["class"].str.startswith("STAG2") & ~(D.a.eq("NEUROD4") | D.b.eq("NEUROD4"))
Sh = pd.read_csv(f"{R1}/inputs/fig_tf_x_shared_pairs_min2_matrix.tsv.gz", sep="\t", index_col=0); Sh.index = Sh.index.str.upper()
def parse(s):
    p = pd.Series(s).str.extract(r"^(\w+):(\d+)-(\d+)$"); return p[0].values, p[1].astype(np.int64).values, p[2].astype(np.int64).values
Lp = []
for res, f in ((5000, "peakachu_5000.loops.0.7.bedpe"), (10000, "peakachu_10000.loops.0.7.bedpe")):
    d = pd.read_csv(f"{LP}/{f}", sep="\t", header=None, usecols=range(6), names=["c1", "s1", "e1", "c2", "s2", "e2"]); d[["s1", "s2"]] -= res; d[["e1", "e2"]] += res; Lp.append(d)
Lp = pd.concat(Lp); LB = {c: g[["s1", "e1", "s2", "e2"]].values for c, g in Lp.groupby("c1")}
def looped(c1, s1, e1, c2, s2, e2):
    """both ends of each edge overlap opposite anchors of one loop"""
    o = np.zeros(len(c1), bool)
    for c in np.unique(c1):
        L = LB.get(c)
        if L is None: continue
        for i in np.flatnonzero(c1 == c):
            a = (L[:, 0] < e1[i]) & (L[:, 1] > s1[i]) & (L[:, 2] < e2[i]) & (L[:, 3] > s2[i]); b = (L[:, 0] < e2[i]) & (L[:, 1] > s2[i]) & (L[:, 2] < e1[i]) & (L[:, 3] > s1[i]); o[i] = a.any() or b.any()
    return o
BINS = np.logspace(np.log10(25000), 6, 11)
def dist_bin(s1, e1, s2, e2): return np.clip(np.digitize(np.abs((s1 + e1) / 2 - (s2 + e2) / 2), BINS) - 1, 0, 9)
EC = {}
def tf_edges(t):
    if t not in EC:
        f = f"{X}/cobinding/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv" if t == "RBBP4" else f"{X}/cobinding/work/ourrun_k50/{t}/{t}.peak_edges.tsv"; C = pd.read_csv(f, sep="\t"); c1, s1, e1 = parse(C.peak1); c2, s2, e2 = parse(C.peak2)
        k = np.abs((s1 + e1) / 2 - (s2 + e2) / 2) > 25000; EC[t] = (dist_bin(s1[k], e1[k], s2[k], e2[k]), looped(c1[k], s1[k], e1[k], c2[k], s2[k], e2[k]))
    return EC[t]
cols = np.array(Sh.columns.tolist()); pa = pd.Series(cols).str.split("|", expand=True); c1, s1, e1 = parse(pa[0]); c2, s2, e2 = parse(pa[1]); SL = looped(c1, s1, e1, c2, s2, e2); SB = dist_bin(s1, e1, s2, e2)
rng = np.random.default_rng(20261102); recs = {}
for r in D[D.cis_candidate].itertuples():
    # a TF absent from the shared-edge matrix has no shared edges: 0 shared, 0 looped, p = 1
    sh = (Sh.loc[r.a].values > 0) & (Sh.loc[r.b].values > 0) if r.a in Sh.index and r.b in Sh.index else np.zeros(len(cols), bool); n = int(sh.sum()); ob = int(SL[sh].sum())
    if not n: recs[(r.a, r.b)] = dict(n_shared_edges=0, n_looped=0, expected_looped=0.0, loop_p=1.0); continue
    ba, la = tf_edges(r.a); bb, lb = tf_edges(r.b); pb = np.concatenate([ba, bb]); pl = np.concatenate([la, lb]); dr = np.zeros(1000)
    for k in range(10):
        m = int((SB[sh] == k).sum())
        if not m: continue
        N = int((pb == k).sum()); K = int(pl[pb == k].sum())
        if N == 0: N, K = len(pb), int(pl.sum())
        dr += rng.hypergeometric(K, N - K, min(m, N), 1000)
    recs[(r.a, r.b)] = dict(n_shared_edges=n, n_looped=ob, expected_looped=round(dr.mean(), 2), loop_p=round((1 + (dr >= ob).sum()) / 1001, 4))
for c in ("n_shared_edges", "n_looped", "expected_looped", "loop_p"): D[c] = [recs.get((a, b), {}).get(c, np.nan) for a, b in zip(D.a, D.b)]
D["cis_candidate"] = D.cis_prequal & (D.cicero_excess_z > 0) & (D.n_shared_edges >= 1)
D["loop_enriched"] = (D.loop_p <= 0.05) & (D.n_looped >= 3); D["class_v3"] = D["class"]; D["w_v3"] = D.w; D["cis_status"] = "."
D.loc[D.cis_prequal & ~D.cis_candidate, "cis_status"] = "excluded: z <= 0 or no shared edges"; coloc = D.cis_candidate & D["class"].eq("co-localized"); D.loc[D.cis_candidate & ~coloc, "cis_status"] = "candidate, keeps existing class"
bg = coloc & ~D.loop_enriched & D.gpair.eq("low-low"); D.loc[bg, "cis_status"] = "stays co-localized: low-low without loop support"; D.loc[bg, "flags"] = ["batch_guard_fail" if f == "." else f + ";batch_guard_fail" for f in D.loc[bg, "flags"]]
elig = D[coloc & (D.loop_enriched | D.gpair.ne("low-low"))].sort_values("cicero_excess_z", ascending=False); cnt = {}
for i, r in elig.iterrows():
    if cnt.get(r.a, 0) >= 5 or cnt.get(r.b, 0) >= 5: D.loc[i, "cis_status"] = "stays co-localized: over the 5-per-TF cap"; continue
    cnt[r.a] = cnt.get(r.a, 0) + 1; cnt[r.b] = cnt.get(r.b, 0) + 1
    if r.loop_enriched: D.loc[i, ["class_v3", "w_v3", "cis_status"]] = ["cis-coordinated (loop-supported)", 0.5, "added"]
    else: D.loc[i, ["class_v3", "w_v3", "cis_status"]] = ["cis-coordinated", 0.25, "added"]
keep = [c for c in F.columns if c not in ("class", "w")]; Ov = D[["a", "b", "class_v3", "w_v3"] + keep[2:] + ["class", "w", "cis_prequal", "cis_candidate", "cis_status", "cicero_excess_z", "n_shared_edges", "n_looped", "expected_looped", "loop_p"]].rename(columns={"class_v3": "class", "w_v3": "w", "class": "class_v1", "w": "w_v1"})
Ov.to_csv(f"{OR}/r1_pairs_frozen_v3.tsv", sep="\t", index=False, float_format="%.4f"); Ov[Ov["class"] != "."].to_csv(f"{OR}/r1_edges_frozen_v3.tsv", sep="\t", index=False, float_format="%.4f")
CC = Ov.groupby("class").agg(n=("a", "size"), n_w_positive=("w", lambda s: int((s > 0).sum())), w_max=("w", "max")).reset_index(); CC.to_csv(f"{OR}/r1_class_counts_frozen_v3.tsv", sep="\t", index=False)
Cand = Ov[Ov.cis_candidate]; Exc = Ov[Ov.cis_prequal & ~Ov.cis_candidate]; Exc[["a", "b", "class", "class_v1", "cis_status", "gpair", "cicero_excess_z", "n_shared_edges", "n_looped", "loop_p", "flags"]].to_csv(f"{OR}/cis_excluded_z_or_no_shared.tsv", sep="\t", index=False, float_format="%.3f"); Cand[["a", "b", "class", "w", "class_v1", "cis_status", "gpair", "cicero_excess_z", "n_shared_edges", "n_looped", "expected_looped", "loop_p", "flags"]].to_csv(f"{OR}/cis_candidates_loop_test.tsv", sep="\t", index=False, float_format="%.3f")
# sources and elements
Tier = pd.read_csv(f"{O}/r2_summary.tsv", sep="\t").set_index("tf").tier; TF_SRC = ["RBBP4", "NFYA", "SOX4", "ZNF507", "MLX", "E4F1", "ZNF777"]; POOL = {"NuRD_MBD3core": ["MBD3"], "WIZ_ZNF644_broad_domain": ["DR1", "THAP6", "CHAMP1", "WIZ", "ZNF644", "SRCAP"], "PRC2_core": ["EZH2", "MTF2", "PCGF6"]}
SRCF = {"ZNF777": "shallow_source", "PRC2_core": "not_PRC2_enriched"}
Fz = pd.read_csv(f"{O}/propagation/propagation_edges.tsv", sep="\t", keep_default_na=False); Fk = set(zip(Fz.target, Fz.source)); New = []
for r in Ov[Ov.cis_status.eq("added")].itertuples():
    for q, p in ((r.a, r.b), (r.b, r.a)):
        if p in TF_SRC: New.append(dict(target=q, source=p, tier="cis-coordinated-loop" if r.w == 0.5 else "cis-coordinated", w=r.w, via_partner=p, via_module=False, flags=r.flags))
        for m, mem in POOL.items():
            if p in mem and q not in mem: New.append(dict(target=q, source=m, tier="cis-coordinated-loop" if r.w == 0.5 else "cis-coordinated", w=r.w, via_partner=p, via_module=True, flags=r.flags))
New = pd.DataFrame(New)
if len(New): New = New[[(t, s) not in Fk for t, s in zip(New.target, New.source)]].sort_values("w", ascending=False); New = New.groupby(["target", "source"], as_index=False).agg(tier=("tier", "first"), w=("w", "first"), via_partner=("via_partner", lambda s: ",".join(sorted(set(s)))), via_module=("via_module", "first"), flags=("flags", "first"))
def glm_b(y, N, pool, poolN, nE_=None):
    pc = pool / poolN * 1e6; Xd = np.column_stack([np.ones(nE), np.log1p(atac_cpm), np.log1p(pc), logw, gc, gc ** 2, mp]); mu = glm_mu(y, np.full(nE, np.log(N / 1e6)), Xd); return np.log2((y + 0.5) / (mu + 0.5))
def tf_b(q): i = tfs.index(q); return glm_b(Y[i], rd[i], Y.sum(0) - Y[i], rd.sum() - rd[i])
def src_contrib(s):
    global t
    mem = POOL.get(s, [s]); ix = [tfs.index(m) for m in mem]; y = Y[ix].sum(0); N = rd[ix].sum(); B = glm_b(y, N, Y.sum(0) - y, rd.sum() - N); Cu = None
    for m in mem:
        t = m; c = cicero_W()
        if c is not None: Cu = c if Cu is None else Cu + c
    Wk = W.copy()
    if Cu is not None: Cu.data[:] = 1.0; Wk = W + W.multiply(Cu)
    return sp.diags(B) @ Wk.tocsr(), np.asarray((Wk.T @ B)).ravel()
gi = {g: k for k, g in enumerate(genes)}; SRC = {}; Bpos = {}
def elem_sets(s):
    if s not in SRC:
        Cm, Sg = src_contrib(s); L_ = pd.read_csv(f"{O}/sources_atacfiltered/{s}.tsv", sep="\t"); Cc = Cm.tocsc(); sets = {}
        chk = np.abs(Sg[[gi[g] for g in L_.gene]] - L_.S_k1.values); assert np.nanmax(chk / np.maximum(np.abs(L_.S_k1.values), 1e-6)) < 1e-2, f"{s} score mismatch with frozen list: {np.nanmax(chk)}"
        for g in L_.gene:
            k = gi[g]; idx = Cc.indices[Cc.indptr[k]:Cc.indptr[k + 1]]; v = Cc.data[Cc.indptr[k]:Cc.indptr[k + 1]]; o = np.argsort(-v); tot = Sg[k]; cum = np.cumsum(v[o]); n_ = int(np.searchsorted(cum, 0.5 * tot) + 1) if tot > 0 else 0; sets[g] = idx[o[:n_]]
        SRC[s] = (L_, sets)
    return SRC[s]
def passes(q, s):
    if q not in Bpos: Bpos[q] = tf_b(q) > 0
    L_, sets = elem_sets(s); b = Bpos[q]; return {g: (b[e].sum() >= (2 if len(e) >= 2 else 1)) for g, e in sets.items() if len(e)}
def top1000(q):
    d = pd.read_csv(f"{O}/per_tf/{q}.tsv.gz", sep="\t", usecols=["gene", "rank_k1"]); return set(d.gene[d.rank_k1 <= 1000])
cells = pd.read_csv(f"{O}/data/tf_cells.tsv", sep="\t") if os.path.exists(f"{O}/data/tf_cells.tsv") else None; rk = pd.Series(rd, index=tfs).rank(); r3 = np.random.default_rng(7); rows, S_rows = [], []
for e in New.itertuples():
    q, s = e.target, e.source; L_, sets = elem_sets(s); ok = passes(q, s); kept = [g for g in L_.gene if ok.get(g, False)]; tp = top1000(q); partners = set(Ov[((Ov.a == q) | (Ov.b == q)) & (Ov.w > 0)][["a", "b"]].values.ravel()) | {q, s}
    near = [x for x in tfs if abs(rk[x] - rk[q]) <= 6 and x not in partners and x != "NEUROD4"]; ctl = list(r3.choice(near, min(5, len(near)), replace=False)) if near else []; cr = []
    for x in ctl:
        okx = passes(x, s); kx = [g for g in L_.gene if okx.get(g, False)]; tx = top1000(x); cr.append(np.mean([g in tx for g in kx]) if kx else np.nan)
    S_rows.append(dict(target=q, tier_recipient=Tier[q], source=s, edge_tier=e.tier, via_module=e.via_module, via_partner=e.via_partner, w=e.w, list_before=len(L_), kept=len(kept), survival=round(len(kept) / len(L_), 3), rate_top1000=round(np.mean([g in tp for g in kept]), 3) if kept else np.nan, rate_source_list_prefilter=round(np.mean([g in tp for g in L_.gene]), 3), rate_matched_recipients=round(np.nanmean(cr), 3) if cr else np.nan, n_matched=len(ctl)))
    rows.append(pd.DataFrame(dict(target=q, source=s, edge_class=e.tier, w=e.w, gene=kept, S_source=L_.set_index("gene").S_k1.reindex(kept).values, locally_supported=True, in_recipient_top1000=[g in tp for g in kept], flags_carried=";".join(x for x in (e.flags, "via_module" if e.via_module else ".", SRCF.get(s, ".")) if x != ".") or ".")))
Sv = pd.DataFrame(S_rows); Sv.to_csv(f"{OP}/cis_edge_summary.tsv", sep="\t", index=False)
# combined table: frozen rows plus the new tiers, w re-normalised per target
Fr = pd.read_csv(f"{O}/propagation/propagation_table.tsv.gz", sep="\t", keep_default_na=False); Fr["S_source"] = Fr.inferred_score / Fr.w_norm; Fr["tier"] = Fr.edge_class; Fr["in_recipient_top1000"] = Fr.locally_supported.astype(str).eq("True")
Nw = pd.concat(rows) if rows else pd.DataFrame(); Nw["tier"] = Nw.edge_class; Nw = Nw.rename(columns={})
Ed = pd.concat([Fz.assign(tier=Fz["class"])[["target", "source", "tier", "w"]], New[["target", "source", "tier", "w"]]], ignore_index=True); Ed["w_norm"] = Ed.w / Ed.groupby("target").w.transform("sum"); Ed.to_csv(f"{OP}/propagation_edges_v3.tsv", sep="\t", index=False, float_format="%.4f")
All = pd.concat([Fr[["target", "source", "edge_class", "w", "gene", "S_source", "locally_supported", "in_recipient_top1000", "flags_carried", "tier"]], Nw[["target", "source", "edge_class", "w", "gene", "S_source", "locally_supported", "in_recipient_top1000", "flags_carried", "tier"]]], ignore_index=True).merge(Ed[["target", "source", "w_norm"]], on=["target", "source"]); All["inferred_score"] = np.round(All.w_norm * All.S_source, 5); All["direct_list_primary"] = All.target.map(Tier).eq("specific"); All.to_csv(f"{OP}/propagation_table_v3.tsv.gz", sep="\t", index=False)
for q, G in All.groupby("target"):
    G.groupby("gene").agg(inferred_score=("inferred_score", "sum"), tiers=("tier", lambda s: ",".join(sorted(set(s)))), sources=("source", lambda s: ",".join(sorted(set(s))))).reset_index().sort_values("inferred_score", ascending=False).to_csv(f"{OP}/inferred_lists/{q}.tsv", sep="\t", index=False, float_format="%.5g")
pd.set_option("display.width", 230); pd.set_option("display.max_rows", 150); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 44)
print(CC.to_string(index=False)); print("batch_guard_fail:", int(Cand["flags"].str.contains("batch_guard_fail").sum()), "| prequalified:", len(Exc) + len(Cand), "excluded by z > 0 / shared edge:", len(Exc), "of which z <= 0:", int((Exc.cicero_excess_z <= 0).sum()), "no shared edges:", int(Exc.n_shared_edges.eq(0).sum())); print(Cand.cis_status.value_counts().to_string()); print(Cand[Cand.cis_status.eq("added")][["a", "b", "class", "gpair", "cicero_excess_z", "n_shared_edges", "n_looped", "expected_looped", "loop_p", "flags"]].round(3).to_string(index=False))
print("loop-enriched among candidates:", int((Cand.loop_p <= 0.05).sum()), "with >= 3 looped:", int(((Cand.loop_p <= 0.05) & (Cand.n_looped >= 3)).sum()), "of", len(Cand)); print(Sv.drop(columns=["via_partner", "list_before", "n_matched"]).sort_values(["via_module", "target"]).to_string(index=False))
if len(Sv):
    g = Sv.groupby("target").kept.sum(); print("TFs gaining via new tiers:", len(g), "median kept per TF:", g.median(), "| TF-source:", sorted(Sv[~Sv.via_module].target.unique()), "| module-source:", sorted(Sv[Sv.via_module].target.unique()), "| sparse:", sorted(set(Sv.target[Sv.tier_recipient.eq("undetermined")])))
    wk = Sv.kept.values; print("survival median", Sv.survival.median(), "weighted rate top1000", round(np.nansum(Sv.rate_top1000 * wk) / wk.sum(), 3), "prefilter same recipient", round(np.nansum(Sv.rate_source_list_prefilter * Sv.list_before) / Sv.list_before.sum(), 3), "matched recipients", round(np.nanmean(Sv.rate_matched_recipients), 3))
fz = set(Fz.target); print("coverage:", len(fz | set(Sv.target)) if len(Sv) else len(fz), "of 79; new TFs:", sorted(set(Sv.target) - fz) if len(Sv) else [])
