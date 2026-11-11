# Motif validation (Fig 4c), scoring + calibration + candidates. Criteria fixed before results (2026-11-11, agreed with Yuyang):
# Methods: fisher = FIMO presence log-odds z (site vs matched background; baseline); panel = the same log-odds standardized per motif across the
# 79 TFs (OLS on log cells removed, empirical-Bayes z = resid / sqrt(tau^2 + se^2)); centrimo = central enrichment at summits (-ln adj p);
# centrimo_panel = CentriMo score standardized across TFs the same way.
# Gate 1 (own motif): own motif rank <= 10 on discovery (odd chr) AND held-out (even + X) chromosomes; a method must beat the HOMER baseline (3 TFs).
# Gate 2 (anchors): A's best own motif chosen on discovery; statistic = its score at B's sites; pass = held-out percentile of the real B among all
# 78 other TFs >= 0.95. Positives CTCF->STAG2, MAZ<->ZNF143 (either direction), NFYA->ZNF644: >= 2 of 3 must pass. Negatives CTCF->NFYA,
# NFYA->CTCF must not pass. A method is usable only if it passes both gates.
# Candidates (candidates_frozen_v1): anchor percentiles for every pair with an anchor TF (own motif strict pass in that method), and the top
# shared motifs by min(panel z_A, panel z_B), Codebook endogenous / artifact motifs flagged.
# Output motif_valid/: scores_<method>_<part>.tsv, own_gate.tsv, calibration.tsv, anchor_controls.tsv, candidates_anchor.tsv, candidates_shared.tsv
import os, re, gzip, glob, json
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = f"{O}/motif_valid"
cells = pd.read_csv(f"{X}/complex_screen/pair_motif_lookup/tf_cells.tsv", sep="\t").set_index("tf").n_cells
alt = {}
for l in open(f"{O}/db/known.meme"):
    if l.startswith("MOTIF "): x = l.split(); alt[x[1]] = x[2] if len(x) > 2 else x[1]
motifs = list(alt); mi = {m: k for k, m in enumerate(motifs)}; M = len(motifs)
ALIAS = {"HP1": ["CBX3", "CBX1", "CBX5"], "ZUFSP": ["ZUP1"]}
def toks(s): return set(re.split(r"[.@:|/_\- ]+", s.upper()))
def own_ids(t): names = ALIAS.get(t, [t]); return [k for k, m in enumerate(motifs) if any(n in toks(m) | toks(alt[m]) for n in names)]
cb = {}
for l in open(f"{X}/motif_analysis/cache/motifdb/codebook_artifacts.meme"):
    if l.startswith("MOTIF "): cb[l.split()[1]] = l.split()[1]
S = pd.read_csv(f"{G}/sites.tsv", sep="\t"); hit = {}
for f in sorted(glob.glob(f"{G}/fimo/c*.tsv.gz")):
    for l in gzip.open(f, "rt"): m, s = l.rstrip("\n").split("\t"); hit.setdefault(s, []).append(mi[m])
tfs = sorted(S.tf.unique()); P = ["disc", "val"]; LO, SE = {}, {}
for pt in P:
    lo = np.full((M, len(tfs)), np.nan); se = lo.copy()
    for j, t in enumerate(tfs):
        g = S[(S.tf == t) & (S.part == pt)]; fg, bg = g.key[g.set == "fg"], g.key[g.set == "bg"]; nf, nb = len(fg), len(bg)
        if nf < 100 or nb < 100: continue
        a = np.zeros(M); b = np.zeros(M)
        for k in fg: a[hit.get(k, [])] += 1
        for k in bg: b[hit.get(k, [])] += 1
        lo[:, j] = np.log((a + .5) / (nf - a + .5)) - np.log((b + .5) / (nb - b + .5)); se[:, j] = np.sqrt(1 / (a + .5) + 1 / (nf - a + .5) + 1 / (b + .5) + 1 / (nb - b + .5))
    LO[pt], SE[pt] = lo, se
def panel(V, E2):  # V, E2: motifs x TFs (value, sampling variance); depth trend removed per motif, empirical-Bayes z
    x = np.log(cells[tfs].values.astype(float)); Z = np.full_like(V, np.nan)
    for i in range(V.shape[0]):
        ok = np.isfinite(V[i]) & np.isfinite(E2[i])
        if ok.sum() < 20: continue
        A = np.c_[np.ones(ok.sum()), x[ok]]; beta = np.linalg.lstsq(A, V[i, ok], rcond=None)[0]; r = V[i, ok] - A @ beta
        tau2 = max(r.var() - E2[i, ok].mean(), 0.01 * r.var()); Z[i, ok] = r / np.sqrt(tau2 + E2[i, ok])
    return Z
SC = {}
for pt in P:
    SC[("fisher", pt)] = LO[pt] / SE[pt]; SC[("panel", pt)] = panel(LO[pt], SE[pt] ** 2)
    cm = np.full((M, len(tfs)), np.nan)
    for j, t in enumerate(tfs):
        f = f"{G}/cm/{t}_{pt}.tsv"
        if not os.path.exists(f) or os.path.getsize(f) == 0: continue
        c = pd.read_csv(f, sep="\t", comment="#", header=None)
        hd = [l for l in open(f) if l.startswith("#")]; cols = hd[-1].lstrip("# ").split() if hd else None
        if cols and len(cols) == c.shape[1]: c.columns = cols
        else: c.columns = ["db_index", "motif_id", "motif_alt_id", "consensus", "E-value", "adj_p-value", "log_adj_p-value", "bin_location", "bin_width", "total_width", "sites_in_bin", "total_sites", "p_success", "p-value", "mult_tests"][:c.shape[1]]
        for m, v in zip(c.motif_id, c["log_adj_p-value"]):
            if m in mi: cm[mi[m], j] = -float(v)
        cm[np.isnan(cm[:, j]), j] = 0.0
    SC[("centrimo", pt)] = cm
    # sampling variance of the CentriMo score is unknown: use the across-TF residual variance only (tau2 path with E2 = 0)
    SC[("centrimo_panel", pt)] = panel(cm, np.zeros_like(cm))
METH = ["fisher", "panel", "centrimo", "centrimo_panel"]
for (m, pt), V in SC.items(): pd.DataFrame(V, index=motifs, columns=tfs).round(3).to_csv(f"{G}/scores_{m}_{pt}.tsv", sep="\t")
RK = {k: pd.DataFrame(V, index=motifs, columns=tfs).rank(ascending=False, method="min") for k, V in SC.items()}
own = []; best = {}
for t in tfs:
    ids = own_ids(t)
    if not ids: continue
    for m in METH:
        rd = RK[(m, "disc")][t].iloc[ids]; rv = RK[(m, "val")][t].iloc[ids]; k = int(np.nanargmin(rd.values)) if rd.notna().any() else 0
        best[(m, t)] = ids[k]; own.append(dict(tf=t, method=m, n_own_motifs=len(ids), best_own=motifs[ids[k]], rank_disc=rd.min(), rank_val=rv.min(), rank_val_of_disc_choice=rv.iloc[k]))
OW = pd.DataFrame(own); OW["strict"] = (OW.rank_disc <= 10) & (OW.rank_val <= 10); OW["loose"] = (OW.rank_disc <= 25) & (OW.rank_val <= 25); OW.to_csv(f"{G}/own_gate.tsv", sep="\t", index=False)
def anchor(m, A, B, pt):
    if (m, A) not in best: return np.nan
    v = SC[(m, pt)][best[(m, A)]]; j = tfs.index(B); o = [i for i, t in enumerate(tfs) if t != A and np.isfinite(v[i])]
    return float(np.mean(v[o] < v[j]) + .5 * np.mean(v[o] == v[j])) if np.isfinite(v[j]) else np.nan
POS = [("CTCF", "STAG2"), ("MAZ", "ZNF143"), ("ZNF143", "MAZ"), ("NFYA", "ZNF644")]; NEG = [("CTCF", "NFYA"), ("NFYA", "CTCF")]; AC = []; CAL = []
for m in METH:
    for A, B in POS + NEG: AC.append(dict(method=m, anchor=A, partner=B, kind="positive" if (A, B) in POS else "negative", pct_disc=anchor(m, A, B, "disc"), pct_val=anchor(m, A, B, "val")))
    a = pd.DataFrame([r for r in AC if r["method"] == m]); ps = lambda A, B: bool((a[(a.anchor == A) & (a.partner == B)].pct_val >= .95).any())
    npos = int(ps("CTCF", "STAG2")) + int(ps("MAZ", "ZNF143") or ps("ZNF143", "MAZ")) + int(ps("NFYA", "ZNF644")); nneg = int(ps("CTCF", "NFYA")) + int(ps("NFYA", "CTCF"))
    o = OW[OW.method == m]; CAL.append(dict(method=m, n_tf_with_own=len(o), own_strict=int(o.strict.sum()), own_loose=int(o.loose.sum()), own_strict_tfs=",".join(o.tf[o.strict]),
                                          positives_passed=npos, negatives_passed=nneg, gate1=int(o.strict.sum()) > 3, gate2=(npos >= 2) and (nneg == 0)))
CA = pd.DataFrame(CAL); CA["usable"] = CA.gate1 & CA.gate2; CA.to_csv(f"{G}/calibration.tsv", sep="\t", index=False); AD = pd.DataFrame(AC); AD.to_csv(f"{G}/anchor_controls.tsv", sep="\t", index=False)
FZ = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_frozen_v1.tsv", sep="\t"); FZ = FZ[FZ.candidate.astype(str) == "True"]; rows = []
for m in METH:
    anc = set(OW.tf[(OW.method == m) & OW.strict])
    for r in FZ.itertuples():
        for A, B in [(r.tf_a, r.tf_b), (r.tf_b, r.tf_a)]:
            if A in anc and B in tfs: rows.append(dict(method=m, pair=r.pair, anchor=A, partner=B, anchor_motif=motifs[best[(m, A)]], pct_disc=anchor(m, A, B, "disc"), pct_val=anchor(m, A, B, "val"), best_pct=r.best_pct, af3=r.af3_5seed_tier if r.af3_5seed_tier != "not_run" else r.af3_1seed_tier))
CN = pd.DataFrame(rows); CN.to_csv(f"{G}/candidates_anchor.tsv", sep="\t", index=False)
# shared motifs by panel z (discovery choose, held-out report)
Zd, Zv = SC[("panel", "disc")], SC[("panel", "val")]; flag = lambda mm: "codebook_artifact_list" if any(mm == c or alt[mm] == c for c in cb) else ""
sh = []; allpairs = [(a, b) for i, a in enumerate(tfs) for b in tfs[i + 1:]]
mind = {p: np.nanmax(np.fmin(Zd[:, tfs.index(p[0])], Zd[:, tfs.index(p[1])])) for p in allpairs}; dist = np.array(list(mind.values()))
for r in FZ.itertuples():
    if r.tf_a not in tfs or r.tf_b not in tfs: continue
    ja, jb = tfs.index(r.tf_a), tfs.index(r.tf_b); md = np.fmin(Zd[:, ja], Zd[:, jb]); o = np.argsort(-np.nan_to_num(md, nan=-99))[:5]
    pair_best = np.nanmax(md); sh.append(dict(pair=r.pair, best_pct=r.best_pct, af3=r.af3_5seed_tier if r.af3_5seed_tier != "not_run" else r.af3_1seed_tier, best_min_z_disc=round(pair_best, 2), panel_pct=round(float(np.mean(dist < pair_best)), 3),
        top5="; ".join(f"{motifs[k]}|{alt[motifs[k]]} disc {md[k]:.2f} val {min(Zv[k, ja], Zv[k, jb]):.2f}{' [' + flag(motifs[k]) + ']' if flag(motifs[k]) else ''}" for k in o)))
SH = pd.DataFrame(sh).sort_values("panel_pct", ascending=False); SH.to_csv(f"{G}/candidates_shared.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print(CA.to_string(index=False)); print(AD.round(3).to_string(index=False))
print(OW[OW.tf.isin(["CTCF", "NFYA", "NRF1", "MAZ", "ZNF143", "MLX", "ZIC2", "FOXK1", "SOX4", "E2F3", "ZBTB7A", "ZNF282", "ZNF768", "ZNF777", "ZNF507"])].pivot(index="tf", columns="method", values="rank_disc").astype("Int64").astype(str)
      + "/" + OW[OW.tf.isin(["CTCF", "NFYA", "NRF1", "MAZ", "ZNF143", "MLX", "ZIC2", "FOXK1", "SOX4", "E2F3", "ZBTB7A", "ZNF282", "ZNF768", "ZNF777", "ZNF507"])].pivot(index="tf", columns="method", values="rank_val").astype("Int64").astype(str))
