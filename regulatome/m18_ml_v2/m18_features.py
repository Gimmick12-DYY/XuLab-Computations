# m18_ml_v2 feature build (SPEC + Amendments 1-2). Writes features/gene_<block>.npz (gene-level: A, R, C-free gene parts, F gene-level) and
# features/pair_<variant>.npz (per (TF, gene) rows for all TFs with a knockdown: C, B, BC, D, F per-TF, G, E) for variants real, swap, pseudo0-2.
# No target is read here except the gene and TF lists. Percentiles are genome-wide (target-free).
import json, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m18_ml_v2"; F = f"{O}/features"; os.makedirs(F, exist_ok=True)
V = "/vast/som/xujie_lab/TAGATG-293T"; QMIN = -np.log10(0.05); SEED = 20261112
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]; ti79 = {t: i for i, t in enumerate(tfs79)}
Rz = np.load(f"{RG}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True); TT = [str(t) for t in Rz["tfs"]]; ug = Rz["genes"].astype(str); NG = len(ug); gi = {g: k for k, g in enumerate(ug)}
EL = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/EnhancerList.txt", sep="\t"); acol = [c for c in EL.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]
EL = EL[["chr", "start", "end", acol, "H3K27ac.RPM"]].rename(columns={acol: "atac", "H3K27ac.RPM": "k27"}).reset_index(drop=True); NE = len(EL); EL["mid"] = (EL.start + EL.end) // 2
ek = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(EL.chr, EL.start, EL.end))}
L = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "TargetGeneTSS", "ABC.Score", "isSelfPromoter"])
L = L[L.TargetGene.isin(gi)].copy(); L["e"] = (L.chr + ":" + L.start.astype(str) + "-" + L.end.astype(str)).map(ek); assert L.e.notna().all(); L["e"] = L.e.astype(int); L["g"] = L.TargetGene.map(gi)
L["self"] = L.isSelfPromoter.astype(str).str.lower().eq("true"); L["abc"] = L["ABC.Score"].astype(float); L["dist"] = (EL.mid.values[L.e.values] - L.TargetGeneTSS.values).astype(float)
Le, Lg, La, Ls, Ld = L.e.values, L.g.values, L.abc.values, L.self.values, L.dist.values; dmask = ~Ls
GL = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t", usecols=["chr", "name", "tss"]).drop_duplicates("name", keep="last").set_index("name").reindex(ug)
def gsum(v, m=None):
    m = np.ones(len(Le), bool) if m is None else m; return np.bincount(Lg[m], weights=v[m], minlength=NG)
def gmax(v, m=None):
    m = np.ones(len(Le), bool) if m is None else m; out = np.zeros(NG); np.maximum.at(out, Lg[m], v[m]); return out
SAVE = lambda name, M, names: np.savez_compressed(f"{F}/{name}.npz", X=np.asarray(M, np.float32), names=np.array(names))
# ---------------- A: gene baseline
A, An = [], []
def addA(v, n): A.append(np.asarray(v, float)); An.append(n)
addA(gsum(EL.atac.values[Le], Ls), "A_prom_atac"); addA(gsum(EL.k27.values[Le], Ls), "A_prom_h3k27ac")
GW = pd.read_csv(f"{RG}/m7_step2/splits/gene_split_wide.tsv", sep="\t"); BW = np.load(f"{RG}/m7_step2/features/raw_hicfixed_wide/base.npz"); addA((pd.Series(BW["base"][:, 4], index=GW.gene).reindex(ug).fillna(0).values > 0).astype(float), "A_cgi")
addA(np.bincount(Lg, minlength=NG), "A_n_links"); addA(gsum(La), "A_abc_sum"); addA(gsum(La, dmask), "A_abc_distal_sum")
addA(gsum(EL.k27.values[Le], dmask), "A_distal_h3k27ac_sum"); addA(gsum(EL.k27.values[Le] * La, dmask), "A_distal_h3k27ac_abc")
CH = pd.read_csv(f"{X}/data/HEK293T_chromHMM18.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 3], names=["c", "s", "e", "st"]); states = sorted(CH.st.unique()); stv = np.array([""] * NG, dtype=object)
for c, H in CH.groupby("c"):
    ix = np.flatnonzero(GL.chr.values == c)
    if not len(ix): continue
    H = H.sort_values("s"); k = np.searchsorted(H.s.values, GL.tss.values[ix].astype(int), side="right") - 1; ok = (k >= 0) & (GL.tss.values[ix] < H.e.values[np.maximum(k, 0)]); stv[ix[ok]] = H.st.values[k[ok]]
for s in states: addA((stv == s).astype(float), f"A_chmm_{s}")
anc = []
for res in (5000, 10000):
    b = pd.read_csv(f"{X}/hic/work_wt293t/loops/peakachu_{res}.loops.0.7.bedpe", sep="\t", header=None, usecols=[0, 1, 2, 3, 4, 5]); anc += [b[[0, 1, 2]].set_axis(["c", "s", "e"], axis=1), b[[3, 4, 5]].set_axis(["c", "s", "e"], axis=1)]
anc = pd.concat(anc).drop_duplicates(); la = np.zeros(NG)
for c, H in anc.groupby("c"):
    ix = np.flatnonzero(GL.chr.values == c); H = H.sort_values("s"); t = GL.tss.values[ix].astype(int); k = np.searchsorted(H.s.values, t, side="right") - 1
    em = np.maximum.accumulate(H.e.values); la[ix] = ((k >= 0) & (t < em[np.maximum(k, 0)])).astype(float)
addA(la, "A_loop_anchor"); SAVE("gene_A", np.column_stack(A), An); print("A", len(An), "chromHMM states", len(states), "loop-anchor genes", int(la.sum()), flush=True)
# ---------------- binding inputs (real): element reads CPM (NE x 79), element q005 peak signal (NE x 79), promoter reads CPM (79 x NG)
C = np.load(f"{RG}/m7_step2/features/canon_counts/raw_counts.npy").astype(float); assert C.shape == (NE, 79); RE = C / np.maximum(C.sum(0), 1) * 1e6
PE = np.zeros((NE, 79)); CHI = {c: (g.start.values, g.end.values, g.index.values) for c, g in EL.groupby("chr")}
def elem_of(c, p):
    r = np.full(len(p), -1)
    for ch in np.unique(c):
        if ch not in CHI: continue
        st, en, ix = CHI[ch]; o = np.argsort(st); st, en, ix = st[o], en[o], ix[o]; m = c == ch; k = np.searchsorted(st, p[m], side="right") - 1; ok = (k >= 0) & (p[m] < en[np.maximum(k, 0)]); rr = np.full(m.sum(), -1); rr[ok] = ix[k[ok]]; r[m] = rr
    return r
for t in tfs79:
    f = f"{V}/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
    if not os.path.exists(f) or os.path.getsize(f) == 0: continue
    P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 6, 8, 9]); P.columns = ["chr", "s", "sig", "q", "off"]; P = P[P.q >= QMIN]; e = elem_of(P.chr.values.astype(str), (P.s + P.off).values); m = e >= 0
    np.maximum.at(PE[:, ti79[t]], e[m], P.sig.values[m])
WP = np.zeros((79, NG))
for t in tfs79:
    c_ = np.load(f"{RG}/m10_promoter_regulome/work/counts/{t}.npz", allow_pickle=True); ix = pd.Index(c_["genes"].astype(str)).get_indexer(ug); WP[ti79[t]] = np.where(ix >= 0, c_["w500"][ix] / float(c_["total"]) * 1e6, 0)
print("binding inputs loaded", flush=True)
# ---------------- motif (C, target-free, not HITAG)
MOTT = sorted(f[7:-4] for f in os.listdir(f"{RG}/m10_promoter_regulome/work/g") if f.startswith("psites_") and f.endswith(".tsv"))
SITE_E = {}; PROMC = {}
for t in MOTT:
    s = pd.read_csv(f"{RG}/m10_promoter_regulome/work/g/sites_{t}.tsv", sep="\t", header=None); SITE_E[t] = np.bincount(s[0].astype(int).values, minlength=NE).astype(float)
    p = pd.read_csv(f"{RG}/m10_promoter_regulome/work/g/psites_{t}.tsv", sep="\t", header=None); PROMC[t] = p[0].astype(str).value_counts().reindex(ug).fillna(0).values.astype(float)
print("motif TFs", len(MOTT), flush=True)
# ---------------- F inputs: modules, hubs, SE, Cicero edges -> element level
def reg_to_elems(regs):
    regs = [r for r in regs if isinstance(r, str) and ":" in r]
    if not regs: return np.array([], int)
    c = np.array([r.split(":")[0] for r in regs]); s = np.array([int(r.split(":")[1].split("-")[0]) for r in regs]); e = np.array([int(r.split("-")[-1]) for r in regs])
    out = []
    for ch in np.unique(c):
        if ch not in CHI: continue
        st, en, ix = CHI[ch]; o = np.argsort(st); st, en, ix = st[o], en[o], ix[o]; m = c == ch
        for a, b in zip(s[m], e[m]):
            lo = max(np.searchsorted(st, a, side="right") - 1, 0); j = lo
            while j < len(st) and st[j] < b:
                if en[j] > a: out.append(ix[j])
                j += 1
    return np.unique(np.array(out, int))
MOD = np.zeros((NE, 79), bool); HASMOD = np.zeros(79, bool)
for t in tfs79:
    f = f"{X}/cobinding/results/ourrun_k50/{t}/modules.tsv"
    if not os.path.exists(f): continue
    M = pd.read_csv(f, sep="\t"); HASMOD[ti79[t]] = True; regs = [r for v in M.regions.astype(str) for r in v.split(";")]; MOD[reg_to_elems(regs), ti79[t]] = True
HUB = np.zeros(NE); H = pd.read_csv(f"{X}/cobinding/results/ourrun_k50_clique_hubs_blacklist_removed_loop_edges/hub_regions.tsv", sep="\t")
for r, n in zip(H.hub_region, H.n_tfs):
    ix = reg_to_elems([r]); HUB[ix] = np.maximum(HUB[ix], n)
SE = pd.read_csv(f"{X}/results_3d_model/se_sedb/HEK293T_WT_SEdb_SE_02_1951_hg38.bed", sep="\t", header=None).iloc[:, :3]; SEf = np.zeros(NE); SEf[reg_to_elems([f"{a}:{b}-{c}" for a, b, c in SE.values])] = 1
# Cicero significant edges per TF (q <= 0.05) between a gene's promoter element(s) and its linked distal elements
prom_e = {}; [prom_e.setdefault(g, set()).add(e) for e, g, s in zip(Le, Lg, Ls) if s]
CIC_N = np.zeros((79, NG)); CIC_MAX = np.zeros((79, NG)); HASCIC = np.zeros(79, bool); pairkey = {(g, e): True for e, g, s in zip(Le, Lg, Ls) if not s}
for t in tfs79:
    f = f"{X}/cobinding/work/ourrun_k50/{t}/{t}.peak_edges.tsv"
    if not os.path.exists(f): continue
    HASCIC[ti79[t]] = True; Ed = pd.read_csv(f, sep="\t"); Ed = Ed[Ed.qval <= .05]
    if not len(Ed): continue
    E2G = {}
    for r1, r2, w in zip(Ed.peak1, Ed.peak2, Ed.coaccess.values):
        for x in reg_to_elems([r1]):
            for y in reg_to_elems([r2]):
                E2G[(x, y)] = max(E2G.get((x, y), 0), w); E2G[(y, x)] = max(E2G.get((y, x), 0), w)
    for (g, y) in pairkey:  # gene g: any of its promoter elements x joined to its linked distal element y
        ws = [E2G[(x, y)] for x in prom_e.get(g, ()) if (x, y) in E2G]
        if ws: CIC_N[ti79[t], g] += 1; CIC_MAX[ti79[t], g] = max(CIC_MAX[ti79[t], g], max(ws))
print("F inputs: modules for", int(HASMOD.sum()), "TFs; Cicero for", int(HASCIC.sum()), "TFs; hub elements", int((HUB > 0).sum()), "SE elements", int(SEf.sum()), flush=True)
nmod_e = MOD.sum(1).astype(float)
Fg = np.column_stack([gsum(nmod_e[Le] * La), gmax(nmod_e[Le]), gmax(HUB[Le]), gmax(SEf[Le]), CIC_N.sum(0), CIC_MAX.max(0)])
SAVE("gene_F", Fg, ["F_abc_n_tf_modules", "F_max_n_tf_modules", "F_max_hub_n_tfs", "F_SE_linked", "F_cicero_edges_all_tfs", "F_cicero_max_coaccess_all_tfs"])
# ---------------- G: candidate partners
CAND = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_frozen_v1.tsv", sep="\t"); PART = {}
for r in CAND.itertuples():
    for a, b in ((r.tf_a, r.tf_b), (r.tf_b, r.tf_a)):
        if a in ti79 and b in ti79: PART.setdefault(a, []).append((b, r.af3_5seed_tier == "strong", bool(r.known)))
# ---------------- E: TF descriptors
FL = pd.read_csv(f"{RG}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf"); KD = pd.read_csv(f"{RG}/phase1/panel_tf_xatlas_status.tsv", sep="\t")
ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; KD["p"] = KD.gene_symbol.map(lambda g: ALIAS.get(g, g)); KDp = KD.groupby("p").knockdown_pct.max()
tert = {"low": 0, "mid": 1, "high": 2}
# ---------------- variants
MIS = dict(zip(tfs79, json.load(open(f"{RG}/m7_step2/features/raw_hicfixed/feature_qc.json"))["mis_paired_map"]["mis0"]))
adec = pd.qcut(pd.Series(EL.atac.values).rank(method="first"), 10, labels=False).values; gdec = pd.qcut(pd.Series(A[0]).rank(method="first"), 10, labels=False).values
def variant(name):
    if name == "real": return RE, PE, WP, MOD
    if name == "swap":
        j = [ti79[MIS[t]] for t in tfs79]; return RE[:, j], PE[:, j], WP[j], MOD[:, j]
    rng = np.random.default_rng(SEED + int(name[-1])); R2, P2, W2, M2 = RE.copy(), PE.copy(), WP.copy(), MOD.copy()
    for k in range(79):
        for d in range(10):
            ix = np.flatnonzero(adec == d); pr = rng.permutation(ix); R2[ix, k] = RE[pr, k]; P2[ix, k] = PE[pr, k]; M2[ix, k] = MOD[pr, k]
            gx = np.flatnonzero(gdec == d); W2[k, gx] = WP[k, rng.permutation(gx)]
    return R2, P2, W2, M2
def pct(v): return rankdata(v) / len(v)
# nearest peak-holding element midpoint from each TSS (per TF), by chromosome
tss = GL.tss.values.astype(float); gch = GL.chr.values
def nearest_peak(PEk):
    out = np.full(NG, 1e7)
    for c in np.unique(gch):
        ix = np.flatnonzero(gch == c); em = np.sort(EL.mid.values[(EL.chr.values == c) & (PEk > 0)])
        if not len(em): continue
        k = np.searchsorted(em, tss[ix]); d = np.minimum(np.abs(tss[ix] - em[np.clip(k - 1, 0, len(em) - 1)]), np.abs(tss[ix] - em[np.clip(k, 0, len(em) - 1)])); out[ix] = d
    return out
for vname in sys.argv[1:] or ["real", "swap", "pseudo0", "pseudo1", "pseudo2"]:
    R_, P_, W_, M_ = variant(vname)
    D = np.column_stack([W_.T, np.column_stack([gsum(R_[Le, k] * La, dmask) for k in range(79)]), np.column_stack([gmax((P_[Le, k] > 0).astype(float)) for k in range(79)]).sum(1)])
    Dn = [f"D_prom_reads_{t}" for t in tfs79] + [f"D_distal_abc_reads_{t}" for t in tfs79] + ["D_n_tfs_linked_peak"]
    ROWS = []; names = None
    for t in TT:
        k = ti79[t]; r = R_[Le, k]; p = P_[Le, k]; ra = r * La; f = {}
        # C (motif; not HITAG, unchanged across variants)
        hm = t in SITE_E; se = SITE_E[t][Le] if hm else np.zeros(len(Le))
        f["C_has_motif"] = np.full(NG, float(hm)); f["C_prom_sites"] = PROMC[t] if hm else np.zeros(NG); f["C_distal_sites"] = gsum(se, dmask); f["C_distal_sites_abc"] = gsum(se * La, dmask)
        # B
        f["B_prom_reads"] = W_[k]; f["B_prom_peak"] = gmax(p, Ls); f["B_distal_abc_reads"] = gsum(ra, dmask); f["B_distal_abc_peak"] = gsum(p * La, dmask); f["B_max_distal_reads_abc"] = gmax(ra, dmask)
        o = np.lexsort((-ra, Lg)); Lg_o = Lg[o]; first = np.r_[0, np.flatnonzero(np.diff(Lg_o)) + 1]; rank_in = np.arange(len(o)) - np.repeat(first, np.diff(np.r_[first, len(o)]))
        for j in range(3):
            m = rank_in == j; idx = o[m]; tr, ta, td = np.zeros(NG), np.zeros(NG), np.full(NG, 5e6)
            tr[Lg[idx]] = r[idx]; ta[Lg[idx]] = La[idx]; td[Lg[idx]] = np.abs(Ld[idx]); f[f"B_top{j + 1}_reads"] = tr; f[f"B_top{j + 1}_abc"] = ta; f[f"B_top{j + 1}_dist"] = np.log10(td + 1)
        f["B_log10_dist_nearest_peak"] = np.log10(nearest_peak(P_[:, k]) + 1); f["B_pct_prom_reads"] = pct(W_[k]); f["B_pct_peak_score"] = pct(gsum(p * La))
        # BC
        f["BC_prom_sites_under_peak"] = f["C_prom_sites"] * (f["B_prom_peak"] > 0); f["BC_distal_sites_under_peak_abc"] = gsum(se * (p > 0) * La, dmask)
        # F per (TF, gene)
        mk = M_[Le, k].astype(float); others = (M_[Le].sum(1) - M_[Le, k]).astype(float)
        f["F_tf_module_abc"] = gsum(mk * La); f["F_tf_module_n_other_tfs"] = gsum(mk * others) / np.maximum(gsum(mk), 1); f["F_tf_has_modules"] = np.full(NG, float(HASMOD[k]))
        cj = ti79[MIS[t]] if vname == "swap" else k
        f["F_tf_cicero_edges"] = CIC_N[cj] if not vname.startswith("pseudo") else CIC_N[cj][np.random.default_rng(SEED + 7).permutation(NG)]; f["F_tf_cicero_max"] = CIC_MAX[cj] if not vname.startswith("pseudo") else CIC_MAX[cj][np.random.default_rng(SEED + 7).permutation(NG)]
        # G
        co, nco, nstrong, nknown = np.zeros(NG), np.zeros(NG), np.zeros(NG), np.zeros(NG)
        for (b, strong, known) in PART.get(t, []):
            kb = ti79[b]; co += gsum(np.minimum(r, R_[Le, kb]) * La); cb = gmax(((p > 0) & (P_[Le, kb] > 0)).astype(float)); nco += cb; nstrong += cb * strong; nknown += cb * known
        f["G_partner_min_reads_abc"] = co; f["G_n_partners_cobound"] = nco; f["G_n_af3strong_cobound"] = nstrong; f["G_n_known_cobound"] = nknown; f["G_n_partners_listed"] = np.full(NG, float(len(PART.get(t, []))))
        # E
        f["E_cells"] = np.full(NG, float(FL.cells.get(t, np.nan))); f["E_depth_tertile"] = np.full(NG, float(tert.get(str(FL.tertile.get(t, "")), np.nan))); f["E_kd_pct"] = np.full(NG, float(KDp.get(t, np.nan)))
        names = list(f); ROWS.append(np.column_stack([f[n] for n in names]))
    Xp = np.vstack(ROWS); SAVE(f"pair_{vname}", Xp, names); SAVE(f"gene_D_{vname}", D, Dn); print(vname, Xp.shape, D.shape, flush=True)
np.savez(f"{F}/index.npz", tfs=np.array(TT), genes=ug); print("done", flush=True)
