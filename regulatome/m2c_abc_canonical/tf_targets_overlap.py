# m2c canonical ABC -> per-TF target genes -> overlap with HEK293T Perturb-seq knockdown responders (SPEC.md steps 3-4; fixed procedure, no tuning).
# Targets: lab MACS2 peaks (TAGATG-293T/bigwig/macs2/TF.<NAME>, read only) with q <= 0.05 (narrowPeak col 9 >= -log10 0.05) whose summit falls in a
# canonical candidate; targets = genes with an ABC >= 0.02 link (self-promoter links included, as the repo default) to >= 1 bound element.
# Perturb-seq: kd_response/out_v2 z, genes with NTC detection >= 0.1, kept primary units averaged per TF (HP1 = CBX1/CBX5, ZUFSP = ZUP1), double-centred
# (b3b4 load(), as B6 / m6). Universe = ABC gene set (GeneList) & KD genes. Responders = top 5% |R| of that TF within the universe.
# Per TF and arm: Fisher OR (2x2, +0 correction; Haldane 0.5 for logOR only) + two-sided p, BH across TFs per arm.
# Null (a): 1,000 draws (seed 20261110 + TF index), same number of elements from all candidates matched on joint ATAC decile x H3K27ac decile,
# same ABC >= 0.02 rule -> median null OR, empirical p = (1 + #null OR >= obs) / 1001. (b) ATAC-only: the N most accessible candidates (N = bound
# elements) -> OR_ATAC. Continuity: B6 effect (own z vs 1,000 expression-matched random gene sets) and specificity (own z minus mean off-KD z, off-KDs
# from the TF's own split set as in m6 / heldout_v1); Wspec in pooled.py. usage: tf_targets_overlap.py <biosample> (HEK293T_hic | HEK293T_powerlaw)
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy import stats
X = "/work/users/d/y/dyy12/XuLab"; C = f"{X}/regulatome/m2c_abc_canonical"; BS = sys.argv[1]; R = f"{C}/results/{BS}"; O = f"{C}/overlap/{BS}"; os.makedirs(O, exist_ok=True)
PK = "/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2"; M4 = f"{X}/regulatome/m4_validation"
SEED = 20261110; ND = 1000; NR = 1000; ARMS = {"arm2": "arm2_all_cell", "arm1": "arm1_high_knockdown"}; ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; panel = lambda t: ALIAS.get(t, t)
QMIN = -np.log10(0.05)
E = pd.read_csv(f"{R}/Neighborhoods/EnhancerList.txt", sep="\t")
ac = [c for c in E.columns if c.startswith("ATAC.") and c.endswith(".readCount")][0]; kc = [c for c in E.columns if c.startswith("H3K27ac.") and c.endswith(".readCount")][0]
E["key"] = E.chr + ":" + E.start.astype(str) + "-" + E.end.astype(str); NE = len(E); ekey = dict(zip(E.key, range(NE)))
L = pd.read_csv(sorted(f"{R}/Predictions/{f}" for f in os.listdir(f"{R}/Predictions") if f.startswith("EnhancerPredictionsFull_threshold0.02") and f.endswith(".tsv"))[0], sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score"])
L["e"] = (L.chr + ":" + L.start.astype(str) + "-" + L.end.astype(str)).map(ekey); assert L.e.notna().all(); L["e"] = L.e.astype(int)
GL = pd.read_csv(f"{R}/Neighborhoods/GeneList.txt", sep="\t", usecols=["name"]).name.astype(str).unique()
genes = np.array(sorted(set(GL))); gix = {g: i for i, g in enumerate(genes)}
M = sp.csr_matrix((np.ones(len(L)), (L.e.values, L.TargetGene.map(gix).values)), shape=(NE, len(genes)))  # element x gene, ABC >= 0.02
M.data[:] = 1
def targets_of(el):
    return np.asarray(M[el].sum(0)).ravel() > 0
# element strata: joint ATAC decile x H3K27ac decile (ranks so ties split)
dec = lambda v: np.minimum((pd.Series(v).rank(method="first").values - 1) * 10 // len(v), 9).astype(int)
strat = dec(E[ac].values) * 10 + dec(E[kc].values); SIDX = {s: np.flatnonzero(strat == s) for s in np.unique(strat)}
atac_order = np.argsort(-E[ac].values, kind="stable")
# bound elements per TF: summit in a candidate (candidates are merged, non-overlapping; sorted per chromosome)
CH = {c: (g.start.values, g.end.values, g.index.values) for c, g in E.sort_values(["chr", "start"]).groupby("chr")}
tfs = sorted(d[3:] for d in os.listdir(PK) if d.startswith("TF."))
Sp = pd.read_csv(f"{X}/regulatome/m6_improve/split_v1.tsv", sep="\t"); SET = dict(zip(Sp.tf, Sp.set))
B = {}; rows0 = []
for t in tfs:
    f = f"{PK}/TF.{t}/TF.{t}_peaks.narrowPeak"; P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 8, 9]); P.columns = ["chr", "start", "q", "off"]
    n_all = len(P); P = P[P.q >= QMIN]; sm = (P.start + P.off).values; el = []
    for c, s in zip(P.chr.values, sm):
        if c not in CH: continue
        st, en, ix = CH[c]; k = np.searchsorted(st, s, side="right") - 1
        if k >= 0 and s < en[k]: el.append(ix[k])
    el = np.unique(np.array(el, int)); B[t] = el; tg = targets_of(el) if len(el) else np.zeros(len(genes), bool)
    rows0.append(dict(tf=t, set=SET.get(t, "not_in_split"), n_peaks_all=n_all, n_peaks_q05=len(P), n_bound_elements=len(el), n_targets=int(tg.sum())))
T0 = pd.DataFrame(rows0); print(T0.to_string(index=False), flush=True)
def load(a):  # b3b4.py load() (no replicate part), as grid6 / bench_k27ac
    d = np.load(f"{M4}/kd_response/out_v2/{ARMS[a]}/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{M4}/kd_response/out_v2/{ARMS[a]}/guide_qc.tsv", sep="\t"); u = d["units"].tolist(); ui = {x: k for k, x in enumerate(u)}; uni = d["ntc_detect"] >= .1; g = d["genes"][uni]; Z = d["z"][:, uni].astype(float); kp = q[q.kept == 1]; prim = kp[kp.kind.isin(["all", "construct"])]; tf_ = sorted(prim.tf.unique())
    P_ = np.vstack([np.nanmean(Z[[ui[x] for x in prim.unit[prim.tf == t]]], axis=0) for t in tf_]); P_ = np.nan_to_num(P_); Rr = P_ - P_.mean(0); Rr = Rr - Rr.mean(1, keepdims=True)
    return dict(g=g, R=Rr, tfs=tf_, mean=d["ntc_mean"][uni].astype(float))
def fisher(inl, resp):
    a = int((inl & resp).sum()); b = int((inl & ~resp).sum()); c = int((~inl & resp).sum()); d = int((~inl & ~resp).sum())
    o, p = stats.fisher_exact([[a, b], [c, d]]); return o, p, a, np.log((a + .5) * (d + .5) / ((b + .5) * (c + .5)))
rows = []
for a in ARMS:
    D = load(a); kg = {x: k for k, x in enumerate(D["g"])}; uni = np.array([g in kg for g in genes]); ug = genes[uni]; ukix = np.array([kg[g] for g in ug])
    pts = sorted({panel(t) for t in D["tfs"]}); Rp = np.vstack([D["R"][[i for i, t in enumerate(D["tfs"]) if panel(t) == p]].mean(0) for p in pts])
    Ap = np.abs(Rp[:, ukix]).astype(np.float32); pi = {p: k for k, p in enumerate(pts)}; nu = len(ug)
    bins = pd.qcut(D["mean"][ukix], 20, labels=False, duplicates="drop"); BG = [np.flatnonzero(bins == b) for b in range(bins.max() + 1)]
    rng_b6 = np.random.default_rng(SEED + (a == "arm2"))
    print(a, "universe", nu, "KD TFs", len(pts), flush=True)
    for ti, t in enumerate(tfs):
        r0 = T0[T0.tf == t].iloc[0].to_dict(); r = dict(arm=a, **r0)
        if t not in pi: r["status"] = "no_KD"; rows.append(r); continue
        tg = targets_of(B[t])[uni] if len(B[t]) else np.zeros(nu, bool); r["n_targets_universe"] = int(tg.sum()); r["universe"] = nu
        if tg.sum() < 20: r["status"] = "untestable_lt20"; rows.append(r); continue
        resp = np.zeros(nu, bool); resp[np.argsort(-Ap[pi[t]], kind="stable")[:int(round(nu * .05))]] = True
        o, p, k, lo = fisher(tg, resp); r.update(status="tested", OR=o, p=p, n_overlap=k, logOR=lo)
        # null (a): matched elements; same draws for both arms (seeded by TF index)
        rng = np.random.default_rng(SEED + ti); cnt = pd.Series(strat[B[t]]).value_counts(); nul = []
        for _ in range(ND):
            el = np.concatenate([rng.choice(SIDX[s], n, replace=len(SIDX[s]) < n) for s, n in cnt.items()]); nul.append(fisher(targets_of(el)[uni], resp)[0])
        nul = np.array(nul); r.update(null_OR_median=float(np.median(nul)), null_p=(1 + int((nul >= o).sum())) / (ND + 1))
        oa = fisher(targets_of(atac_order[:len(B[t])])[uni], resp)[0]; r.update(OR_ATAC=oa, OR_minus_ATAC=o - oa)
        # continuity: B6 effect and specificity over the TF's own split set of KDs (m6 convention)
        ix = np.flatnonzero(tg); cntb = np.bincount(bins[ix], minlength=len(BG)); rs = np.hstack([BG[b][rng_b6.integers(0, len(BG[b]), (NR, c))] for b, c in enumerate(cntb) if c])
        same = [q for q in pts if SET.get(q) == SET.get(t)] if SET.get(t) in ("development", "heldout") else pts; si = [pi[q] for q in same]
        obs = Ap[si][:, ix].mean(1); rm = np.stack([Ap[k_][rs].mean(1) for k_ in si]); z = (obs - rm.mean(1)) / rm.std(1); own = z[same.index(t)]
        r.update(b6_effect_z_own=own, z_off_mean=float(np.delete(z, same.index(t)).mean()), specificity=own - float(np.delete(z, same.index(t)).mean()), own_rank=int((z > own).sum() + 1), n_kd_set=len(same))
        rows.append(r)
    print(a, "done", flush=True)
P = pd.DataFrame(rows)
for a in ARMS:
    m = (P.arm == a) & (P.status == "tested"); P.loc[m, "BH"] = stats.false_discovery_control(P.loc[m, "p"].values)
P.to_csv(f"{O}/per_tf.tsv", sep="\t", index=False, float_format="%.4g")
