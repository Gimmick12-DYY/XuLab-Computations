# m57 Amendment 6: pure distal TF-gene links split by ABC score, peak strength, loop support, element count, H3K27ac, class; nearest-gene and loop-only rules (observed only).
import sys, os; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = "/work/users/d/y/dyy12/XuLab/regulatome"; s = open(f"{R}/m48_stage1_top10k/m48_tables.py").read(); exec(s[:s.index("VERS = {")])
O = f"{R}/m57_stage1_q_cutoff/out"; NT = len(TFS); NGn = len(UG); X_ = "/work/users/d/y/dyy12/XuLab"
LX = pd.read_csv(f"{NB}/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["TargetGene", "ABC.Score", "class"]); LX = LX[LX.TargetGene.isin(gi)]; ABCs = LX["ABC.Score"].values; CLS = LX["class"].values; assert len(ABCs) == len(Le)
ELq = pd.read_csv(f"{NB}/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["H3K27ac.RPM.quantile"])["H3K27ac.RPM.quantile"].values; assert len(ELq) == NE
K27 = ELq >= 0.5; MID = ((EL.start + EL.end) // 2).values; ECH = EL.chr.values.astype(str)
TSSg = GL.tss.values.astype(float); TCHg = GL.chr.values.astype(str)
LP = pd.read_csv(f"{X_}/hic/work_wt293t/loops/peakachu_5000.loops.0.7.bedpe", sep="\t", header=None, usecols=[0, 1, 2, 3, 4, 5], names=["c1", "s1", "e1", "c2", "s2", "e2"]); LP = LP[LP.c1 == LP.c2]; print("intra-chromosomal loops", len(LP), flush=True)
LPC = {c: g for c, g in LP.groupby("c1")}
# loop support per ABC link row (TF independent)
LOOPROW = np.zeros(len(Le), bool)
for c, G in LPC.items():
    rows = np.flatnonzero(ECH[Le] == c)
    if not len(rows): continue
    em = MID[Le[rows]][:, None]; ts = TSSg[Lg[rows]][:, None]; s1, e1, s2, e2 = (G[k].values[None, :] for k in ("s1", "e1", "s2", "e2"))
    for st in range(0, len(rows), 4000):
        a = em[st:st + 4000]; b = ts[st:st + 4000]
        h = (((a >= s1 - 5000) & (a < e1 + 5000) & (b >= s2 - 5000) & (b < e2 + 5000)) | ((a >= s2 - 5000) & (a < e2 + 5000) & (b >= s1 - 5000) & (b < e1 + 5000))).any(1); LOOPROW[rows[st:st + 4000]] = h
print("ABC non-self links with loop support:", int(LOOPROW[~Ls].sum()), "of", int((~Ls).sum()), flush=True)
by_el = pd.Series(np.arange(len(Le))).groupby(Le).apply(np.array).to_dict()
F = np.load(f"{R}/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True)
SC = pd.DataFrame(F["score"], index=F["tfs"].astype(str), columns=F["genes"].astype(str)).reindex(index=TFS, columns=UG).values; assert (np.isfinite(SC) == OK).all()
PA = np.full(SC.shape, np.nan); PS = np.full(SC.shape, np.nan)
for i in range(NT): ok = OK[i]; n = ok.sum(); PA[i, ok] = 100 * (rankdata(np.abs(SC[i, ok])) - 0.5) / n; PS[i, ok] = 100 * (rankdata(SC[i, ok]) - 0.5) / n
sh = lambda dt, v: np.full((NT, NGn), v, dtype=dt)
PROM = sh(bool, False); WIN = sh(bool, False); DIS = sh(bool, False); BABC = sh(float, 0.); BSIG = sh(float, 0.); NEL = sh(int, 0); LOOP = sh(bool, False); HK = sh(bool, False); BCLS = sh(object, ""); LOOPONLY = sh(bool, False); NEAR = sh(bool, False); NEAR100 = sh(bool, False)
for i, t in enumerate(TFS):
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= 10000]
    c5, p5, sg = P.c.values.astype(str), (P.s + P.off).values, P.sig.values; e = elem_of(c5, p5); seen = {}
    for k in np.flatnonzero(e >= 0):
        rows = by_el.get(e[k])
        if rows is None: continue
        g = Lg[rows]; slf = Ls[rows]; PROM[i, g[slf]] = True
        for r_, gg in zip(rows[~slf], g[~slf]):
            DIS[i, gg] = True; seen.setdefault(gg, set()).add(e[k])
            if ABCs[r_] > BABC[i, gg]: BABC[i, gg] = ABCs[r_]; BCLS[i, gg] = CLS[r_]
            BSIG[i, gg] = max(BSIG[i, gg], sg[k]); LOOP[i, gg] |= LOOPROW[r_]; HK[i, gg] |= K27[e[k]]
    for gg, es in seen.items(): NEL[i, gg] = len(es)
    for ch in np.unique(c5):
        if ch not in TB: continue
        tp, ix = TB[ch]; m = c5 == ch; x = p5[m]
        a = np.searchsorted(tp, x - 10000); b = np.searchsorted(tp, x + 10000, side="right")
        for aa, bb in zip(a, b): WIN[i, ix[aa:bb]] = True
        far = b == a  # no TSS within 10 kb
        if len(tp) and far.any():
            k_ = np.clip(np.searchsorted(tp, x[far]), 1, len(tp) - 1) if len(tp) > 1 else np.zeros(far.sum(), int)
            if len(tp) > 1: k_ = np.where(np.abs(tp[k_ - 1] - x[far]) <= np.abs(tp[k_] - x[far]), k_ - 1, k_)
            d_ = np.abs(tp[k_] - x[far]); NEAR[i, ix[k_]] = True; NEAR100[i, ix[k_[d_ <= 100000]]] = True
        if ch in LPC:
            G = LPC[ch]
            for A1, A2, B1, B2 in ((G.s1.values, G.e1.values, G.s2.values, G.e2.values), (G.s2.values, G.e2.values, G.s1.values, G.e1.values)):
                hit = ((x[:, None] >= A1[None, :] - 5000) & (x[:, None] < A2[None, :] + 5000)).any(0)
                for bs, be in zip(B1[hit], B2[hit]): LOOPONLY[i, ix[np.searchsorted(tp, bs - 5000):np.searchsorted(tp, be + 5000)]] = True
    print(t, flush=True)
OKm = lambda M: M & OK
CUR = OKm(PROM | DIS | WIN); assert int(CUR.sum()) == 212087
PD = OKm(DIS & ~PROM & ~WIN); NONPW = OKm(~PROM & ~WIN)
TG = {}
for x in (5, 1): TG[(x, "either")] = OK & (PA > 100 - x); TG[(x, "up")] = OK & (PS > 100 - x); TG[(x, "down")] = OK & (PS < x)
def stats(block, name, M):
    M = OKm(M); r = {"block": block, "split": name, "links": int(M.sum()), "median genes per TF": float(np.median(M.sum(1)))}
    for (x, d), T in TG.items():
        ib = int((M & T).sum()); r[f"top {x}% {d}: in both"] = ib; r[f"top {x}% {d}: recall contribution (%)"] = 100 * ib / T.sum(); r[f"top {x}% {d}: precision (%)"] = 100 * ib / max(M.sum(), 1); r[f"top {x}% {d}: precision / chance"] = (ib / max(M.sum(), 1)) / (x / 100)
    return r
rows = [stats("reference", "all links, current rule", CUR), stats("reference", "pure distal links (ABC, no promoter summit, no summit within 10 kb)", PD)]
for lo, hi in ((0.02, 0.03), (0.03, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, np.inf)): rows.append(stats("A ABC score of best linking element", f"{lo}-{hi}" if np.isfinite(hi) else f"> {lo}", PD & (BABC >= lo) & (BABC < hi)))
Q = np.zeros((NT, NGn), int)
for i in range(NT):
    g = np.flatnonzero(PD[i])
    if len(g): q = pd.qcut(pd.Series(BSIG[i, g]).rank(method="first"), 5, labels=False).values; Q[i, g] = q + 1
for q in range(1, 6): rows.append(stats("B peak strength quintile within TF (5 = strongest)", f"quintile {q}", PD & (Q == q)))
for n in (100, 250, 500):
    M = np.zeros((NT, NGn), bool)
    for i in range(NT): g = np.flatnonzero(PD[i]); M[i, g[np.argsort(-BSIG[i, g], kind="stable")][:n]] = True
    rows.append(stats("B peak strength", f"top {n} distal genes per TF", M))
rows += [stats("C Peakachu loop", "loop-supported element-TSS pair", PD & LOOP), stats("C Peakachu loop", "no loop support", PD & ~LOOP), stats("C loop-only rule (no ABC)", "summit in a loop anchor, TSS in the partner anchor (all)", LOOPONLY), stats("C loop-only rule (no ABC)", "same, no promoter / 10 kb link", LOOPONLY & NONPW)]
for lab, M in (("1", NEL == 1), ("2", NEL == 2), ("3-4", (NEL >= 3) & (NEL <= 4)), (">= 5", NEL >= 5)): rows.append(stats("D bound distal elements linked to the gene", lab, PD & M))
rows += [stats("E H3K27ac (element in top half of candidate H3K27ac RPM)", "at least one bound element in top half", PD & HK), stats("E H3K27ac (element in top half of candidate H3K27ac RPM)", "none", PD & ~HK)]
for c in ("genic", "intergenic", "promoter"): rows.append(stats("ABC class of the best linking element", c, PD & (BCLS == c)))
rows += [stats("F nearest-gene rule", "summit > 10 kb from any TSS -> nearest TSS (all)", NEAR), stats("F nearest-gene rule", "same, capped at 100 kb", NEAR100), stats("F nearest-gene rule", "nearest TSS (any distance), no promoter / 10 kb link", NEAR & NONPW), stats("F nearest-gene rule", "nearest TSS <= 100 kb, no promoter / 10 kb link", NEAR100 & NONPW)]
rows += [stats("G combination", "ABC > 0.05 and strongest peak quintile", PD & (BABC > 0.05) & (Q == 5)), stats("G combination", "loop-supported and strongest peak quintile", PD & LOOP & (Q == 5))]
T = pd.DataFrame(rows)
with pd.ExcelWriter(f"{O}/stage1_distal_links_raw.xlsx") as w:
    T.to_excel(w, sheet_name="distal splits", index=False)
    pd.DataFrame({"note": ["Observed only, no shuffle. 63 TFs, top 10,000 lab peaks, summits; canonical m2c Hi-C ABC (>= 0.02). Pure distal = linked through a non-self ABC element with no summit in the gene's own promoter element and no summit within 10 kb of its TSS.",
      "Per link: best ABC.Score among the bound linking elements (class from that element); strongest linking peak signalValue; distinct bound elements; loop support = a Peakachu loop (peakachu_5000.loops.0.7, intra-chromosomal) with the element midpoint within one anchor +- 5 kb and the TSS within the other anchor +- 5 kb, either orientation; H3K27ac = any bound element with EnhancerList H3K27ac.RPM.quantile >= 0.5.",
      "Loop-only rule: gene linked if a summit lies within a loop anchor +- 5 kb and its TSS within the partner anchor +- 5 kb. Nearest-gene rule: each summit with no universe TSS within 10 kb assigned to the nearest universe TSS on its chromosome. ChromHMM split skipped: ENCFF071AXS not available on Longleaf.",
      "Targets: top X % of |score| within the TF; up / down from the signed score. Recall contribution = in both / all targets; precision = in both / links of the row; precision / chance = precision / (X/100). Non-self ABC links carry no 'promoter' class in this build (all 22,066 promoter-class links are self-promoter)."]}).to_excel(w, sheet_name="notes", index=False)
# append block to the report
REPf = f"{O}/stage1_precision_recall_report.xlsx"; REP = pd.read_excel(REPf, sheet_name="report"); PT = pd.read_excel(REPf, sheet_name="per_TF"); NO = pd.read_excel(REPf, sheet_name="notes")
REP = REP[REP.block != "distal links"]; add = []
for _, r in T[T.block.isin(["reference", "A ABC score of best linking element", "B peak strength", "C Peakachu loop", "C loop-only rule (no ABC)", "E H3K27ac (element in top half of candidate H3K27ac RPM)", "ABC class of the best linking element", "F nearest-gene rule", "G combination"])].iterrows():
    if r.split == "all links, current rule" or r.links == 0: continue
    add.append({"block": "distal links", "setting": f"{r.block}: {r.split}" if r.block != "reference" else r.split, "peaks per TF (median)": 10000.0, "our linked genes per TF (median)": r["median genes per TF"], "our links (total)": r.links, "Perturb-seq targets per TF (median)": 561.0,
                "targets (total)": 35343, "targets as % of all genes": 5.0, "in both": r["top 5% either: in both"], "recall %": r["top 5% either: recall contribution (%)"], "precision %": r["top 5% either: precision (%)"], "TFs with no links": np.nan})
REP = pd.concat([REP, pd.DataFrame(add)], ignore_index=True); NO = NO[NO.block != "distal links"]
NO = pd.concat([NO, pd.DataFrame([{"block": "distal links", "source": "m57 distal run (m57_distal.py, spec Amendment 6), top 5 % targets; recall column = contribution to the headline recall; details and top 1 % / up / down in stage1_distal_links_raw.xlsx."}])], ignore_index=True)
with pd.ExcelWriter(REPf) as w:
    REP.to_excel(w, sheet_name="report", index=False); PT.to_excel(w, sheet_name="per_TF", index=False); NO.to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 330); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 70)
k = ["block", "split", "links", "median genes per TF", "top 5% either: in both", "top 5% either: recall contribution (%)", "top 5% either: precision (%)", "top 5% either: precision / chance", "top 1% either: precision / chance", "top 5% up: precision / chance", "top 5% down: precision / chance"]
print(T[k].round(3).to_string(index=False)); print("done")
