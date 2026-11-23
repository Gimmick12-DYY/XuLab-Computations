# m57 Amendment 4: Stage 1 links by link type and by strength rank (top 10,000 peaks, 10 kb), observed recall and precision.
import sys, os; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = "/work/users/d/y/dyy12/XuLab/regulatome"; s = open(f"{R}/m48_stage1_top10k/m48_tables.py").read(); exec(s[:s.index("VERS = {")])  # TFS, UG, OK, LISTS, elem_of, TB, L, Le, Lg, Ls, NB, V
O = f"{R}/m57_stage1_q_cutoff/out"; NT = len(TFS)
LX = pd.read_csv(f"{NB}/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["TargetGene", "ABC.Score", "distance"]); LX = LX[LX.TargetGene.isin(gi)]
ABCs = LX["ABC.Score"].values; DIST = LX["distance"].values.astype(float); assert len(ABCs) == len(Le)
NE_ = int(Le.max()) + 1; by_el = pd.Series(np.arange(len(Le))).groupby(Le).apply(np.array).to_dict()  # element -> link rows
F = np.load(f"{R}/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True)
SC = pd.DataFrame(F["score"], index=F["tfs"].astype(str), columns=F["genes"].astype(str)).reindex(index=TFS, columns=UG).values; assert (np.isfinite(SC) == OK).all()
PA = np.full(SC.shape, np.nan); PS = np.full(SC.shape, np.nan)
for i in range(NT): ok = OK[i]; n = ok.sum(); PA[i, ok] = 100 * (rankdata(np.abs(SC[i, ok])) - 0.5) / n; PS[i, ok] = 100 * (rankdata(SC[i, ok]) - 0.5) / n
NGn = len(UG); PROM = np.zeros((NT, NGn), bool); DIS = np.zeros((NT, NGn), bool); WIN = np.zeros((NT, NGn), bool); DMIN = np.full((NT, NGn), np.inf); S1 = np.zeros((NT, NGn)); S2 = np.zeros((NT, NGn))
for i, t in enumerate(TFS):
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= 10000]
    c5, p5, sg = P.c.values.astype(str), (P.s + P.off).values, P.sig.values; e = elem_of(c5, p5)
    for k in np.flatnonzero(e >= 0):
        rows = by_el.get(e[k]);
        if rows is None: continue
        g = Lg[rows]; slf = Ls[rows]; ab = ABCs[rows]; ds = DIST[rows]
        PROM[i, g[slf]] = True; DIS[i, g[~slf]] = True; np.minimum.at(DMIN[i], g[~slf], np.abs(ds[~slf])); np.maximum.at(S1[i], g, sg[k]); np.maximum.at(S2[i], g, sg[k] * ab)
    for ch in np.unique(c5):
        if ch not in TB: continue
        tp, ix = TB[ch]; m = c5 == ch; x = p5[m]; sx = sg[m]; a = np.searchsorted(tp, x - 10000); b = np.searchsorted(tp, x + 10000, side="right")
        for aa, bb, sv in zip(a, b, sx):
            gg = ix[aa:bb]; WIN[i, gg] = True; S1[i, gg] = np.maximum(S1[i, gg], sv); S2[i, gg] = np.maximum(S2[i, gg], sv * 0.02)
    print(t, flush=True)
LINK = PROM | DIS | WIN; B10 = np.vstack([LISTS["10 kb"][t] for t in TFS]); assert (LINK == B10).all(), "link union differs from the m48 10 kb lists"
LINK &= OK; PROM &= OK; DIS &= OK; WIN &= OK; npair = OK.sum()
C1 = PROM; C2 = DIS & ~PROM; C3 = WIN & ~PROM & ~DIS
TG = {}
for x in (5, 1): TG[(x, "either")] = OK & (PA > 100 - x); TG[(x, "up")] = OK & (PS > 100 - x); TG[(x, "down")] = OK & (PS < x)
assert int((LINK & TG[(5, "either")]).sum()) == 11536 and int((LINK & TG[(1, "either")]).sum()) == 2613, "in-both counts differ from m56"
print("sanity passed: links", int(LINK.sum()), flush=True)
def stats(name, M, kind):
    out = []
    for (x, d), T in TG.items():
        ib = (M & T).sum(); out.append({"class": name, "kind": kind, "targets": f"top {x}%", "direction": d, "links": int(M.sum()), "median genes per TF": float(np.median(M.sum(1))), "share of all TF-gene pairs (%)": 100 * M.sum() / npair,
                                        "in both": int(ib), "recall contribution (%)": 100 * ib / T.sum(), "precision (%)": 100 * ib / max(M.sum(), 1), "precision / chance": (ib / max(M.sum(), 1)) / (x / 100)})
    return out
rows = []
for nm, M in (("(1) promoter of the gene", C1), ("(2) distal ABC element, no promoter summit", C2), ("(3) 10 kb window only", C3), ("all links (10 kb rule)", LINK)): rows += stats(nm, M, "exclusive")
for nm, M in (("any promoter", PROM), ("any distal ABC element", DIS), ("any 10 kb window", WIN)): rows += stats(nm, M, "non-exclusive")
for lo, hi in ((0, 10e3), (10e3, 50e3), (50e3, 200e3), (200e3, np.inf)):
    M = C2 & (DMIN >= lo) & (DMIN < hi); rows += stats(f"(2) distal ABC, nearest bound element {lo / 1e3:g}-{hi / 1e3:g} kb" if np.isfinite(hi) else f"(2) distal ABC, nearest bound element > {lo / 1e3:g} kb", M, "distal by distance")
A_ = pd.DataFrame(rows)
assert A_[(A_.kind == "exclusive") & (A_["class"].str.startswith("(")) & (A_.targets == "top 5%") & (A_.direction == "either")]["in both"].sum() == 11536
RK = []
for sn, Sx in (("strongest linked peak signalValue", S1), ("max signalValue x ABC.Score (window-only: x 0.02)", S2)):
    for N in (100, 250, 500, 1000, 2000, None):
        M = np.zeros_like(LINK)
        for i in range(NT):
            g = np.flatnonzero(LINK[i]); o = g[np.argsort(-Sx[i, g], kind="stable")]; M[i, o[:N] if N else o] = True
        for (x, d), T in TG.items():
            ib = (M & T).sum(); RK.append({"ranking": sn, "top N genes per TF": N if N else "all", "targets": f"top {x}%", "direction": d, "links": int(M.sum()), "in both": int(ib), "recall (%)": 100 * ib / T.sum(), "precision (%)": 100 * ib / M.sum(), "precision / chance": (ib / M.sum()) / (x / 100)})
RK = pd.DataFrame(RK)
with pd.ExcelWriter(f"{O}/stage1_link_type_and_rank_raw.xlsx") as w:
    A_.to_excel(w, sheet_name="link type", index=False); RK.to_excel(w, sheet_name="strength rank", index=False)
    pd.DataFrame({"note": ["Observed only, no shuffle. 63 TFs, top 10,000 lab peaks (signalValue), 10 kb rule; links = summit in an element ABC-linked to the gene (canonical m2c Hi-C build, ABC >= 0.02, self-promoter included) or summit within 10 kb of the TSS.",
      "Classes: (1) summit in the gene's self-promoter element; (2) else summit in a non-self element ABC-linked to the gene; (3) else window only. Distance = ABC 'distance' of the nearest bound distal element.",
      "Targets: top X % of |score| within the TF (paper per-gene score); up / down from the signed score. Recall contribution = in both / targets; precision = in both / links of the class; precision / chance = precision / (X/100).",
      "Ranking (i): per gene the strongest signalValue of any peak linking it; (ii): max over links of signalValue x ABC.Score, window-only links signalValue x 0.02."]}).to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 20)
k = ["class", "targets", "direction", "links", "median genes per TF", "share of all TF-gene pairs (%)", "in both", "recall contribution (%)", "precision (%)", "precision / chance"]
print(A_[A_.direction == "either"][k].round(3).to_string(index=False)); print(A_[A_.direction != "either"][k].round(3).to_string(index=False))
print(RK.round(3).to_string(index=False)); print("done")
