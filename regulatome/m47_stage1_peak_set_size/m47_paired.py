# m47 Amendment 3: paired bulk-vs-ours test with one shared permutation null, plus a matched-list-size version.
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; X = "/work/users/d/y/dyy12/XuLab"; M45 = f"{R}/m45_stage1_summary"
s = open(f"{M45}/m45_tables.py").read(); exec(s[:s.index("VERS = {")])
O = f"{R}/m47_stage1_peak_set_size/out"  # set after the exec, which defines its own O
D = f"{X}/data"
REF = {"CTCF": "CTCF_HEK293T_GSE103651_peaks.bed", "ZNF282": "ZNF282_HEK293_peaks.bed", "MAZ": "MAZ_HEK293_peaks.bed", "ZIC2": "ZIC2_HEK293_peaks.bed", "ZNF777": "ZNF777_HEK293_peaks.bed", "SETDB1": "SETDB1_HEK293_peaks.bed"}
ti = {t: i for i, t in enumerate(TFS)}; NG_ = len(UG); NP = 1000; rng = np.random.default_rng(20261118)
def ranked(path, n, w):
    P = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]).sort_values("sig", ascending=False, kind="stable").reset_index(drop=True)
    P = P.iloc[:n]; rank = np.full(NG_, np.inf); done = np.zeros(NG_, bool); k = 0
    for b in range(0, len(P), 100):
        x = P.iloc[b:b + 100]; c5, p5 = x.c.values.astype(str), (x.s + x.off).values; A, _ = rule_abc(c5, p5); A = A | window(c5, p5, w) if w else A
        new = np.flatnonzero(A & ~done)
        rank[new] = k + np.arange(len(new)); k += len(new); done |= A
    return rank  # finite = in list; order = best linking peak, then gene index
rows = []
for t, f in REF.items():
    i = ti[t]; ok = OK[i]; s_ = STRAT[ok]; idx = [np.flatnonzero(s_ == st) for st in np.unique(s_)]
    for c in (2, 3, 4, 5):
        lab = (AZ[i] >= c)[ok]; n = int(lab.sum())
        perm = np.zeros((NP, ok.sum()), bool)
        for k in range(NP):
            pl = lab.copy()
            for g in idx: pl[g] = lab[g][rng.permutation(len(g))]
            perm[k] = pl
        permi = perm.astype(np.int32)
        for p, w in PADS.items():
            for pair, nb, no in (("top 5,000", 5000, 5000), ("top 10,000", 10000, 10000)):
                rb = ranked(f"{D}/{f}", nb, w)[ok]; ro = ranked(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", no, w)[ok]
                for ver in ("as is", "matched list size"):
                    lb, lo = np.isfinite(rb), np.isfinite(ro)
                    if ver == "matched list size":
                        m_ = min(lb.sum(), lo.sum()); lb = rb < m_; lo = ro < m_
                    base = {"TF": t, "pair": pair, "version": ver, "padding": p, "cutoff": f"|z| >= {c}", "Perturb-seq genes": n, "n < 10": n < 10, "bulk linked genes": int(lb.sum()), "ours linked genes": int(lo.sum())}
                    if n == 0: rows.append(base); continue
                    ob, oo = int((lb & lab).sum()), int((lo & lab).sum()); kb, ko = permi @ lb.astype(np.int32), permi @ lo.astype(np.int32); mb, mo = kb.mean(), ko.mean()  # integer counts (a bool @ bool product would give 0/1)
                    Dr = oo / mo - ob / mb; Dg = 100 * ((oo - mo) - (ob - mb)) / n
                    nr = ko / mo - kb / mb; ng = 100 * ((ko - mo) - (kb - mb)) / n
                    rows.append({**base, "bulk recall (%)": 100 * ob / n, "bulk shuffle recall (%)": 100 * mb / n, "bulk ratio": ob / mb, "bulk gap (points)": 100 * (ob - mb) / n,
                                 "ours recall (%)": 100 * oo / n, "ours shuffle recall (%)": 100 * mo / n, "ours ratio": oo / mo, "ours gap (points)": 100 * (oo - mo) / n,
                                 "ratio difference (ours - bulk)": Dr, "p two-sided (ratio difference)": (1 + int((np.abs(nr) >= abs(Dr) - 1e-12).sum())) / (NP + 1),
                                 "gap difference (ours - bulk, points)": Dg, "p two-sided (gap difference)": (1 + int((np.abs(ng) >= abs(Dg) - 1e-12).sum())) / (NP + 1)})
        print(t, c, flush=True)
T = pd.DataFrame(rows); T.to_csv(f"{O}/5_paired_bulk_vs_ours.tsv", sep="\t", index=False, float_format="%.4g")
ntest = int((~T["n < 10"]).sum()); print("tests with n >= 10 (each with a ratio and a gap difference):", ntest)
with pd.ExcelWriter(f"{O}/stage1_bulk_vs_ours_paired_test_no_ZBTB7A.xlsx") as w:
    for ver in ("as is", "matched list size"):
        for p in PADS: T[(T.version == ver) & (T.padding == p)].to_excel(w, sheet_name=f"{'matched size' if ver != 'as is' else 'as is'}, padding {p}"[:31], index=False)
    pd.DataFrame({"note": ["Paired test: one shared permutation null per TF and cutoff (1,000 permutations of the Perturb-seq labels within NTC expression x promoter ATAC deciles, seed 20261118); both lists scored on each permutation.",
                           "ratio = observed / permutation-mean overlap; gap = 100 x (observed - mean) / Perturb-seq genes. Difference = ours - bulk; null of the difference is formed from the same permutations, centred at 0 by construction; p two-sided = (1 + #|null| >= |observed|) / 1001.",
                           "Matched list size: the longer linked-gene list is cut to the shorter one's size, genes ranked by their best linking peak (signalValue; peaks added in batches of 100, ties by gene index).",
                           f"Tests with >= 10 Perturb-seq genes: {ntest} (each reported for the ratio and the gap difference). ZBTB7A removed. Bulk references: CTCF and ZNF282 HEK293T; MAZ, ZIC2, ZNF777, SETDB1 HEK293."]}).to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 400)
c_ = ["TF", "pair", "version", "cutoff", "Perturb-seq genes", "bulk linked genes", "ours linked genes", "bulk ratio", "ours ratio", "ratio difference (ours - bulk)", "p two-sided (ratio difference)", "gap difference (ours - bulk, points)", "p two-sided (gap difference)"]
print(T[(T.padding == "10 kb") & (~T["n < 10"])][c_].round(3).to_string(index=False)); print("done")
