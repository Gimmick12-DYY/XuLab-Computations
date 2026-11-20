# m51 Amendment 1: the paper's per-gene score (per-cell z against the cell's own GEM NTC mean and SD, averaged over the TF's top-25% cells) as the Stage 1 responder rule.
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr
SRC50 = open("/work/users/d/y/dyy12/XuLab/regulatome/m50_stage1_log2fc/m50_stage1.py").read()
exec(SRC50[:SRC50.index("# ---- responder counts")]); exec(SRC50[SRC50.index("# ---- Stage 1"):SRC50.index("ST, LOO = stage1(")])
O = "/work/users/d/y/dyy12/XuLab/regulatome/m51_edistance/out"; M23 = "/work/users/d/y/dyy12/XuLab/regulatome/m23_sgrna_top25/out/matrices"
GS = np.load(f"{O}/ntc_gem_stats.npz", allow_pickle=True); sym = GS["genes"].astype(str); gp = pd.Series(np.arange(len(sym)), index=sym); gp = gp[~gp.index.duplicated(keep="last")]; col = gp.reindex(UG).values.astype(int)
KD = sp.load_npz(f"{O}/kd_cells.npz")[:, col].toarray().astype(np.float32); KO = pd.read_csv(f"{O}/kd_cells.tsv.gz", sep="\t")
Q = pd.read_csv(f"{M23}/top25_own/guide_qc.tsv", sep="\t"); KEPT = set(Q[(Q.kept == 1) & Q.kind.isin(["all", "construct"])].unit); dual = set(Q[Q.kind == "construct"].tf); cn = {}
for g in dual:
    for c, gt in enumerate(sorted(KO[KO.gene_target == g].guide_target.unique())): cn[(g, gt)] = f"{g}|construct{c + 1}"
KO["unit"] = [cn.get((g, gt), f"{g}|all") if g in dual else f"{g}|all" for g, gt in zip(KO.gene_target, KO.guide_target)]; KO["TF"] = [ALIAS.get(g, g) for g in KO.gene_target]
k_ = KO.unit.isin(KEPT).values & KO.TF.isin(TFS).values; KD, KO = KD[k_], KO[k_].reset_index(drop=True); assert KO.TF.nunique() == len(TFS)
MU = GS["mean"][:, col]; SD = np.sqrt(GS["var"][:, col]); gem = KO.gem.values
DX = KD - MU[gem]; ZC = np.where(SD[gem] > 0, DX / np.where(SD[gem] > 0, SD[gem], 1), np.where(DX == 0, 0., np.nan))  # per-cell z against the cell's own GEM; GEM SD 0 and value = mean -> 0, value != mean -> left out
print("cells", len(KO), "| share of cell x gene values with GEM SD 0:", float(np.isnan(ZC).mean()), flush=True)
U = {}
for u in sorted(KO.unit.unique()): U[u] = np.nanmean(ZC[KO.unit.values == u], 0)
P = {}
for u, v in U.items(): P.setdefault(ALIAS.get(u.split("|")[0], u.split("|")[0]), []).append(v)
SC = np.vstack([np.nanmean(P[t], 0) for t in TFS]); SC[~OK] = np.nan; nf = int((~np.isfinite(SC[OK])).sum()); print('TF x gene pairs without a score (all cells left out):', nf, flush=True); SC[OK & ~np.isfinite(SC)] = 0.
r_ = spearmanr(SC[OK], Z[OK]); print("sign agreement score vs z:", float((np.sign(SC[OK]) == np.sign(Z[OK])).mean()), "| Spearman score vs z:", r_.statistic, "| vs log2FC:", spearmanr(SC[OK], L1[OK]).statistic, flush=True)
A = np.abs(np.nan_to_num(SC)); VS = [("paper per-gene score", f">= {c}", OK & (A >= c), SC) for c in (0.1, 0.2, 0.3, 0.4, 0.5)]
RC = []
for v, c, M, S_ in VS:
    n = M.sum(1); dn = (M & (np.nan_to_num(S_) < 0)).sum(); rho = spearmanr(n, NC); tot = max(M.sum(), 1)
    d = {"version": v, "cutoff": c, "pairs": int(M.sum()), "down (%)": 100 * dn / tot, "TFs >= 1 responder": int((n >= 1).sum()), "TFs >= 10 responders": int((n >= 10).sum()), "median responders per TF": float(np.median(n)), "median |z| of pairs": float(np.median(AZ[M])) if M.any() else np.nan,
         "Spearman responders vs cells per TF": rho.statistic, "Spearman p": rho.pvalue, "median responders, TFs with 25-42 cells": float(np.median(n[TER == "25-42"])), "43-73 cells": float(np.median(n[TER == "43-73"])), "74-216 cells": float(np.median(n[TER == "74-216"]))}
    for k in range(10): d[f"decile {k + 1} (%)"] = 100 * M[:, DEX == k].sum() / tot
    RC.append(d)
RC = pd.DataFrame(RC); pd.set_option("display.width", 330); pd.set_option("display.max_columns", 40); print(RC.round(3).to_string(index=False), flush=True)
three = np.array([t in ("NRF1", "E4F1", "DNMT1") for t in TFS])
SETS = {"all 63 TFs": np.ones(len(TFS), bool), "55 non-broad TFs": np.array([t not in BROAD for t in TFS]), "without NRF1, E4F1, DNMT1": ~three}
ST, LOO = stage1(VS, 20261118, True)
print(ST[ST.padding == "10 kb"][["cutoff", "direction", "TF set", "TFs contributing", "Perturb-seq genes", "recall (%)", "shuffle mean (%)", "gap (points)", "observed / shuffle", "p (one-sided)", "precision (%)", "precision shuffle (%)"]].round(3).to_string(index=False))
print(LOO[LOO.direction == "all"].round(2).to_string(index=False), flush=True)
TF_ = pd.DataFrame(SC, index=TFS, columns=UG)
note = pd.DataFrame({"note": ["Paper per-gene score (X-Atlas/Orion z-normalised expression): for each top-25% knockdown cell, (log1p CP10k - NTC mean of that cell's GEM) / NTC SD of that cell's GEM (all NTC cells of the GEM), averaged over the TF unit's cells; two constructs averaged, HP1 = mean of CBX1/3/5; own gene removed; Stage 1 universe (NTC detection >= 10%).",
  "Stage 1: top 10,000 lab peaks per TF; promoter or canonical Hi-C ABC element + padding; matched gene shuffle within NTC expression x promoter ATAC deciles, exact hypergeometric per TF and stratum, 1,000 shuffles, seed 20261118, one run; one-sided p = (1 + #>= observed)/1001. Direction = sign of the score."]})
with pd.ExcelWriter(f"{O}/stage1_paper_score.xlsx") as w:
    RC.to_excel(w, sheet_name="responder counts", index=False); ST.to_excel(w, sheet_name="Stage 1", index=False); LOO.to_excel(w, sheet_name="leave one TF out 10 kb", index=False); note.to_excel(w, sheet_name="notes", index=False)
TF_.round(4).to_csv(f"{O}/paper_score_tf_by_gene.tsv.gz", sep="\t"); print("done")
