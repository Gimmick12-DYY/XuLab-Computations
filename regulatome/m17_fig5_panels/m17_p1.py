# m17_fig5_panels P1 (SPEC.md): TF x TF response matrix, target set of TF i vs knockdown of TF j. Diagnostic, benchmark already read.
# usage: m17_p1.py -> out/p1_entries.tsv, out/p1_summary.tsv, out/p1_ownrank.tsv, out/p1_heatmap_<set>_<matrix>.png/.svg
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import glob as _g, matplotlib.font_manager as _fm
for _f in _g.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): _fm.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Arial"; plt.rcParams["svg.fonttype"] = "none"
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; OUT = f"{RG}/m17_fig5_panels/out"; LAB = "diagnostic, benchmark already read"; SEED = 20261111; rng = np.random.default_rng(SEED)
OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
STRONG = ["CTCF", "DNMT1", "E4F1", "MEF2A", "NRF1", "PA2G4", "RBBP7", "RBPJ", "SAFB", "SETDB1"]
Rz = np.load(f"{RG}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True); TT = list(Rz["tfs"]); ug = pd.Index(Rz["genes"].astype(str)); n = len(ug); Z = Rz["r1"].astype(float).copy()
for i, t in enumerate(TT):
    for o in OWN.get(t, [t]):
        if o in ug: Z[i, ug.get_loc(o)] = np.nan
A = np.abs(Z); OK = np.isfinite(A)
RS = np.zeros_like(OK)
for j in range(len(TT)):
    idx = np.flatnonzero(OK[j]); RS[j, idx[np.argsort(-A[j, idx], kind="stable")[:int(round(len(idx) * .05))]]] = True
Ws = pd.read_csv(f"{RG}/m11_peak_regulome/out/wide_score_q0.05.tsv.gz", sep="\t", index_col=0).reindex(columns=ug).fillna(0)
def prom(t):
    f = f"{RG}/m10_promoter_regulome/work/counts/{t}.npz"
    if not os.path.exists(f): return None
    c = np.load(f, allow_pickle=True); gi = pd.Index(c["genes"].astype(str)).get_indexer(ug); v = np.where(gi >= 0, c["w500"][gi] / float(c["total"]) * 1e6, 0); return v
BIND = {"A_peak_score_x_abc": {t: Ws.loc[t].values for t in TT if t in Ws.index}, "B_raw_promoter_reads": {t: v for t in TT if (v := prom(t)) is not None}}
GLf = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t").drop_duplicates("name", keep="last").set_index("name"); acol = [c for c in GLf.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]
atac = GLf[acol].reindex(ug).fillna(0).values; GW = pd.read_csv(f"{RG}/m7_step2/splits/gene_split_wide.tsv", sep="\t"); BW = np.load(f"{RG}/m7_step2/features/raw_hicfixed_wide/base.npz"); cgi = pd.Series(BW["base"][:, 4], index=GW.gene).reindex(ug).fillna(0).values
strat = pd.qcut(pd.Series(atac).rank(method="first"), 10, labels=False).values * 2 + (cgi > 0); SI = {s: np.flatnonzero(strat == s) for s in np.unique(strat)}
def nmean(M, idx): S = M[:, idx]; return np.nanmean(S, 1)
def orv(tg):  # OR of set among top-5% responders, per column j, over genes finite for j
    a = (RS & tg).sum(1); b = (OK & ~RS & tg).sum(1); c = (RS & ~tg).sum(1); d = (OK & ~RS & ~tg).sum(1); return (a + .5) * (d + .5) / ((b + .5) * (c + .5))
ENT = []; MAT = {}
for bs, B in BIND.items():
    D = np.full((len(TT), len(TT)), np.nan); OR = D.copy(); NS = {}
    for i, t in enumerate(TT):
        if t not in B: continue
        p = B[t]; pos = np.flatnonzero(p > 0)
        if not len(pos): NS[t] = 0; continue
        top = pos[np.lexsort((rng.random(len(pos)), -p[pos]))][:1000]; tg = np.zeros(n, bool); tg[top] = True; NS[t] = len(top)
        cnt = {s: int(tg[SI[s]].sum()) for s in SI}; mm = []
        for _ in range(100):
            idx = np.concatenate([rng.choice(SI[s], min(c, len(SI[s])), replace=False) for s, c in cnt.items() if c]); mm.append(nmean(A, idx))
        D[i] = nmean(A, top) - np.mean(mm, 0); OR[i] = orv(tg)
    MAT[bs] = (D, OR, NS)
    for i, t in enumerate(TT):
        for j, u in enumerate(TT): ENT.append(dict(label=LAB, binding=bs, target_set_tf=t, knockdown_tf=u, n_set=NS.get(t, np.nan), D=D[i, j], OR=OR[i, j]))
E = pd.DataFrame(ENT); E.to_csv(f"{OUT}/p1_entries.tsv", sep="\t", index=False, float_format="%.5g")
def stdz(M, axis):
    mu = np.nanmean(M, axis, keepdims=True); sd = np.nanstd(M, axis, keepdims=True); sd[sd == 0] = np.nan; return (M - mu) / sd
SUM = []; OWNR = []
for bs, (D, OR, NS) in MAT.items():
    for mname, tfs in (("strong10", STRONG), ("all", TT)):
        ix = [TT.index(t) for t in tfs if t in TT]; Dm = D[np.ix_(ix, ix)]; ORm = np.log2(OR[np.ix_(ix, ix)]); names = [TT[k] for k in ix]
        rows_ok = np.isfinite(Dm).any(1)
        VERS = {"raw_difference": Dm, "column_z": stdz(Dm[rows_ok], 0), "row_z": stdz(Dm, 1), "log2_OR": ORm}
        VERS["column_z"] = np.vstack([VERS["column_z"][list(np.flatnonzero(rows_ok)).index(r)] if rows_ok[r] else np.full(len(ix), np.nan) for r in range(len(ix))])
        for vname, M in VERS.items():
            for drop in ((), ("NRF1", "E4F1")):
                keep = [k for k, t in enumerate(names) if t not in drop]; Mk = M[np.ix_(keep, keep)]; nk = [names[k] for k in keep]; r_ok = np.flatnonzero(np.isfinite(Mk).any(1))
                diag = np.array([Mk[r, r] for r in r_ok]); off = np.concatenate([np.delete(Mk[r], r) for r in r_ok]); obs = np.nanmean(diag) - np.nanmean(off)
                nul = []
                for _ in range(10000):
                    pr = rng.permutation(len(nk)); dg = np.array([Mk[r, pr[r]] for r in r_ok]); nul.append(np.nanmean(dg) - (np.nansum(Mk[r_ok]) - np.nansum(dg)) / (np.isfinite(Mk[r_ok]).sum() - np.isfinite(dg).sum()))
                own = {}
                for r in r_ok:
                    row = Mk[r]; own[nk[r]] = int(1 + (row[np.isfinite(row)] > row[r]).sum()) if np.isfinite(row[r]) else np.nan
                SUM.append(dict(label=LAB, binding=bs, matrix=mname, standardisation=vname, dropped=",".join(drop) or "none", n_rows=len(r_ok), n_cols=len(nk), diag_mean=float(np.nanmean(diag)), offdiag_mean=float(np.nanmean(off)), diag_minus_off=float(obs),
                                p_perm=float((1 + (np.array(nul) >= obs).sum()) / 10001), n_own_best=int(sum(v == 1 for v in own.values())), own_best_tfs=",".join(k for k, v in own.items() if v == 1)))
                if not drop:
                    for k, v in own.items(): OWNR.append(dict(label=LAB, binding=bs, matrix=mname, standardisation=vname, tf=k, own_rank=v, n_cols=len(nk), diag_value=Mk[nk.index(k), nk.index(k)]))
            if vname in ("raw_difference", "column_z"):
                lim = np.nanpercentile(np.abs(M), 99) or 1; big = len(names) > 20
                fig, ax = plt.subplots(figsize=(11, 10) if big else (5.2, 4.6)); im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=5 if big else 8); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=5 if big else 8)
                ax.set_xlabel("knockdown of TF j (HEK293T Perturb-seq, Arm 2)"); ax.set_ylabel(f"target set of TF i ({'q005 peak x ABC' if bs.startswith('A') else 'raw promoter reads'}, top 1,000)")
                cb = fig.colorbar(im, ax=ax, shrink=.7); cb.set_label("mean |z| set minus matched" if vname == "raw_difference" else "column z-score of that difference", fontsize=8)
                ax.set_title(f"{LAB}; {mname}, {vname}", fontsize=7); fig.tight_layout()
                for ext in ("png", "svg"): fig.savefig(f"{OUT}/p1_heatmap_{bs}_{mname}_{vname}.{ext}", dpi=200)
                plt.close(fig)
S = pd.DataFrame(SUM); S.to_csv(f"{OUT}/p1_summary.tsv", sep="\t", index=False, float_format="%.4g"); pd.DataFrame(OWNR).to_csv(f"{OUT}/p1_ownrank.tsv", sep="\t", index=False, float_format="%.4g")
print(S.drop(columns="label").to_string(index=False))
