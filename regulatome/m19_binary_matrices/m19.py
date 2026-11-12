# m19_binary_matrices (SPEC + Amendment 1): binary Regulome TF x gene matrix vs binary Perturb-seq responder matrix, every declared combination. Diagnostic, benchmark-developed.
import sys
import numpy as np, pandas as pd
from scipy.stats import norm, false_discovery_control
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import glob as _g, matplotlib.font_manager as _fm
for _f in _g.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): _fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "font.size": 6})
X_ = "/work/users/d/y/dyy12/XuLab"; RG_ = f"{X_}/regulatome"; O_ = f"{RG_}/m19_binary_matrices"; OUT_ = f"{O_}/out"; LAB_ = "diagnostic, benchmark-developed"; SEED_ = 20261112; rng_ = np.random.default_rng(SEED_)
STRONG = ["CTCF", "DNMT1", "E4F1", "MEF2A", "NRF1", "PA2G4", "RBBP7", "RBPJ", "SAFB", "SETDB1"]
# m15 halves (raw per-TF z) for P4
sys.argv = ["m15.py", "raw"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); exec(src[:src.index("# ---- binding percentiles")])
H1 = pd.DataFrame(Z1, index=TF, columns=UG); H2 = pd.DataFrame(Z2, index=TF, columns=UG)
IX = np.load(f"{RG_}/m18_ml_v2/features/index.npz", allow_pickle=True); TT = list(IX["tfs"].astype(str)); ug = IX["genes"].astype(str); NG = len(ug); nT = len(TT)
Rz = np.load(f"{RG_}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True); assert list(Rz["tfs"].astype(str)) == TT; Z = Rz["r1"].astype(float).copy()
OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}; gi = {g: k for k, g in enumerate(ug)}
for i, t in enumerate(TT):
    for o in OWN.get(t, [t]):
        if o in gi: Z[i, gi[o]] = np.nan
OK = np.isfinite(Z)
z1 = H1.reindex(index=TT, columns=ug).values; z2 = H2.reindex(index=TT, columns=ug).values
def feats(v):
    d = np.load(f"{RG_}/m18_ml_v2/features/pair_{v}.npz", allow_pickle=True); n = list(d["names"]); A = d["X"].reshape(nT, NG, len(n)); return {k: A[:, :, n.index(k)].astype(float) for k in ("B_prom_peak", "B_distal_abc_peak", "B_pct_peak_score")}
def bmats(F):
    anyp = (F["B_prom_peak"] > 0) | (F["B_distal_abc_peak"] > 0); B = {"B1_any_linked_peak": anyp, "B2_promoter_peak": F["B_prom_peak"] > 0}
    for N in (500, 1000, 2000):
        M = np.zeros((nT, NG), bool)
        for i in range(nT):
            s = np.where(anyp[i], F["B_pct_peak_score"][i], -1); pos = np.flatnonzero(s > 0); o = pos[np.lexsort((rng_.random(len(pos)), -s[pos]))][:N]; M[i, o] = True
        B[f"B3_top{N}_peakxABC"] = M
    return B
BR, BS, BP = bmats(feats("real")), bmats(feats("swap")), bmats(feats("pseudo0"))
# Perturb-seq matrices
P = {}
a = np.abs(np.nan_to_num(Z)); top5 = np.zeros((nT, NG), bool)
for i in range(nT):
    idx = np.flatnonzero(OK[i]); top5[i, idx[np.argsort(-a[i, idx], kind="stable")[:int(round(len(idx) * .05))]]] = True
P["P1_top5pct"] = top5; P["P2_absz_ge3"] = OK & (a >= 3)
bh = np.zeros((nT, NG), bool)
for i in range(nT):
    idx = np.flatnonzero(OK[i]); q = false_discovery_control(2 * norm.sf(np.abs(Z[i, idx]))); bh[i, idx[q < .05]] = True
P["P3_BH_q05"] = bh; P["P4_reproducible_halves"] = OK & np.isfinite(z1) & np.isfinite(z2) & (np.abs(z1) >= 2) & (np.abs(z2) >= 2) & (np.sign(z1) == np.sign(z2))
DIRS = {"all": np.ones((nT, NG), bool), "down": np.nan_to_num(Z) < 0, "up": np.nan_to_num(Z) > 0}
# responder counts first
CNT = pd.DataFrame({f"{p}|{d}": (P[p] & DIRS[d]).sum(1) for p in P for d in DIRS}, index=TT); CNT.insert(0, "strong_phenotype", [t in STRONG for t in TT]); CNT.to_csv(f"{OUT_}/responder_counts.tsv", sep="\t")
print(CNT.loc[STRONG].to_string()); print("TFs with >= 20 responders:", (CNT.drop(columns="strong_phenotype") >= 20).sum().to_string())
# strata for matched genes
gA = np.load(f"{RG_}/m18_ml_v2/features/gene_A.npz", allow_pickle=True); an = list(gA["names"]); atac = gA["X"][:, an.index("A_prom_atac")]; cgi = gA["X"][:, an.index("A_cgi")]
strat = pd.qcut(pd.Series(atac).rank(method="first"), 10, labels=False).values * 2 + (cgi > 0); SI = {s: np.flatnonzero(strat == s) for s in np.unique(strat)}
def tab(b, r, ok): return np.array([(b & r & ok).sum(), (b & ~r & ok).sum(), (~b & r & ok).sum(), (~b & ~r & ok).sum()], float)
def orj(t): return (t[0] + .5) * (t[3] + .5) / ((t[1] + .5) * (t[2] + .5)), t[0] / max(t[0] + t[1] + t[2], 1)
def boot(t):
    m = rng_.multinomial(int(t.sum()), t / t.sum(), 2000).astype(float); o = (m[:, 0] + .5) * (m[:, 3] + .5) / ((m[:, 1] + .5) * (m[:, 2] + .5)); j = m[:, 0] / np.maximum(m[:, :3].sum(1), 1)
    return np.percentile(o, [2.5, 97.5]), np.percentile(j, [2.5, 97.5])
def matched(b, r, ok, nd=200):
    cnt = {s: int((b & ok)[SI[s]].sum()) for s in SI}; res = []
    for _ in range(nd):
        m = np.zeros(NG, bool)
        for s, c in cnt.items():
            if c: pool = SI[s][ok[SI[s]]]; m[rng_.choice(pool, min(c, len(pool)), replace=False)] = True
        res.append(orj(tab(m, r, ok))[0])
    return float(np.median(res))
def mh(tabs):
    num = den = P_ = Q_ = S_ = 0.0
    for a_, b_, c_, d_ in tabs:
        N = a_ + b_ + c_ + d_
        if N == 0: continue
        rr, ss = a_ * d_ / N, b_ * c_ / N; num += rr; den += ss; p_, q_ = (a_ + d_) / N, (b_ + c_) / N; P_ += p_ * rr; Q_ += p_ * ss + q_ * rr; S_ += q_ * ss
    if num == 0 or den == 0: return np.nan, np.nan, np.nan
    o = num / den; se = np.sqrt(P_ / (2 * num ** 2) + Q_ / (2 * num * den) + S_ / (2 * den ** 2)); return o, o * np.exp(-1.96 * se), o * np.exp(1.96 * se)
ROWS = []; POOL = []; GRID = []; KEEP = {}
for bn in BR:
    for pn in P:
        for dn in DIRS:
            R_ = P[pn] & DIRS[dn]; T_ = {}
            for i, t in enumerate(TT):
                ok = OK[i]; nres = int(R_[i].sum()); row = dict(label=LAB_, B=bn, P=pn, direction=dn, tf=t, strong_phenotype=t in STRONG, n_list=int((BR[bn][i] & ok).sum()), n_responders=nres)
                if nres < 20 or row["n_list"] == 0: row["status"] = "too few responders" if nres < 20 else "empty list"; ROWS.append(row); continue
                tb = tab(BR[bn][i], R_[i], ok); T_[t] = tb; o, j = orj(tb); ci, cj = boot(tb)
                row.update(status="ok", n_both=int(tb[0]), OR=o, OR_lo=ci[0], OR_hi=ci[1], jaccard=j, jac_lo=cj[0], jac_hi=cj[1], OR_swapped_tf=orj(tab(BS[bn][i], R_[i], ok))[0] if BS[bn][i].any() else np.nan,
                           OR_pseudo_tf=orj(tab(BP[bn][i], R_[i], ok))[0], OR_matched_genes=matched(BR[bn][i], R_[i], ok)); ROWS.append(row)
            for grp, names in (("strong10", STRONG), ("other66", [t for t in TT if t not in STRONG])):
                tl = [T_[t] for t in names if t in T_]; o, lo, hi = mh(tl); POOL.append(dict(label=LAB_, B=bn, P=pn, direction=dn, group=grp, n_tfs_tested=len(tl), MH_OR=o, lo=lo, hi=hi))
            # grid over TFs with enough responders
            ti = [i for i, t in enumerate(TT) if t in T_]
            if len(ti) >= 3:
                Bm = BR[bn][ti].astype(float); Rm = (R_[ti] & OK[ti]).astype(float); Om = OK[ti].astype(float)
                n11 = Bm @ Rm.T; nb = Bm @ Om.T; nr = Rm.sum(1)[None, :] * np.ones((len(ti), 1)); ntot = Om.sum(1)[None, :] * np.ones((len(ti), 1))
                n10 = nb - n11; n01 = nr - n11; n00 = ntot - n11 - n10 - n01; G = (n11 + .5) * (n00 + .5) / ((n10 + .5) * (n01 + .5))
                def dstat(M): d = np.diag(M); off = M[~np.eye(len(M), dtype=bool)]; return d.mean() - off.mean()
                obs = dstat(G); obsl = dstat(np.log(G)); nul = []; null = []
                for _ in range(10000):
                    pr_ = rng_.permutation(len(ti)); Gp = G[pr_]; nul.append(dstat(Gp)); null.append(dstat(np.log(Gp)))
                own_best = int(sum(np.argmax(G[k]) == k for k in range(len(ti))))
                GRID.append(dict(label=LAB_, B=bn, P=pn, direction=dn, n_tfs=len(ti), diag_minus_offdiag_OR=obs, p_perm=(1 + (np.array(nul) >= obs).sum()) / 10001, diag_minus_offdiag_logOR=obsl, p_perm_log=(1 + (np.array(null) >= obsl).sum()) / 10001, n_own_best=own_best, own_best_tfs=",".join(TT[ti[k]] for k in range(len(ti)) if np.argmax(G[k]) == k)))
                if bn == "B1_any_linked_peak" and pn == "P4_reproducible_halves" and dn == "all": KEEP["grid"] = (G, [TT[k] for k in ti])
            print(bn, pn, dn, len(T_), flush=True)
A = pd.DataFrame(ROWS); A.to_csv(f"{OUT_}/all_combinations_per_tf.tsv", sep="\t", index=False, float_format="%.4g")
PO = pd.DataFrame(POOL); PO.to_csv(f"{OUT_}/pooled_MH.tsv", sep="\t", index=False, float_format="%.4g"); GR = pd.DataFrame(GRID); GR.to_csv(f"{OUT_}/grid_specificity.tsv", sep="\t", index=False, float_format="%.4g")
# per-combination summary
S = A[A.status == "ok"].groupby(["B", "P", "direction"]).apply(lambda d: pd.Series(dict(n_tfs_tested=len(d), n_strong_tested=int(d.strong_phenotype.sum()), median_OR=d.OR.median(), median_OR_matched=d.OR_matched_genes.median(), median_OR_swapped=d.OR_swapped_tf.median(), median_OR_pseudo=d.OR_pseudo_tf.median(),
    n_OR_lo_above_1=int((d.OR_lo > 1).sum()), frac_OR_gt_matched=float((d.OR > d.OR_matched_genes).mean()), median_jaccard=d.jaccard.median())), include_groups=False).reset_index()
S.insert(0, "label", LAB_); S.to_csv(f"{OUT_}/summary_by_combination.tsv", sep="\t", index=False, float_format="%.4g"); print(S.drop(columns="label").round(3).to_string(index=False))
# figure: B1 x P4 (all), 10 strong TFs
sel = [t for t in STRONG]; si = [TT.index(t) for t in sel]; Bm = BR["B1_any_linked_peak"][si] & OK[si]; Rm = P["P4_reproducible_halves"][si]
cols = np.flatnonzero(Rm.any(0) | Bm.any(0)); o = np.lexsort((-Bm[:, cols].sum(0), -Rm[:, cols].sum(0))); cols = cols[o]
fig = plt.figure(figsize=(180 / 25.4, 78 / 25.4))
for k, (M, ttl) in enumerate(((Bm, "Regulome (B1): q005 peak in an\nABC-linked element"), (Rm, "Perturb-seq (P4): |z| >= 2 in both\nhalves of cells, same sign"))):
    ax = fig.add_axes([.075 + k * .28, .17, .25, .66]); ax.imshow(M[:, cols], aspect="auto", interpolation="nearest", cmap="Greys"); ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel if k == 0 else []); ax.set_xticks([])
    ax.set_title(ttl, fontsize=6)
fig.text(.075, .1, f"Columns: the {len(cols):,} genes called in either matrix for these 10 TFs, ordered by number of responder calls, then peak calls (same order in both panels)", fontsize=6)
if "grid" in KEEP:
    G, nm = KEEP["grid"]; keep = [nm.index(t) for t in sel if t in nm]; Gs = np.log2(G[np.ix_(keep, keep)]); lab = [nm[k] for k in keep]
    ax = fig.add_axes([.72, .25, .19, .58]); lim = np.nanmax(np.abs(Gs)); im = ax.imshow(Gs, cmap="RdBu_r", vmin=-lim, vmax=lim); ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=90); ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
    ax.set_xlabel("responders of TF j (P4)", labelpad=2); ax.set_ylabel("Regulome list of TF i (B1)", labelpad=2); cb = fig.colorbar(im, ax=ax, fraction=.05, pad=.03); cb.set_label("log2 OR")
    ax.set_title(f"OR grid ({len(lab)} strong TFs\nwith >= 20 responders)", fontsize=6)
for ext in ("png", "svg"): fig.savefig(f"{OUT_}/fig_B1xP4_strong10.{ext}", dpi=300)
print("done")
