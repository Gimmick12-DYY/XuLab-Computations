# B2: knockdown response, full transcriptome, two pre-specified arms for every TF. Cells: pass_guide_filter. Arm 1 "high-knockdown": summed sgRNA UMI >= the Q3 of Non-Targeting cells in the same GEM batch, for targeted and NTC cells alike; units with >= 50 cells after the cut. Arm 2 "all-cell": no cut; units with >= 50 cells before any cut.
# Per unit: mean log1p(CP10k) vs NTC from the same GEMs, weighted by the unit's cells per GEM; z with within-GEM NTC variance. Units: full + two replicate halves per TF; the two constructs separately for dual-pair TFs. Each arm is double-centred on its own kept full and construct units; halves use the same gene means. Units under the minimum are listed in the QC table and left out of the matrices.
# Reads only streamed cells, guide calls and phase-1 KD status; no network output.
import glob, json, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq, scipy.sparse as sp
ind, gene_meta, calls_f, guide_stats, status, out = sys.argv[1:7]; MIN_CELLS = 50
os.makedirs(out, exist_ok=True); gm = pd.read_parquet(gene_meta); G = int(gm.gene_token_id.max()) + 1; sym = np.array(["."] * G, dtype=object); sym[gm.gene_token_id.values] = gm.gene_name.values; gs = pd.read_csv(guide_stats, sep="\t"); dual = set(gs[gs.n_guide_pairs == 2].gene_symbol); panel = list(gs.gene_symbol); fs = sorted(glob.glob(f"{ind}/*.parquet")); print(len(fs), "files; dual", sorted(dual), flush=True)
Ca = pd.read_csv(calls_f, usecols=["cell_barcode", "num_umis"]); sgu = pd.Series(Ca.num_umis.astype(str).str.split("|").map(lambda v: sum(float(x) for x in v)).values, index=Ca.cell_barcode.values); del Ca; print("guide calls", len(sgu), flush=True)
def stats(N):
    m = np.asarray(N.mean(0)).ravel(); m2 = np.asarray(N.multiply(N).mean(0)).ravel(); n = N.shape[0]; return m.astype(np.float32), (np.maximum(m2 - m ** 2, 0) * n / max(n - 1, 1)).astype(np.float32), n
Mh, Vh, Nh, Ma, Va, Na, Det, gems, blocks, obs = [], [], [], [], [], [], [], [], [], []
for gi, f in enumerate(fs):
    t = pq.read_table(f, columns=["gene_token_id", "gene_expression", "cell_barcode", "sample", "guide_target", "gene_target", "total_counts", "pass_guide_filter"]); o = t.select(["cell_barcode", "sample", "guide_target", "gene_target", "total_counts", "pass_guide_filter"]).to_pandas(); assert o["sample"].nunique() == 1, f; a = t["gene_token_id"].combine_chunks(); b = t["gene_expression"].combine_chunks(); ptr = a.offsets.to_numpy(); j = a.values.to_numpy()[ptr[0]:ptr[-1]]; x = b.values.to_numpy().astype(np.float32)[ptr[0]:ptr[-1]]; ptr = ptr - ptr[0]; X_ = sp.csr_matrix((x, j.astype(np.int32), ptr.astype(np.int64)), shape=(len(o), G)); sg = sgu.reindex(o.cell_barcode.values).values; ok = (o.pass_guide_filter.values == 1) & np.isfinite(sg); ntc = ok & (o.gene_target.values == "Non-Targeting"); assert ntc.sum() >= 4, (f, ntc.sum()); thr = np.quantile(sg[ntc], .75); hi = sg >= thr
    L_ = (sp.diags(1e4 / o.total_counts.values.astype(np.float32)) @ X_).tocsr(); L_.data = np.log1p(L_.data); m, v, n = stats(L_[ntc & hi]); Mh.append(m); Vh.append(v); Nh.append(n); m, v, n = stats(L_[ntc]); Ma.append(m); Va.append(v); Na.append(n); Det.append(np.asarray((L_[ntc] > 0).sum(0)).ravel()); gems.append(o["sample"].iloc[0]); keep = ok & ~(o.gene_target.values == "Non-Targeting"); blocks.append(L_[keep]); obs.append(o.loc[keep, ["cell_barcode", "gene_target", "guide_target"]].assign(gem=gi, top25=hi[keep], sg_umi=sg[keep])); print(os.path.basename(f), len(o), int(ntc.sum()), Nh[-1], int(keep.sum()), flush=True)
Mh, Vh, Nh, Ma, Va, Na = np.vstack(Mh), np.vstack(Vh), np.array(Nh), np.vstack(Ma), np.vstack(Va), np.array(Na); det = np.sum(Det, axis=0) / Na.sum(); L = sp.vstack(blocks).tocsr(); O = pd.concat(obs, ignore_index=True); cg = np.flatnonzero((det >= .01) & (sym != ".")); print("targeted cells", L.shape, "genes kept", len(cg), flush=True); rng = np.random.default_rng(293)
def halves(ix):
    h1, h2 = [], []
    for _, sub in O.iloc[ix].groupby("gem").groups.items():
        p = rng.permutation(np.array(sub)); h1 += list(p[::2]); h2 += list(p[1::2])
    return (np.array(h1, dtype=int), np.array(h2, dtype=int)) if rng.random() < .5 else (np.array(h2, dtype=int), np.array(h1, dtype=int))
def r_(a, b):
    k = np.isfinite(a) & np.isfinite(b); return float(np.corrcoef(a[k], b[k])[0, 1]) if k.sum() > 2 and a[k].std() > 0 and b[k].std() > 0 else np.nan
S_ = pd.read_csv(status, sep="\t").set_index("gene_symbol"); gpos = {s: i for i, s in enumerate(sym[cg])}
def resp(ix, M_, V_, N_):
    mp = np.asarray(L[ix].mean(0)).ravel(); ng = np.bincount(O.gem.values[ix], minlength=len(M_)).astype(float); n = ng.sum(); ref = (ng[:, None] * M_).sum(0) / n; var = ((ng / n ** 2 + ng ** 2 / n ** 2 / np.maximum(N_, 1))[:, None] * V_).sum(0); return mp - ref, (mp - ref) / np.sqrt(var + 1e-6)
SUM = dict(n_files=len(fs), n_genes=len(cg), dual_tfs=sorted(dual), targeted_cells=int(len(O)), targeted_after_cut=int(O.top25.sum()), ntc_all=int(Na.sum()), ntc_after_cut=int(Nh.sum())); pd.DataFrame(dict(gem=gems, ntc_after_cut=Nh, ntc_all=Na)).to_csv(f"{out}/gem_ntc_counts.tsv", sep="\t", index=False)
def arm(name, M_, V_, N_, cut):
    units = []
    def sel(ix): return ix[O.top25.values[ix]] if cut else ix
    def add(u, g, ty, pr, ixa):
        use = sel(ixa); units.append(dict(unit=u, tf=g, kind=ty, guide_pair=pr, n_cells_all=len(ixa), n_cells_after_cut=int(O.top25.values[ixa].sum()), n_used=len(use), kept=int(len(use) >= MIN_CELLS), ix=use)); return use
    for g in panel:
        ix = np.flatnonzero(O.gene_target.values == g)
        if g in dual:
            for k, (pr, sub) in enumerate(sorted(O.iloc[ix].groupby("guide_target").groups.items())): add(f"{g}|construct{k + 1}", g, "construct", pr, np.array(sub))
        else:
            use = add(f"{g}|all", g, "all", ".", ix); h1, h2 = halves(use); kp = units[-1]["kept"]; na, nc = units[-1]["n_cells_all"], units[-1]["n_cells_after_cut"]
            for nm, h in (("half1", h1), ("half2", h2)): units.append(dict(unit=f"{g}|{nm}", tf=g, kind="half", guide_pair=".", n_cells_all=na, n_cells_after_cut=nc, n_used=len(h), kept=kp, ix=h))
    rows, LF, Z = [], [], []
    for u in units:
        ix = u.pop("ix"); g = u["tf"]; tj = np.flatnonzero(sym == g); r = dict(u, n_gems=int(O.gem.iloc[ix].nunique()) if len(ix) else 0, median_sg_umi=float(O.sg_umi.iloc[ix].median()) if len(ix) else np.nan, ontarget_logfc=np.nan, ontarget_z=np.nan, ontarget_in_kept_genes=g in gpos, phase1_kd_q=S_.kd_q_bh.get(g, np.nan), phase1_kd_pct=S_.knockdown_pct.get(g, np.nan))
        if u["kept"]:
            d, z = resp(ix, M_, V_, N_); LF.append(d[cg].astype(np.float32)); Z.append(z[cg].astype(np.float32))
            if len(tj): r["ontarget_logfc"], r["ontarget_z"] = float(d[tj[0]]), float(z[tj[0]])
        rows.append(r)
    Q = pd.DataFrame(rows); K = Q[Q.kept == 1].reset_index(drop=True); O_ = f"{out}/{name}"; os.makedirs(O_, exist_ok=True); Q["replicate_r"] = np.nan
    if len(K):
        Z = np.vstack(Z); LF = np.vstack(LF); prim = K.kind.isin(["all", "construct"]).values; gmean = Z[prim].mean(0); R = Z - gmean; R = R - R.mean(1, keepdims=True); idx = {u: i for i, u in enumerate(K.unit)}; qi = {u: i for i, u in enumerate(Q.unit)}
        for g in panel:
            a_ = [u for u in idx if u.startswith(f"{g}|construct")] if g in dual else [u for u in (f"{g}|half1", f"{g}|half2") if u in idx]
            if len(a_) >= 2:
                for u in a_ + ([f"{g}|all"] if g not in dual else []): Q.loc[qi[u], "replicate_r"] = r_(R[idx[a_[0]]], R[idx[a_[1]]])
        np.savez_compressed(f"{O_}/kd_response.npz", units=K.unit.values.astype(str), genes=sym[cg].astype(str), response=R.astype(np.float32), z=Z, logfc=LF, gene_mean_primary=gmean.astype(np.float32))
    Q.to_csv(f"{O_}/guide_qc.tsv", sep="\t", index=False, float_format="%.4g"); P = Q[Q.kind.isin(["all", "construct"])]; Kp = P[P.kept == 1]; SUM[name] = dict(tfs_kept=int(Kp.tf.nunique()), units_kept=int(len(Kp)), tfs_with_all_units_kept=int((P.groupby("tf").kept.min() == 1).sum()), tfs_dropped=sorted(set(panel) - set(Kp.tf)), units_dropped=sorted(P[P.kept == 0].unit.tolist()), median_cells_used=float(Kp.n_used.median()) if len(Kp) else None, median_replicate_r=float(Kp.replicate_r.median()) if len(Kp) else None); print(name, json.dumps(SUM[name]), flush=True)
arm("arm1_high_knockdown", Mh, Vh, Nh, True); arm("arm2_all_cell", Ma, Va, Na, False); both = set(pd.read_csv(f"{out}/arm1_high_knockdown/guide_qc.tsv", sep="\t").query("kept == 1 and kind != 'half'").tf) & set(pd.read_csv(f"{out}/arm2_all_cell/guide_qc.tsv", sep="\t").query("kept == 1 and kind != 'half'").tf); SUM["tfs_in_both_arms"] = len(both); json.dump(SUM, open(f"{out}/summary.json", "w"), indent=1); print(json.dumps({k: SUM[k] for k in ("n_genes", "targeted_cells", "targeted_after_cut", "ntc_all", "ntc_after_cut", "tfs_in_both_arms")}), flush=True)
