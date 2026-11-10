# Step 4 (reversed design): sequence defines the sets. For each TF and known motif, the imputed signal (CPM minus 79-TF mean, rank-normal
# per TF) is residualized on strata (GC 0.025 x CpG o/e quintile x promoter), then the mean residual at motif-hit bins (FIMO p <= 1e-4) is
# turned into z = mean * sqrt(n_hit) / sd. Computed separately on odd (discovery) and even + X (validation) chromosomes.
# A motif passes for a TF at strict (z_disc >= 4 and z_val >= 2) or loose (z_disc >= 3 and z_val >= 1.5). Generic = passes in > 20 TFs.
# Pair: shared non-generic passing motifs; empirical p vs 200 random panel pairs matched on depth (cell tertiles) and correlation pct bin.
# Output: reversed/z_disc.tsv, z_val.tsv, own_motif_check.tsv, pairs.tsv, rates.tsv, controls.tsv
import os, sys, gzip, glob
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import norm, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import summarize as S
X = "/work/users/d/y/dyy12/XuLab"; O = os.path.dirname(os.path.abspath(__file__)); R = f"{O}/reversed"; rng = np.random.default_rng(20261110)
z = np.load(f"{X}/complex_screen/motif_shared_20261103/fg/signal_mask.npz"); Sg = z["S"]; tfs = [t.upper() for t in z["tfs"]]; D = Sg - Sg.mean(0)
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t"); num = Q.chrom.str.replace("chr", "")
part = np.where(num.isin([str(i) for i in range(1, 23, 2)]), "disc", np.where(num.isin([str(i) for i in range(2, 23, 2)] + ["X"]), "val", "none"))
ok = Q.n_ok.values & ~Q.blacklist.values & (Q.exonic.values <= 0.5) & (part != "none"); ei = np.flatnonzero(ok)
prom = np.zeros(len(Q), bool); prom[[int(x) for x in open(f"{O}/ref/promoter_bins.txt").read().split()]] = True
cq = np.quantile(Q.cpg_oe.values[ei], [0.2, 0.4, 0.6, 0.8]); strat = (np.floor(Q.gc.values / 0.025).astype(int) * 10 + np.searchsorted(cq, Q.cpg_oe.values)) * 2 + prom
pos = {f"b{b}": i for i, b in enumerate(Q.bin.values[ei])}; motifs = []; rows, cols = [], []
for f in sorted(glob.glob(f"{R}/fimo/c*.tsv.gz")):
    for l in gzip.open(f, "rt"):
        m, s = l.rstrip("\n").split("\t")
        if s in pos: rows.append(pos[s]); cols.append(m)
mid = {m: k for k, m in enumerate(sorted(set(cols)))}; motifs = sorted(mid, key=mid.get)
H = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, [mid[c] for c in cols])), shape=(len(ei), len(motifs))); H.data[:] = 1
print(len(motifs), "motifs with hits;", H.nnz, "hits", flush=True)
Y = np.apply_along_axis(lambda v: norm.ppf((rankdata(v) - 0.5) / len(v)), 1, D[:, ei]).T  # bins x TFs
st = strat[ei]; pe = part[ei]; Zs = {}
for pt in ["disc", "val"]:
    sel = pe == pt; Ys = Y[sel].copy(); ss = st[sel]
    for s in np.unique(ss): q = ss == s; Ys[q] -= Ys[q].mean(0)
    Hs = H[sel]; n = np.asarray(Hs.sum(0)).ravel(); sd = Ys.std(0)
    Zs[pt] = pd.DataFrame((Hs.T @ Ys) / np.maximum(n, 1)[:, None] * np.sqrt(np.maximum(n, 1))[:, None] / sd[None, :], index=motifs, columns=tfs)
    Zs[pt][n < 50] = np.nan
Zs["disc"].round(2).to_csv(f"{R}/z_disc.tsv", sep="\t"); Zs["val"].round(2).to_csv(f"{R}/z_val.tsv", sep="\t")
alt = {}
for l in open(f"{O}/db/known.meme"):
    if l.startswith("MOTIF "): x = l.split(); alt[x[1]] = x[2] if len(x) > 2 else x[1]
def name(m): return f"{m}|{alt.get(m, m)}"
JOWN = {"CTCF": "CTCF", "NRF1": "NRF1", "NFYA": "NFYA", "MAZ": "MAZ", "E2F3": "E2F3", "SOX4": "SOX4", "ZNF143": "ZNF143", "MLX": "MLX", "SREBF2": "SREBF2", "HES1": "HES1", "FOXK1": "FOXK1", "DDIT3": "DDIT3", "TBX3": "TBX3", "RELA": "RELA", "EPAS1": "EPAS1", "ZIC2": "ZIC2", "MEF2A": "MEF2A", "E4F1": "E4F1"}
own = []
for t, g in JOWN.items():
    c = [m for m in motifs if alt.get(m, "").upper() == g or m.upper().startswith(g + ".") or m.upper() == g]
    if not c: continue
    zd = Zs["disc"][t]; rk = zd.rank(ascending=False); b = max(c, key=lambda m: zd.get(m, -99) if not np.isnan(zd.get(m, np.nan)) else -99)
    own.append(dict(tf=t, own=name(b), z_disc=round(zd[b], 2), z_val=round(Zs["val"].loc[b, t], 2), rank=int(rk[b]) if not np.isnan(rk[b]) else -1, n_motifs=int(zd.notna().sum()), top=name(zd.idxmax()), top_z=round(zd.max(), 2)))
OWN = pd.DataFrame(own); OWN.to_csv(f"{R}/own_motif_check.tsv", sep="\t", index=False); print(OWN.to_string(), flush=True)
LV = {"strict": (4, 2), "loose": (3, 1.5)}; P_ = {lv: (Zs["disc"] >= a) & (Zs["val"] >= b) for lv, (a, b) in LV.items()}
gen = {lv: P_[lv].sum(1) > 20 for lv in LV}; P2 = {lv: P_[lv] & ~gen[lv].values[:, None] for lv in LV}
print({lv: int(gen[lv].sum()) for lv in LV}, "generic motifs;", {lv: int(P2[lv].values.sum()) for lv in LV}, "TF-motif passes", flush=True)
R1 = pd.read_csv(f"{X}/regulatome/r1_tftf/v2/r1v2_pairs_provisional.tsv", sep="\t"); R1 = R1[R1.a.isin(tfs) & R1.b.isin(tfs)].copy()
R1["pct"] = R1[["pearson_domain_pct", "spqn_noquies_pct"]].max(1); R1["pair"] = R1.a + "-" + R1.b; R1["novel"] = R1.string_physical.isna() & ~R1.capselex.astype(str).str.startswith("called")
cells = pd.read_csv(f"{X}/complex_screen/pair_motif_lookup/tf_cells.tsv", sep="\t").set_index("tf").n_cells; tert = np.quantile(cells.values, [1 / 3, 2 / 3])
cb = {t: int(np.searchsorted(tert, cells[t])) if t in cells.index else 3 for t in tfs}
R1["dbin"] = [tuple(sorted((cb[a], cb[b]))) for a, b in zip(R1.a, R1.b)]; R1["pbin"] = pd.cut(R1.pct, [-1, .6, .8, .9, 1.01], labels=False)
for lv in LV:
    M = P2[lv]; R1[f"n_{lv}"] = [int((M[a] & M[b]).sum()) for a, b in zip(R1.a, R1.b)]
    R1[f"shared_{lv}"] = [";".join(f"{name(m)}(zA={Zs['disc'].loc[m, a]:.1f}/{Zs['val'].loc[m, a]:.1f},zB={Zs['disc'].loc[m, b]:.1f}/{Zs['val'].loc[m, b]:.1f},nTF={int(P_[lv].loc[m].sum())})" for m in M.index[(M[a] & M[b]).values]) for a, b in zip(R1.a, R1.b)]
    def emp(r):
        pool = R1[(R1.dbin == r.dbin) & (R1.pbin == r.pbin) & (R1.pair != r.pair)]
        if len(pool) < 20: pool = R1[(R1.dbin == r.dbin) & (R1.pair != r.pair)]
        return (1 + np.sum(rng.choice(pool[f"n_{lv}"].values, 200) >= r[f"n_{lv}"])) / 201 if r[f"n_{lv}"] else 1.0
    R1[f"emp_{lv}"] = [emp(r) for _, r in R1.iterrows()]
af3 = {}
for f in glob.glob(f"{X}/complex_screen/af3_corr060_20261104/scores_s1/*.tsv"):
    x = open(f).read().split("\t"); a_, b_ = x[0].split("__"); af3[tuple(sorted((a_, b_)))] = (x[4], float(x[1]), float(x[2]))
for f, ti, ii in [(f"{X}/complex_screen/af3_novel_20261103/af3_scores_weak.tsv", 9, 7), (f"{X}/complex_screen/af3_novel_20261103/af3_scores.tsv", 9, 7)]:
    for l in open(f):
        x = l.rstrip("\n").split("\t")
        if x[0] != "pair": a_, b_ = [y.replace("CBX3", "HP1").replace("CBX1", "HP1").replace("ZUP1", "ZUFSP") for y in x[0].split("__")]; af3[tuple(sorted((a_, b_)))] = (x[ti], float(x[1]), float(x[ii]))
R1["af3_tier"], R1["af3_iptm"], R1["af3_ipsae"] = zip(*[af3.get(tuple(sorted((a, b))), ("not_run", np.nan, np.nan)) for a, b in zip(R1.a, R1.b)])
cols = ["pair", "pct", "pearson_domain_pct", "spqn_noquies_pct", "novel", "af3_tier", "af3_iptm", "af3_ipsae", "n_strict", "emp_strict", "n_loose", "emp_loose", "shared_strict", "shared_loose"]
rk = {"strong": 0, "weak": 1}; R1["_a"] = R1.af3_tier.map(rk).fillna(2)
R1.sort_values(["_a", "emp_strict", "emp_loose", "pct"], ascending=[True, True, True, False])[cols].to_csv(f"{R}/pairs.tsv", sep="\t", index=False)
CON = ["CTCF-STAG2", "MAZ-ZNF143", "NFYA-ZNF644", "EZH2-MTF2", "RBBP7-SRCAP"]; C = R1[R1.pair.isin(CON) | (R1.b + "-" + R1.a).isin(CON)][cols]; C.to_csv(f"{R}/controls.tsv", sep="\t", index=False)
nv = R1[R1.novel]; rates = []
for lab, s in [("novel pct>=0.90", nv[nv.pct >= .9]), ("novel 0.80-0.90", nv[(nv.pct >= .8) & (nv.pct < .9)]), ("novel 0.60-0.80", nv[(nv.pct >= .6) & (nv.pct < .8)]), ("novel <0.60", nv[nv.pct < .6]), ("all", R1)]:
    rates.append(dict(set=lab, n=len(s), any_strict=round((s.n_strict > 0).mean(), 3), emp_strict_le05=round((s.emp_strict <= .05).mean(), 3), any_loose=round((s.n_loose > 0).mean(), 3), emp_loose_le05=round((s.emp_loose <= .05).mean(), 3)))
print(pd.DataFrame(rates).to_string()); pd.DataFrame(rates).to_csv(f"{R}/rates.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120); print(C[["pair", "n_strict", "emp_strict", "n_loose", "emp_loose", "shared_strict"]].to_string())
s = R1[R1.novel & (R1.pct >= .6) & ((R1.emp_strict <= .05) | (R1.emp_loose <= .05))].sort_values(["_a", "emp_strict"]); print(len(s), "novel pct>=0.6 pairs with emp<=0.05"); print(s[["pair", "pct", "af3_tier", "af3_ipsae", "n_strict", "emp_strict", "n_loose", "emp_loose", "shared_strict"]].head(30).to_string())
