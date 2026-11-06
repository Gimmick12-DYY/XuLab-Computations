# HOX-cluster / PRC2-target pseudobulk check. Are HOX clusters empty in the EZH2/MTF2/PCGF6 libraries (method/tag problem) or
# enriched (signal real but diluted genome-wide)?
#   step "bins <TF>": pseudobulk raw read 5' ends per 1 kb genome bin (v3 grid) -> hox_check/bins/<TF>.npy
#   step "analyze": regions = HOXA-D clusters (GENCODE v46 HOX gene span +/- 50 kb) and a canonical PRC2 target set
#     (promoters, TSS +/- 2.5 kb, of protein-coding genes whose TSS +/- 2.5 kb overlaps a SUZ12 HEK293T IDR peak, ENCFF708RSP).
#     Per TF: reads per kb per million (RPKM-like, per TF's total reads), ratio to the leave-one-out pooled panel, and ratio to a
#     GC x mappability-matched background (20 random sets of mappable 1 kb bins with the same GC x map decile composition).
#     Browser plot per cluster: EZH2, MTF2, PCGF6, RBBP4 and pooled panel, CPM per 1 kb bin.
import json, os, re, subprocess, sys, gzip
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; Hx = f"{X}/regulatome/hox_check"; V3 = f"{X}/regulatome/gate1/v3_binned"
g = np.load(f"{V3}/units/grid.npz", allow_pickle=True); ci = {c: i for i, c in enumerate(g["chrom"])}; off = g["offset"]; NB = int(off[-1])
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
step = sys.argv[1]

if step == "bins":
    tf = sys.argv[2]
    r = pd.read_csv(f"/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment/TF.{tf}.bed.gz", sep="\t", header=None, usecols=[0, 1],
                    names=["chrom", "start"], dtype={"chrom": str, "start": np.int64})
    r = r[r.chrom.isin(ci)]
    np.save(f"{Hx}/bins/{tf}.npy", np.bincount(off[r.chrom.map(ci).values] + r.start.values // 1000, minlength=NB).astype(np.int32))
    print(tf, len(r)); sys.exit()

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.5, "svg.fonttype": "none", "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "savefig.dpi": 600, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
B = {t: np.load(f"{Hx}/bins/{t}.npy").astype(float) for t in tfs}; TOT = {t: B[t].sum() for t in tfs}; ALL = sum(B.values()); ALLT = sum(TOT.values())
def bins_of(c, s, e): return np.arange(off[ci[c]] + s // 1000, off[ci[c]] + (e - 1) // 1000 + 1)
# HOX clusters from GENCODE gene spans
gtf = f"{X}/data/gencode_v46/gencode.v46.basic.annotation.gtf.gz"
hox = {}
with gzip.open(gtf, "rt") as f:
    for l in f:
        if l[0] == "#": continue
        c = l.split("\t", 9)
        if c[2] != "gene": continue
        m = re.search(r'gene_name "(HOX[ABCD]\d+)"', c[8])
        if m:
            k = m.group(1)[3]; s, e = int(c[3]), int(c[4])
            hox[k] = (c[0], min(s, hox.get(k, (0, 1e12))[1]), max(e, hox.get(k, (0, 0, 0))[2]))
REG = {f"HOX{k}": (c, int(s) - 50000, int(e) + 50000) for k, (c, s, e) in sorted(hox.items())}
# PRC2 targets: protein-coding TSS +/- 2.5 kb overlapping a SUZ12 HEK293T peak
G = pd.read_csv(f"{X}/regulatome/m2_abc/inputs/genes_pc_v46.bed", sep="\t", header=None, names=["chrom", "start", "end", "name", "s", "strand", "gid", "type"])
G["tss"] = np.where(G.strand == "+", G.start, G.end)
P = pd.read_csv(f"{X}/regulatome/encode_repressive_check/data/ENCFF708RSP.bed.gz", sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
tgt = []
for r in G.itertuples():
    p = P[(P.chrom == r.chrom) & (P.end > r.tss - 2500) & (P.start < r.tss + 2500)]
    if len(p) and r.chrom in ci: tgt.append(r)
T = pd.DataFrame(tgt)
T[["chrom", "tss", "name"]].to_csv(f"{Hx}/prc2_target_genes.tsv", sep="\t", index=False)
tgt_bins = np.unique(np.concatenate([bins_of(c, max(t - 2500, 0), t + 2500) for c, t in zip(T.chrom, T.tss)]))
regions = {k: bins_of(*v) for k, v in REG.items()}; regions["HOX_all"] = np.concatenate([regions[k] for k in REG]); regions["PRC2_targets_SUZ12"] = tgt_bins
cov = np.load(f"{V3}/units/bin1k_cov.npz"); gc, mp = cov["gc"], cov["map"]; ok = mp >= 0.5
qg = np.quantile(gc[ok], np.linspace(0, 1, 11)); qm = np.quantile(mp[ok], np.linspace(0, 1, 11))
cls = np.clip(np.digitize(gc, qg[1:-1]), 0, 9) * 10 + np.clip(np.digitize(mp, qm[1:-1]), 0, 9)
rng = np.random.default_rng(1); excl = np.zeros(NB, bool); excl[regions["HOX_all"]] = True; excl[tgt_bins] = True
pool_by = {k: np.flatnonzero(ok & ~excl & (cls == k)) for k in np.unique(cls)}
def bg_sets(bins_, n=20):
    b = bins_[ok[bins_]]; ks, cn = np.unique(cls[b], return_counts=True)
    return [np.concatenate([rng.choice(pool_by[k], size=c, replace=False) for k, c in zip(ks, cn)]) for _ in range(n)]
SHOW = ["EZH2", "MTF2", "PCGF6", "RBBP4", "RBBP7", "MBD3", "CTCF", "NFYA"]
rows = []
for name, b in regions.items():
    b = b[ok[b]]; bgs = bg_sets(b); kb = len(b)
    for t in tfs:
        rpkm = B[t][b].sum() / kb / (TOT[t] / 1e6)
        pool = (ALL[b].sum() - B[t][b].sum()) / kb / ((ALLT - TOT[t]) / 1e6)
        bg = np.mean([B[t][x].sum() / len(x) / (TOT[t] / 1e6) for x in bgs])
        rows.append(dict(region=name, kb=kb, tf=t, reads=int(B[t][b].sum()), rpkm=rpkm, vs_panel=rpkm / pool if pool > 0 else np.nan,
                         vs_matched_bg=rpkm / bg if bg > 0 else np.nan))
R = pd.DataFrame(rows)
for c in ("vs_panel", "vs_matched_bg"): R[f"rank_{c}"] = R.groupby("region")[c].rank(ascending=False).astype(int)
R.to_csv(f"{Hx}/hox_prc2_enrichment.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 220)
print(f"PRC2 target set: {len(T)} protein-coding genes with a SUZ12 HEK293T peak within TSS +/- 2.5 kb ({len(tgt_bins)} kb); HOX regions:",
      {k: f"{v[0]}:{v[1]:,}-{v[2]:,}" for k, v in REG.items()})
print(R[R.tf.isin(SHOW)].pivot(index="tf", columns="region", values="vs_matched_bg").reindex(SHOW).round(2).to_string())
print(R[R.tf.isin(SHOW)].pivot(index="tf", columns="region", values="rank_vs_matched_bg").reindex(SHOW).to_string())
print(R[R.tf.isin(SHOW)].pivot(index="tf", columns="region", values="vs_panel").reindex(SHOW).round(2).to_string())
print(R[R.tf.isin(SHOW)].pivot(index="tf", columns="region", values="reads").reindex(SHOW).to_string())
print("panel median vs_matched_bg:", R.groupby("region").vs_matched_bg.median().round(2).to_dict())
# browser plots
cols = {"EZH2": "#1F3F8F", "MTF2": "#1F3F8F", "PCGF6": "#1F3F8F", "RBBP4": "#D98C1F", "panel": "#8C8C8C"}
for name, (c, s, e) in REG.items():
    b = bins_of(c, s, e); x = (b - off[ci[c]]) * 1000
    fig, axs = plt.subplots(5, 1, figsize=(3.4, 2.8), sharex=True, gridspec_kw=dict(hspace=0.2))
    for ax, t in zip(axs, ["EZH2", "MTF2", "PCGF6", "RBBP4", "panel"]):
        y = (ALL[b] / ALLT * 1e6) if t == "panel" else (B[t][b] / TOT[t] * 1e6)
        ax.bar(x, y, width=1000, align="edge", color=cols[t], lw=0); ax.set_ylabel(t if t != "panel" else "panel (79)", rotation=0, ha="right", va="center", fontsize=6)
        ax.set_yticks([0, round(float(y.max()), 2)]); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(axis="x", length=0)
    hs, he = REG[name][1] + 50000, REG[name][2] - 50000
    for ax in axs: ax.axvspan(hs, he, color="#F2F2F2", zorder=0, lw=0)
    axs[-1].set_xticks([s, e]); axs[-1].set_xticklabels([f"{c}:{s:,}", f"{e:,}"], fontsize=5); axs[-1].tick_params(axis="x", length=2)
    axs[0].set_title(f"{name} cluster +/- 50 kb (grey = HOX gene span); CPM per 1 kb", fontsize=6)
    fig.savefig(f"{Hx}/browser_{name}.svg", bbox_inches="tight"); plt.close(fig)
print("browser plots written")
