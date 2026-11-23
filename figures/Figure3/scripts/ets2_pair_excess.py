"""ETS2 vs RBBP4 vs the 79-TF panel: why ETS2 has so many coordinated binding pairs and so many 'other' ends (2026-11-23).

Per TF (78 ourrun_k50 TFs + RBBP4 sweep run): cells and peaks that entered Cicero (cicero_info / cds_info), single-cell reads
(lines of bigwig/fragment/TF.<TF>.bed.gz; single-end, one line per read) and barcodes, Cicero links tested, pairs (p <= 0.05, the
peak_edges table) and pairs > 25 kb, the share of tested links that pass, and the 18-state ChromHMM / promoter / union-ATAC
make-up of the Cicero input peaks. For ETS2 and RBBP4 also: per-chromosome peak density, share of peaks in the 24 largest
networks, pair distance quantiles, pairs per region, and the state split of other-other pairs.
  python ets2_pair_excess.py <outdir>     (Longleaf, cobinding env)"""
import gzip, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx

X = Path("/work/users/d/y/dyy12/XuLab"); COB = X / "cobinding"; SRC = Path("/vast/som/xujie_lab/TAGATG-293T")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
SD = X / "paper_figures/figure3/source_data"
NULL_MEAN, NULL_SD = 0.0019128237229209, 0.0117377057986638   # RBBP4.perm.fitConns.para.txt, used for every TF's p-value

def iv(r):
    r = r.replace("_", ":", 1).replace("_", "-", 1) if ":" not in r else r
    c, se = r.split(":"); s, e = se.split("-"); return c, int(s), int(e)

# annotation lookups -------------------------------------------------------------------------------------------------------
H = defaultdict(list)
for ln in gzip.open(X / "data/HEK293T_chromHMM18.bed.gz", "rt"):
    f = ln.split("\t"); H[f[0]].append((int(f[1]), int(f[2]), f[3]))
HS = {c: (np.array([a for a, _, _ in sorted(v)]), np.array([b for _, b, _ in sorted(v)]), np.array([s for _, _, s in sorted(v)])) for c, v in H.items()}
T = pd.read_csv(SD / "refseq_select_tss.tsv.gz", sep="\t"); TSS = {c: np.sort(g.tss.to_numpy()) for c, g in T.groupby("chrom")}
A = pd.read_csv("/vast/som/xujie_lab/Ref/HEK293T/ATAC/HEK293T_ATAC_union3.bed.gz", sep="\t", header=None, usecols=[0, 1, 2], comment="#")
ATAC = {}
for c, g in A.groupby(0):
    g = g.sort_values(1); ATAC[c] = (g[1].to_numpy(), np.maximum.accumulate(g[2].to_numpy()))
SIZES = dict(l.split()[:2] for l in open("/vast/som/xujie_lab/Ref/hg38/chrom.sizes")); SIZES = {k: int(v) for k, v in SIZES.items()}
E_ = {"EnhA1", "EnhA2", "EnhG1", "EnhG2", "EnhWk", "EnhBiv"}; R_ = {"ReprPC", "ReprPCWk", "Het", "ZNF/Rpts"}

def annotate(regs):
    """DataFrame of regions with midpoint state, promoter flag (TSS +-1 kb), union-ATAC overlap, width."""
    rows = []
    for r in regs:
        c, s, e = iv(r); m = (s + e) // 2; st = "none"
        if c in HS:
            a, b, lab = HS[c]; i = np.searchsorted(a, m, side="right") - 1
            if i >= 0 and b[i] > m: st = lab[i]
        prom = False
        if c in TSS:
            p = TSS[c]; i = np.searchsorted(p, s - 1000); prom = bool(i < len(p) and p[i] <= e + 1000)
        atac = False
        if c in ATAC:
            a, bmax = ATAC[c]; i = np.searchsorted(a, e, side="left") - 1   # last ATAC start < e
            atac = bool(i >= 0 and bmax[i] > s)
        cls = "P" if prom else ("E" if st in E_ else "R" if st in R_ else "O")
        rows.append((r, c, s, e, e - s, st, prom, atac, cls))
    return pd.DataFrame(rows, columns=["region", "chrom", "start", "end", "width", "state", "promoter", "in_atac_union", "cls"])

def count_reads(tf):
    p = SRC / f"bigwig/fragment/TF.{tf}.bed.gz"
    o = subprocess.run(f"zcat {p} | awk '{{n++; b[$4]=1}} END {{print n, length(b)}}'", shell=True, capture_output=True, text=True).stdout.split()
    return int(o[0]), int(o[1])

# per-TF table ----------------------------------------------------------------------------------------------------------------
tfs = sorted(p.name for p in (COB / "work/ourrun_k50").iterdir() if p.is_dir()) + ["RBBP4"]
RUN = {t: COB / f"work/ourrun_k50/{t}" for t in tfs}; RUN["RBBP4"] = COB / "work/RBBP4_sweep_k50_i10000_s555"
rows, states, ann_keep, edges_keep = [], {}, {}, {}
for t in tfs:
    w = RUN[t]
    info = {}
    for f in (w / "cicero/cicero/cicero_info.tsv", w / "cicero/cds/cds_info.tsv"):
        if f.exists():
            info.update(dict(l.rstrip("\n").split("\t")[:2] for l in open(f) if "\t" in l))
    ef = w / f"{t}.peak_edges.tsv"
    if not ef.exists(): print("skip", t); continue
    E = pd.read_csv(ef, sep="\t", usecols=["peak1", "peak2", "coaccess", "pval", "qval"])
    E = E[E.pval <= 0.05]
    a, b = zip(*[(iv(x), iv(y)) for x, y in zip(E.peak1, E.peak2)]) if len(E) else ([], [])
    d = np.array([abs((p[1] + p[2]) // 2 - (q[1] + q[2]) // 2) for p, q in zip(a, b)])
    reads, bcs = count_reads(t)
    # Cicero input peaks: regions_used (ourrun) or annot table (both list the matrix peaks)
    rf = w / "cicero/cds/regions_used_canonical.tsv.gz"
    regs = pd.read_csv(rf, sep="\t", header=None)[0].astype(str).tolist() if rf.exists() else pd.read_csv(w / f"{t}.peak_annot.tsv", sep="\t").peak.tolist()
    regs = [r for r in regs if r[:3] == "chr"]
    an = annotate(regs)
    states[t] = an.state.value_counts(normalize=True)
    n_links = float(info.get("n_links", "nan"))
    rows.append(dict(tf=t, cells=int(float(info.get("n_cells", info.get("n_cells_in", "nan")))), reads=reads, barcodes=bcs,
                     reads_per_barcode=reads / max(bcs, 1), peaks_in_cicero=int(float(info.get("n_peaks", info.get("n_peaks_in", "nan")))),
                     peaks_listed=len(regs), links_tested=n_links, pairs=len(E), pairs_gt25kb=int((d > 25000).sum()),
                     pass_rate=len(E) / n_links if n_links == n_links else np.nan, pairs_per_peak=len(E) / max(len(regs), 1),
                     regions_in_pairs=len(set(E.peak1) | set(E.peak2)),
                     promoter_share=an.promoter.mean(), atac_union_share=an.in_atac_union.mean(), median_width=an.width.median(),
                     P=(an.cls == "P").mean(), E=(an.cls == "E").mean(), R=(an.cls == "R").mean(), O=(an.cls == "O").mean(),
                     dist_median=np.median(d) if len(d) else np.nan, dist_q25=np.percentile(d, 25) if len(d) else np.nan,
                     dist_q75=np.percentile(d, 75) if len(d) else np.nan))
    if t in ("ETS2", "RBBP4"):
        ann_keep[t], edges_keep[t] = an, (E, d)
    print(t, rows[-1]["cells"], rows[-1]["peaks_in_cicero"], rows[-1]["pairs"], flush=True)
TF = pd.DataFrame(rows).set_index("tf")
TF["pairs_per_region_in_pairs"] = TF.pairs / TF.regions_in_pairs
# where does ETS2 sit: log-log fit of pairs on peaks (and on cells) across TFs
fit = {}
for xcol in ("peaks_in_cicero", "cells", "reads"):
    ok = (TF[xcol] > 0) & (TF.pairs > 0)
    x, y = np.log10(TF.loc[ok, xcol]), np.log10(TF.loc[ok, "pairs"])
    k, c0 = np.polyfit(x, y, 1); res = y - (k * x + c0)
    TF.loc[ok, f"resid_vs_{xcol}"] = res
    fit[xcol] = dict(slope=k, r=np.corrcoef(x, y)[0, 1], ETS2_resid=res.get("ETS2", np.nan), ETS2_resid_rank=int((res > res.get("ETS2", 0)).sum()) + 1,
                     ETS2_fold_over_fit=10 ** res.get("ETS2", np.nan), n=int(ok.sum()))
for col in ("pairs", "pairs_per_peak", "pass_rate", "cells", "reads", "reads_per_barcode", "peaks_in_cicero"):
    TF[f"rank_{col}"] = TF[col].rank(ascending=False, method="min")
TF.to_csv(OUT / "per_tf.tsv", sep="\t")
pd.DataFrame(fit).T.to_csv(OUT / "pairs_vs_input_fit.tsv", sep="\t")
ST = pd.DataFrame(states).T.fillna(0); ST.to_csv(OUT / "peak_states_per_tf.tsv", sep="\t")

# ETS2 vs RBBP4 details ---------------------------------------------------------------------------------------------------------
summ = []
for t in ("ETS2", "RBBP4"):
    an = ann_keep[t]; E, d = edges_keep[t]
    # signal and width from the MACS2 narrowPeak (q <= 0.05 rows; RBBP4 = its MACS2 run if present)
    npk = SRC / f"bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
    if npk.exists():
        N = pd.read_csv(npk, sep="\t", header=None); N = N[N[8] >= -np.log10(0.05)]
        sig = dict(narrowpeak_rows_q005=len(N), median_fold_enrichment=N[6].median(), median_pileup_summit_reads=np.nan,
                   median_neglog10q=N[8].median(), median_narrowpeak_width=(N[2] - N[1]).median())
    else:
        sig = {}
    # chromosome density
    dens = an.groupby("chrom").size().rename("peaks").to_frame(); dens["Mb"] = [SIZES.get(c, np.nan) / 1e6 for c in dens.index]
    dens["peaks_per_Mb"] = dens.peaks / dens.Mb; dens.to_csv(OUT / f"{t}_peaks_per_chrom.tsv", sep="\t")
    # 1 Mb bin concentration
    b = Counter(zip(an.chrom, an.start // 1_000_000)); v = np.sort(np.array(list(b.values())))[::-1]
    top1 = v[: max(1, len(v) // 100)].sum() / v.sum(); top10 = v[: max(1, len(v) // 10)].sum() / v.sum()
    # networks: q-graph (same as the network panels) -> 24 largest components
    Q = pd.read_csv(RUN[t] / f"{t}.peak_edges.tsv", sep="\t"); Q = Q[(Q.qval <= 0.05) & (Q.coaccess >= 0) & (Q.n_links >= 1)]
    G = nx.Graph()
    for x, y in zip(Q.peak1, Q.peak2):
        p, q = iv(x), iv(y)
        if p[0] == q[0] and abs((p[1] + p[2]) // 2 - (q[1] + q[2]) // 2) <= 1_000_000: G.add_edge(x, y)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    top24 = sum(len(c) for c in comps[:24])
    # other-other pairs (> 25 kb) split by state
    st = dict(zip(an.region, an.state)); cl = dict(zip(an.region, an.cls))
    def canon(r): c, s, e = iv(r); return f"{c}:{s}-{e}"
    st = {canon(k): v for k, v in st.items()}; cl = {canon(k): v for k, v in cl.items()}
    far = E[d > 25000]
    oo = Counter()
    n_far = len(far); n_oo = 0
    grp = lambda s: "Tss*" if s.startswith("Tss") else s
    for x, y in zip(far.peak1, far.peak2):
        if cl.get(canon(x)) == "O" and cl.get(canon(y)) == "O":
            n_oo += 1; oo["–".join(sorted([grp(st.get(canon(x), "none")), grp(st.get(canon(y), "none"))]))] += 1
    pd.Series(oo).sort_values(ascending=False).rename("pairs").to_frame().assign(share_of_OO=lambda z: z.pairs / max(n_oo, 1),
        share_of_all_gt25kb=lambda z: z.pairs / max(n_far, 1)).to_csv(OUT / f"{t}_other_other_pairs_by_state.tsv", sep="\t")
    q = np.percentile(d, [10, 25, 50, 75, 90]) if len(d) else [np.nan] * 5
    summ.append(dict(tf=t, peaks=len(an), promoter_share=an.promoter.mean(), atac_union_share=an.in_atac_union.mean(),
                     low_accessibility_share=1 - an.in_atac_union.mean(), median_width=an.width.median(), **sig,
                     top1pct_1Mb_bins_share=top1, top10pct_1Mb_bins_share=top10, networks=len(comps), regions_in_networks=G.number_of_nodes(),
                     top24_network_regions=top24, top24_share_of_peaks=top24 / len(an), top24_share_of_network_regions=top24 / max(G.number_of_nodes(), 1),
                     pairs=len(E), pairs_gt25kb=n_far, other_other_gt25kb=n_oo, other_other_share=n_oo / max(n_far, 1),
                     dist_p10=q[0], dist_p25=q[1], dist_median=q[2], dist_p75=q[3], dist_p90=q[4],
                     pairs_per_region_in_pairs=len(E) / max(len(set(E.peak1) | set(E.peak2)), 1), qgraph_edges_per_region=G.number_of_edges() / max(G.number_of_nodes(), 1)))
    an.state.value_counts().rename("peaks").to_frame().assign(share=lambda z: z.peaks / z.peaks.sum()).to_csv(OUT / f"{t}_peak_states.tsv", sep="\t")
S = pd.DataFrame(summ).set_index("tf").T; S.to_csv(OUT / "ets2_vs_rbbp4.tsv", sep="\t")
med = TF.median(numeric_only=True).rename("panel_median_79")
pd.concat([TF.loc[["ETS2", "RBBP4"]].T, med], axis=1).to_csv(OUT / "ets2_rbbp4_panel_median.tsv", sep="\t")
print(pd.DataFrame(fit).T.to_string()); print(S.to_string()); print(pd.concat([TF.loc[["ETS2", "RBBP4"]].T, med], axis=1).to_string())
