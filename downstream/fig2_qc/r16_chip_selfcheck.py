#!/usr/bin/env python
"""Bulk ChIP self-check for the 6 references used in the round 6 heatmaps: are the references weak, or is the mismatch
with our sites?
  BAM stats come from r16_bamstats.sbatch (samtools/bedtools; OS/bamstats_<TF>.json).
  This script: (a) each ChIP's CPM bigWig (r13_bamcoverage.sbatch) over +-2 kb, 50 bp bins, around its OWN top 1,000
                  summits by signalValue and around OUR 1,000 differential regions for that TF (round 6 heatmap block);
                  centre/flank = mean of +-250 bp / mean of the outer 500 bp on each side; (b) fraction of our regions whose
                  centre is within 500 bp of any reference peak; (c) fraction of the reference's top 1,000 summits that are
                  CpG-rich promoters: within 1 kb of a GENCODE v46 basic TSS and +-500 bp GC > 0.5 and CpG obs/exp > 0.6
                  (computed from hg38; no CpG-island file is used); (d) ENCODE portal QC already saved in
                  OS/encode_json (FRiP, NSC, RSC, audits). Figure chip_selfcheck.{png,pdf,svg} and chip_selfcheck.xlsx.
Peak files: CTCF = ENCFF314ZAL (ENCODE HEK293 optimal IDR); MAZ/ZBTB7A/ZIC2/ZNF777 = ENCODE IDR peaks in REF;
ZNF282 = GSM2466515 HA ChIP-exo (HEK293T) MACS2 re-call in REF."""
import gzip
import re
import json
import sys

import numpy as np
import pandas as pd

from fig2_common import BAM, FASTA, GTF, OUT, REF, ROOT, TFS

O6 = OUT / "round6_diffpeaks"
OH = O6 / "heatmaps"
OS = O6 / "chip_selfcheck"
OS.mkdir(parents=True, exist_ok=True)
PEAKS = dict(REF)
PEAKS["CTCF"] = ROOT / "downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed"
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def read_peaks(t):
    P = pd.read_csv(PEAKS[t], sep="\t", header=None, comment="#")
    P = P[P[0].isin(CHROMS)].copy()
    P["summit"] = P[1] + P[9].clip(lower=0)
    return P


import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyBigWig  # noqa: E402
import subprocess  # noqa: E402
from matplotlib import font_manager  # noqa: E402

W, STEP, NB = 2000, 50, 80
xs = np.arange(-W, W, STEP) + STEP // 2
CTR, FL = slice(NB // 2 - 5, NB // 2 + 5), np.r_[0:10, NB - 10:NB]
META = {"CTCF": ("ENCSR000DTW", "ENCFF314ZAL / ENCFF139DCW", "HEK293", "ChIP-seq, anti-CTCF antibody (ENCAB000AFP)"),
        "MAZ": ("ENCSR290SSQ", "ENCFF529XTF", "HEK293", "ChIP-seq, eGFP-MAZ (Flp-In), anti-GFP"),
        "ZBTB7A": ("ENCSR773REP", "ENCFF114DGV", "HEK293", "ChIP-seq, eGFP-ZBTB7A (Flp-In), anti-GFP"),
        "ZIC2": ("ENCSR728MWW", "ENCFF359GIY", "HEK293", "ChIP-seq, eGFP-ZIC2 (Flp-In), anti-GFP"),
        "ZNF282": ("GSM2466515", "MACS2 re-call (hg38)", "HEK293T", "ChIP-exo, HA-ZNF282, anti-HA"),
        "ZNF777": ("ENCSR295BIP", "ENCFF735CAY", "HEK293", "ChIP-seq, eGFP-ZNF777 (Flp-In), anti-GFP")}
PEAKFILE_ACC = {"CTCF": "ENCFF314ZAL", "MAZ": "ENCFF529XTF", "ZBTB7A": "ENCFF114DGV", "ZIC2": "ENCFF359GIY",
                "ZNF777": "ENCFF735CAY"}

# ENCODE portal QC (saved JSON)
enc = {}
for t, (acc, *_r) in META.items():
    if not acc.startswith("ENCSR"):
        continue
    d = json.load(open(OS / "encode_json" / f"{acc}.json"))
    pg = json.load(open(OS / "encode_json" / f"{acc}_page.json"))
    nsc, rsc, frip = [], [], None
    for f in d["files"]:
        for q in f.get("quality_metrics", []):
            if not isinstance(q, dict) or f.get("assembly") != "GRCh38":
                continue
            if f["output_type"] == "alignments" and "NSC" in q:
                nsc.append(q["NSC"]); rsc.append(q["RSC"])
            if f["accession"] == PEAKFILE_ACC[t] and "frip" in q:
                frip = q["frip"]
    ctrl = d.get("possible_controls", [])
    enc[t] = dict(encode_frip_peakfile=frip, encode_NSC=", ".join(f"{v:.2f}" for v in sorted(set(nsc))),
                  encode_RSC=", ".join(f"{v:.2f}" for v in sorted(set(rsc))),
                  control=f"yes ({len(ctrl)} ENCODE control experiment{'s' if len(ctrl) > 1 else ''})" if ctrl else "none listed",
                  encode_audit="; ".join(f"{k}: {', '.join(sorted(set(a['category'] for a in v)))}"
                                         for k, v in pg.get("audit", {}).items() if k != "INTERNAL_ACTION"))

# our regions per TF (same rows as the heatmap)
R = pd.read_csv(O6 / "rows.tsv", sep="\t")
S = np.load(O6 / "diff_sets.npz")
sets = {t: S[f"D-B_1000_{t}"] for t in TFS}
u, cnt = np.unique(np.concatenate(list(sets.values())), return_counts=True)
shared = set(u[cnt > 1])
sets = {t: np.array([i for i in v if i not in shared]) for t, v in sets.items()}


def bwmat(bw, chrom, centre):
    M = np.zeros((len(centre), NB))
    for r, (c, x) in enumerate(zip(chrom, centre)):
        L = bw.chroms().get(c)
        if L is None or x - W < 0 or x + W > L:
            continue
        M[r] = np.nan_to_num(np.array(bw.stats(c, int(x - W), int(x + W), nBins=NB, type="mean"), dtype=float))
    return M


def cf(M):
    p = M.mean(0)
    return p[CTR].mean() / max(p[FL].mean(), 1e-12)


# TSS for CpG-rich promoter call
tss = {}
with gzip.open(GTF, "rt") as fh:
    for ln in fh:
        if ln[0] == "#":
            continue
        f = ln.split("\t", 8)
        if f[2] != "transcript":
            continue
        tss.setdefault(f[0], []).append(int(f[3]) if f[6] == "+" else int(f[4]))
tss = {c: np.unique(v) for c, v in tss.items()}


def fetch_many(regs):
    """regs: list of (chrom, start0, end) -> sequences via samtools faidx."""
    tmpf = OS / "faidx_regions.txt"
    tmpf.write_text("".join(f"{c}:{a + 1}-{b}\n" for c, a, b in regs))
    out = subprocess.run(["samtools", "faidx", str(FASTA), "-r", str(tmpf)], check=True, capture_output=True, text=True).stdout
    seqs, cur = [], []
    for ln in out.splitlines():
        if ln.startswith(">"):
            if cur or seqs:
                seqs.append("".join(cur))
            cur = []
        else:
            cur.append(ln)
    seqs.append("".join(cur))
    return seqs


def near(sorted_pos, c, x, d):
    v = sorted_pos.get(c)
    if v is None or not len(v):
        return False
    i = np.searchsorted(v, x)
    return any(abs(v[j] - x) <= d for j in (i - 1, i) if 0 <= j < len(v))


def cpg_promoter_frac(chrom, pos):
    pos = np.asarray(pos, int)
    nt = np.array([near(tss, c, int(x), 1000) for c, x in zip(chrom, pos)])
    if not nt.any():
        return 0.0
    seqs = fetch_many([(c, max(0, int(x) - 500), int(x) + 500) for c, x, k in zip(chrom, pos, nt) if k])
    ok = 0
    for s in seqs:
        s = s.upper(); n = max(len(s), 1); g = s.count("G"); cc = s.count("C"); cg = s.count("CG")
        ok += ((g + cc) / n > 0.5) and (cg * n / max(g * cc, 1) > 0.6)
    return ok / len(pos)


rows, mats = [], {}
for t in TFS:
    P = read_peaks(t)
    top = P.sort_values(6, ascending=False).head(1000)
    bw = pyBigWig.open(str(OH / "bw" / f"{t}.bw"))
    Mown = bwmat(bw, top[0].values, top.summit.values)
    ix = sets[t]
    oc = R.chrom.values[ix]; ox = ((R.start.values[ix] + R.end.values[ix]) // 2).astype(int)
    Mour = bwmat(bw, oc, ox)
    mats[t] = (Mown, Mour)
    # our regions near any reference peak (interval distance <= 500 bp)
    hit = 0
    for c, x in zip(oc, ox):
        pc = P[P[0] == c]
        if len(pc) and (np.maximum(0, np.maximum(pc[1].values - x, x - pc[2].values)) <= 500).any():
            hit += 1
    cgp = cpg_promoter_frac(top[0].values, top.summit.values)
    ourcgp = cpg_promoter_frac(oc, ox)
    bs = json.loads(re.sub(r":\s*\.", ": 0.", (OS / f"bamstats_{t}.json").read_text()))  # bc prints .28
    acc, pfile, cell, assay = META[t]
    rows.append(dict(TF=t, accession=acc, peak_file=pfile, cell_line=cell, assay_tag=assay,
                     control=enc.get(t, {}).get("control", "not recorded locally (GEO; not checked)"),
                     paired_end=bs["paired"], reads_counted=bs["reads_counted"], unique_5p_ends=bs["unique_5p_ends"],
                     dup_rate=round(bs["dup_rate_5p"], 4), peaks_in_file=bs["n_peaks"], frip_dedup_ours=round(bs["frip_dedup"], 4),
                     encode_frip=enc.get(t, {}).get("encode_frip_peakfile"), encode_NSC=enc.get(t, {}).get("encode_NSC"),
                     encode_RSC=enc.get(t, {}).get("encode_RSC"), encode_audit_warnings=enc.get(t, {}).get("encode_audit"),
                     centre_flank_own_top1000=round(cf(Mown), 3), centre_flank_our_regions=round(cf(Mour), 3),
                     n_our_regions=len(ix), our_regions_within500bp_of_ref_peak=round(hit / len(ix), 4),
                     own_top1000_CpG_promoter=round(cgp, 4), our_regions_CpG_promoter=round(ourcgp, 4)))
    print(rows[-1], flush=True)
T = pd.DataFrame(rows)
T.to_csv(OS / "chip_selfcheck.tsv", sep="\t", index=False)
with pd.ExcelWriter(OS / "chip_selfcheck.xlsx") as xw:
    T.to_excel(xw, sheet_name="summary", index=False)
    pd.DataFrame({"column": ["dup_rate", "frip_dedup_ours", "encode_frip", "encode_NSC / RSC", "centre_flank_*",
                             "our_regions_within500bp_of_ref_peak", "*_CpG_promoter"],
                  "definition": ["1 - unique 5' ends (chrom, pos, strand) / primary mapped reads (read 1 only if paired)",
                                 "unique 5' ends inside the reference peak file / all unique 5' ends",
                                 "FRiP that ENCODE reports for this peak file (portal QC)",
                                 "strand cross-correlation of each GRCh38 replicate alignment (ENCODE: NSC > 1.05, RSC > 0.8 expected)",
                                 "ChIP CPM mean within +-250 bp / mean of the outer 500 bp of a +-2 kb window (profile over rows)",
                                 "our differential regions (heatmap block) whose centre lies within 500 bp of any reference peak",
                                 "within 1 kb of a GENCODE v46 basic TSS and +-500 bp GC > 0.5, CpG obs/exp > 0.6"]}
                 ).to_excel(xw, sheet_name="definitions", index=False)

# ---- figure ----
for f in (ROOT.parent / "fonts/arial").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402
P_ = rp.use()
plt.rcParams.update({"font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2, "xtick.labelsize": 6, "ytick.labelsize": 6})
CM = P_.seq_cmap()
MM = 1 / 25.4
fig = plt.figure(figsize=(183 * MM, 120 * MM))
gs = fig.add_gridspec(1, 6, wspace=0.28, left=0.04, right=0.995, top=0.86, bottom=0.08)
for k, t in enumerate(TFS):
    Mown, Mour = mats[t]
    nz = Mown[Mown > 0]
    vm = float(np.percentile(nz, 98)) if nz.size else 1.0
    ylim = 1.1 * max(Mown.mean(0).max(), Mour.mean(0).max())
    sg = gs[k].subgridspec(4, 2, height_ratios=[1.1, 6, 0.35, 0.16], hspace=0.06, wspace=0.1)
    for j, (M, lab) in enumerate(((Mown, "own top\n1,000"), (Mour, f"our\n{len(Mour):,}"))):
        o = np.argsort(-M[:, CTR].sum(1), kind="stable")
        axp = fig.add_subplot(sg[0, j])
        axp.plot(xs, M.mean(0), color=P_.neutral.ink, lw=0.6)
        axp.set_xlim(-W, W); axp.set_ylim(0, ylim); axp.set_xticks([])
        axp.spines[["top", "right"]].set_visible(False)
        tk = float(f"{ylim / 1.1:.1g}")
        axp.set_yticks([0, tk]); axp.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        if j:
            axp.tick_params(labelleft=False)
        axp.set_title(f"{lab}\nc/f {cf(M):.1f}", fontsize=6, pad=2)
        ax = fig.add_subplot(sg[1, j])
        im = ax.imshow(M[o], aspect="auto", cmap=CM, vmin=0, vmax=vm, interpolation="nearest",
                       extent=[-W, W, len(M), 0], rasterized=True)
        ax.set_yticks([]); ax.set_xticks([-W, 0, W], ["−2", "0", "+2"])
        for lb, ha in zip(ax.get_xticklabels(), ("left", "center", "right")):
            lb.set_ha(ha)
        for s in ax.spines.values():
            s.set_visible(False)
    cax = fig.add_subplot(sg[3, :])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, vm]); cb.set_ticklabels(["0", f"{vm:.2g}"]); cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    for lb, ha in zip(cb.ax.get_xticklabels(), ("left", "right")):
        lb.set_ha(ha)
    x0 = gs[k].get_position(fig)
    fig.text((x0.x0 + x0.x1) / 2, 0.95, f"{t} ({META[t][2]})", ha="center", fontsize=7, fontweight="bold")
fig.text(0.52, 0.012, "Distance from summit / region centre (kb); ChIP CPM", ha="center", fontsize=7)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OS / f"chip_selfcheck.{ext}", dpi=600)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(T.drop(columns=["encode_audit_warnings"]).to_string())
print("done")
