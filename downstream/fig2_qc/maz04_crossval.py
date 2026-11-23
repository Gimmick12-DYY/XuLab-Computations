"""MAZ reference audit, step 4: cross-validate bulk MAZ datasets against our Fig 2 MAZ reference
(ENCODE ENCSR290SSQ, HEK293, exogenous eGFP-MAZ, IDR peaks ENCFF529XTF).

Inputs: data/external/maz_refs/{peaks,bw} from maz03_download.sbatch.
Per dataset (top = 1000 strongest summits unless the set is smaller):
  A. peaks: n; overlap with the reference (fraction of this set's peaks hitting a reference peak, fraction of reference
     peaks hit, bp Jaccard); recall of the reference top-1000 summits (any peak within 200 bp).
  B. CpG-rich promoter fraction of the top summits (same rule as r16: <=1 kb of a GENCODE v46 basic TSS, +-500 bp GC > 0.5
     and CpG obs/exp > 0.6).
  C. motif: top 500 summits +-250 bp; FIMO p < 1e-4; fraction with a hit within +-50 bp of the summit vs the same
     sequences dinucleotide-shuffled (3 copies, keeps GC and CpG); centrality = hit density in +-50 bp / density in
     150-250 bp flanks. MAZ (HOCOMOCO v11 MAZ 0.A, 1.A) plus GC-rich controls SP1, KLF4, PATZ1, ZN281, EGR1, and CTCF.
  D. signal: mean bigWig signal +-250 bp at the union of all sets' top summits (merged within 250 bp); Spearman of log1p;
     own centre (+-100 bp) / flank (1-2 kb) ratio at the set's top summits.
  E. MAZ-dependence: HAP1 anti-MAZ ChIP, parental vs MAZ-KO (GSE215081, ChIP-Atlas RPM bigWigs): log2 parental/KO at the
     set's top summits, vs random CpG-rich promoters and CTCF reference summits as backgrounds.
Non-HEK sets are used only to judge the HEK293 reference, never as references for our HEK293T data.
"""
import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
from pyfaidx import Fasta

ROOT = Path("/work/users/d/y/dyy12/XuLab")
D = ROOT / "data/external/maz_refs"
OUT = ROOT / "results_293T_analysis/maz_ref_audit"
TMP = OUT / "tmp"; TMP.mkdir(parents=True, exist_ok=True)
FASTA = ROOT / "downstream/cache/hg38.fa"
GTF = ROOT / "data/gencode_v46/gencode.v46.basic.annotation.gtf.gz"
MEME_DB = ROOT / "downstream/cache/HOCOMOCOv11_full_HUMAN_mono_meme_format.meme"
BLACK = ROOT / "downstream/cache/hg38-blacklist.v2.bed.gz"
CTCF_REF = ROOT / "data/CTCF_majority2of3.bed"
REF = "HEK293_eGFP-MAZ_ENCSR290SSQ"
MOTIFS = ["MAZ_HUMAN.H11MO.0.A", "MAZ_HUMAN.H11MO.1.A", "SP1_HUMAN.H11MO.0.A", "KLF4_HUMAN.H11MO.0.A",
          "PATZ1_HUMAN.H11MO.0.C", "ZN281_HUMAN.H11MO.0.A", "EGR1_HUMAN.H11MO.0.A", "CTCF_HUMAN.H11MO.0.A"]
NTOP, NMOT, HW = 1000, 500, 250
CHROMS = {f"chr{i}" for i in list(range(1, 23)) + ["X"]}
RNG = np.random.default_rng(0)

# name, cell line, assay/construct, group, peak file, peak format, bigwig
SETS = [
    (REF, "HEK293", "ChIP-seq, exogenous eGFP-MAZ (Flp-In, CMV), anti-GFP; ENCODE IDR", "HEK", f"{REF}.encode_idr.bed.gz", "np", REF),
    ("HEK293_GFP-MAZ_GSE76494", "HEK293", "ChIP-seq, exogenous GFP-MAZ (Flp-In), anti-GFP, 1 rep; ChIP-Atlas MACS2 q<1e-5", "HEK", "HEK293_GFP-MAZ_GSE76494.chipatlas_q1e-5.bed", "ca", "HEK293_GFP-MAZ_GSE76494"),
    ("HEK293T_V5-MAZ_CnR_rep1", "HEK293T", "CUT&RUN, transient V5-MAZ, anti-V5; GEO MACS2 q<0.05", "HEK", "HEK293T_V5-MAZ_CnR_rep1.geo_macs2_q0.05.narrowPeak.gz", "np", "HEK293T_V5-MAZ_CnR_rep1"),
    ("HEK293T_V5-MAZ_CnR_rep2", "HEK293T", "CUT&RUN, transient V5-MAZ, anti-V5; GEO MACS2 q<0.05", "HEK", "HEK293T_V5-MAZ_CnR_rep2.geo_macs2_q0.05.narrowPeak.gz", "np", "HEK293T_V5-MAZ_CnR_rep2"),
    ("HEK293T_V5-CTCF_CnR_rep1", "HEK293T", "CUT&RUN, V5-CTCF (same batch; positive control)", "control", "HEK293T_V5-CTCF_CnR_rep1.geo_macs2_q0.05.narrowPeak.gz", "np", "HEK293T_V5-CTCF_CnR_rep1"),
    ("HEK293T_V5-mock_CnR_rep1", "HEK293T", "CUT&RUN, V5 mock IP (same batch; negative control); ChIP-Atlas q<1e-5", "control", "HEK293T_V5-mock_CnR_rep1.chipatlas_q1e-5.bed", "ca", "HEK293T_V5-mock_CnR_rep1"),
    ("HEK293T_V5-mock_CnR_rep2", "HEK293T", "CUT&RUN, V5 mock IP (same batch; negative control); ChIP-Atlas q<1e-5", "control", "HEK293T_V5-mock_CnR_rep2.chipatlas_q1e-5.bed", "ca", "HEK293T_V5-mock_CnR_rep2"),
    ("K562_LAP-MAZ_ENCSR643JRH", "K562", "ChIP-seq, endogenous LAP-tag knock-in, anti-GFP; ENCODE IDR", "nonHEK", "K562_LAP-MAZ_ENCSR643JRH.encode_idr.bed.gz", "np", "K562_LAP-MAZ_ENCSR643JRH"),
    ("GM12878_LAP-MAZ_ENCSR903MVU", "GM12878", "ChIP-seq, endogenous LAP-tag knock-in, anti-GFP; ENCODE IDR", "nonHEK", "GM12878_LAP-MAZ_ENCSR903MVU.encode_idr.bed.gz", "np", "GM12878_LAP-MAZ_ENCSR903MVU"),
    ("HepG2_FLAG-MAZ_ENCSR700PNE", "HepG2", "ChIP-seq, endogenous 3xFLAG knock-in, anti-FLAG; ENCODE IDR", "nonHEK", "HepG2_FLAG-MAZ_ENCSR700PNE.encode_idr.bed.gz", "np", "HepG2_FLAG-MAZ_ENCSR700PNE"),
    ("K562_FLAG-MAZ_ENCSR163IUV", "K562", "ChIP-seq, endogenous 3xFLAG knock-in, anti-FLAG; ENCODE IDR", "nonHEK", "K562_FLAG-MAZ_ENCSR163IUV.encode_idr.bed.gz", "np", "K562_FLAG-MAZ_ENCSR163IUV"),
    ("GM12878_abMAZ_ENCSR000DZA", "GM12878", "ChIP-seq, anti-MAZ ab85725; ENCODE IDR", "nonHEK", "GM12878_abMAZ_ENCSR000DZA.encode_idr.bed.gz", "np", "GM12878_abMAZ_ENCSR000DZA"),
    ("MCF-7_abMAZ_ENCSR288IJC", "MCF-7", "ChIP-seq, anti-MAZ ab85725; ENCODE IDR", "nonHEK", "MCF-7_abMAZ_ENCSR288IJC.encode_idr.bed.gz", "np", "MCF-7_abMAZ_ENCSR288IJC"),
    ("HepG2_abMAZ_ENCSR000EDN", "HepG2", "ChIP-seq, anti-MAZ ab85725; ENCODE IDR", "nonHEK", "HepG2_abMAZ_ENCSR000EDN.encode_idr.bed.gz", "np", None),
    ("K562_abMAZ_ENCSR000EFX", "K562", "ChIP-seq, anti-MAZ ab85725; ENCODE IDR", "nonHEK", "K562_abMAZ_ENCSR000EFX.encode_idr.bed.gz", "np", None),
    ("IMR-90_abMAZ_ENCSR000EFF", "IMR-90", "ChIP-seq, anti-MAZ ab85725; ENCODE IDR", "nonHEK", "IMR-90_abMAZ_ENCSR000EFF.encode_idr.bed.gz", "np", None),
    ("A549_abMAZ_ENCSR636YLV", "A549", "ChIP-seq, anti-MAZ ab85725; ENCODE IDR", "nonHEK", "A549_abMAZ_ENCSR636YLV.encode_idr.bed.gz", "np", None),
    ("HeLa-S3_abMAZ_ENCSR000ECL", "HeLa-S3", "ChIP-seq, anti-MAZ ab85725; ENCODE IDR", "nonHEK", "HeLa-S3_abMAZ_ENCSR000ECL.encode_idr.bed.gz", "np", None),
    ("HAP1_abMAZ_parental_rep2", "HAP1", "ChIP-seq, anti-MAZ, parental; ChIP-Atlas q<1e-5", "nonHEK", "HAP1_abMAZ_parental_rep2.chipatlas_q1e-5.bed", "ca", "HAP1_abMAZ_parental_rep2"),
    ("HAP1_abMAZ_MAZKO_rep2", "HAP1", "ChIP-seq, anti-MAZ, MAZ knockout (negative control); ChIP-Atlas q<1e-5", "control", "HAP1_abMAZ_MAZKO_rep2.chipatlas_q1e-5.bed", "ca", "HAP1_abMAZ_MAZKO_rep2"),
]
EXTRA_BW = ["HAP1_abMAZ_parental_rep1", "HAP1_abMAZ_MAZKO_rep1", "HEK293T_V5-CTCF_CnR_rep2"]


def sh(*cmd, **kw):
    return subprocess.run(list(cmd), check=True, capture_output=True, text=True, **kw).stdout


def load_black():
    b = pd.read_csv(BLACK, sep="\t", header=None, usecols=[0, 1, 2])
    return {c: g[[1, 2]].values for c, g in b.groupby(0)}


BL = load_black()


def in_black(c, x):
    v = BL.get(c)
    return v is not None and ((v[:, 0] <= x) & (v[:, 1] > x)).any()


def read_peaks(fname, fmt):
    p = D / "peaks" / fname
    op = gzip.open if str(p).endswith(".gz") else open
    rows = []
    with op(p, "rt") as fh:
        for ln in fh:
            if ln.startswith(("track", "#", "browser")):
                continue
            f = ln.rstrip("\n").split("\t")
            c, a, b = f[0], int(f[1]), int(f[2])
            if fmt == "np" or len(f) >= 10:  # ChIP-Atlas bed05 is MACS2 narrowPeak too
                s = a + int(f[9]) if len(f) > 9 and int(f[9]) >= 0 else (a + b) // 2
                sc = float(f[6])
            else:  # ChIP-Atlas bed05: chrom start end name score(-10log10 q); no summit -> midpoint
                s = (a + b) // 2
                sc = float(f[4]) if len(f) > 4 else 0.0
            rows.append((c, a, b, s, sc))
    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "summit", "score"])
    df = df[df.chrom.isin(CHROMS)]
    df = df[[not in_black(c, x) for c, x in zip(df.chrom, df.summit)]]
    # collapse duplicate summits (IDR files repeat a region for multiple summits): keep the strongest per 100 bp
    df = df.sort_values("score", ascending=False)
    key = df.chrom + ":" + (df.summit // 100).astype(str)
    return df.loc[~key.duplicated()].reset_index(drop=True)


def interval_index(df):
    return {c: (g.start.values[np.argsort(g.start.values)], g.end.values[np.argsort(g.start.values)]) for c, g in df.groupby("chrom")}


def near_any(idx, c, x, d):
    v = idx.get(c)
    if v is None:
        return False
    s, e = v
    i = np.searchsorted(s, x + d, side="right")
    lo = max(0, i - 200)
    return bool(((s[lo:i] - d <= x) & (e[lo:i] + d >= x)).any())


def frac_hit(df, idx, d=0, mid=False):
    if not len(df):
        return np.nan
    if mid:
        return float(np.mean([near_any(idx, c, x, d) for c, x in zip(df.chrom, df.summit)]))
    # interval overlap: summit-independent, test peak start..end against index
    hits = []
    for c, a, b in zip(df.chrom, df.start, df.end):
        v = idx.get(c)
        if v is None:
            hits.append(False); continue
        s, e = v
        i = np.searchsorted(s, b, side="left")
        lo = max(0, i - 200)
        hits.append(bool((e[lo:i] > a).any()))
    return float(np.mean(hits))


def bp_jaccard(a, b):
    fa, fb = TMP / "ja.bed", TMP / "jb.bed"
    for df, f in ((a, fa), (b, fb)):
        df[["chrom", "start", "end"]].sort_values(["chrom", "start"]).to_csv(f, sep="\t", header=False, index=False)
    out = sh("bash", "-c", f"bedtools jaccard -a <(bedtools merge -i {fa}) -b <(bedtools merge -i {fb})")
    return float(out.splitlines()[1].split("\t")[2])


# --- CpG-rich promoters (r16 rule)
tss = {}
with gzip.open(GTF, "rt") as fh:
    for ln in fh:
        if ln[0] == "#":
            continue
        f = ln.split("\t", 8)
        if f[2] != "transcript":
            continue
        tss.setdefault(f[0], []).append(int(f[3]) - 1 if f[6] == "+" else int(f[4]) - 1)
tss = {c: np.unique(v) for c, v in tss.items()}
FA = Fasta(str(FASTA), as_raw=True, sequence_always_upper=True)


def near_tss(c, x, d=1000):
    v = tss.get(c)
    if v is None:
        return False
    i = np.searchsorted(v, x)
    return any(abs(v[j] - x) <= d for j in (i - 1, i) if 0 <= j < len(v))


def is_cpg_prom(c, x):
    if not near_tss(c, x):
        return False
    s = FA[c][max(0, x - 500): x + 500]
    n = max(len(s), 1); g = s.count("G"); cc = s.count("C"); cg = s.count("CG")
    return (g + cc) / n > 0.5 and cg * n / max(g * cc, 1) > 0.6


# --- motifs
def fimo_hits(seqs, tag):
    fa = TMP / f"{tag}.fa"
    fa.write_text("".join(f">s{i}\n{s}\n" for i, s in enumerate(seqs)))
    args = ["fimo", "--text", "--thresh", "1e-4", "--max-strand", "--skip-matched-sequence"]
    for m in MOTIFS:
        args += ["--motif", m]
    txt = sh(*args, str(MEME_DB), str(fa))
    rows = [ln.split("\t")[:7] for ln in txt.splitlines()[1:] if ln.strip() and not ln.startswith("#")]
    h = pd.DataFrame(rows, columns=["motif", "alt", "seq", "start", "stop", "strand", "score"])
    if len(h):
        h["pos"] = (h.start.astype(int) + h.stop.astype(int)) / 2 - HW - 0.5
        h["si"] = h.seq.str[1:].astype(int)
    return h


def shuffled(seqs, tag, copies=3):
    fa = TMP / f"{tag}.fa"
    fa.write_text("".join(f">s{i}\n{s}\n" for i, s in enumerate(seqs)))
    out = sh("fasta-shuffle-letters", "-kmer", "2", "-copies", str(copies), "-seed", "1", str(fa))
    sq, cur = [], []
    for ln in out.splitlines():
        if ln.startswith(">"):
            if cur:
                sq.append("".join(cur)); cur = []
        else:
            cur.append(ln.strip())
    if cur:
        sq.append("".join(cur))
    return sq


def motif_stats(h, n):
    res = {}
    for m in MOTIFS:
        x = h[h.motif == m] if len(h) else h
        cen = x[x.pos.abs() <= 50] if len(x) else x
        fl = x[(x.pos.abs() >= 150) & (x.pos.abs() <= HW)] if len(x) else x
        res[f"{m}:central_frac"] = cen.si.nunique() / n if len(x) else 0.0
        res[f"{m}:centrality"] = (len(cen) / 101) / max(len(fl) / 202, 1e-9) if len(x) else np.nan
    return res


def motif_block(df, tag):
    t = df.head(NMOT)
    seqs = [FA[c][x - HW: x + HW] for c, x in zip(t.chrom, t.summit)]
    seqs = [s for s in seqs if len(s) == 2 * HW and s.count("N") < 50]
    n = len(seqs)
    if n < 20:
        return {"motif_n": n}
    hr = fimo_hits(seqs, tag)
    sq = shuffled(seqs, tag + "_in")
    hs = fimo_hits(sq, tag + "_shuf")
    r = {"motif_n": n}
    a, b = motif_stats(hr, n), motif_stats(hs, len(sq))
    for m in MOTIFS:
        r[f"{m}:central_frac"] = a[f"{m}:central_frac"]
        r[f"{m}:central_frac_shuf"] = b[f"{m}:central_frac"]
        r[f"{m}:enrich_vs_shuf"] = a[f"{m}:central_frac"] / max(b[f"{m}:central_frac"], 1e-9)
        r[f"{m}:centrality"] = a[f"{m}:centrality"]
    # position histogram for MAZ 0.A (profile plot)
    x = hr[hr.motif == MOTIFS[0]] if len(hr) else hr
    r["_maz_pos"] = x.pos.values if len(x) else np.array([])
    return r


# --- bigwig helpers
BWS = {}


def bw(name):
    if name not in BWS:
        BWS[name] = pyBigWig.open(str(D / "bw" / f"{name}.bw"))
    return BWS[name]


def bw_mean(b, chrom, pos, half):
    out = np.full(len(pos), np.nan)
    ch = b.chroms()
    for i, (c, x) in enumerate(zip(chrom, pos)):
        if c not in ch or x - half < 0 or x + half > ch[c]:
            continue
        v = b.stats(c, int(x - half), int(x + half), type="mean")[0]
        out[i] = 0.0 if v is None else v
    return out


def centre_flank(b, chrom, pos):
    cen = bw_mean(b, chrom, pos, 100)
    fl = (bw_mean(b, chrom, np.asarray(pos) - 1500, 500) + bw_mean(b, chrom, np.asarray(pos) + 1500, 500)) / 2
    return np.nanmean(cen) / max(np.nanmean(fl), 1e-9)


def main():
    peaks = {}
    for name, cell, assay, grp, pf, fmt, bwn in SETS:
        if not (D / "peaks" / pf).exists():
            print("missing", pf, file=sys.stderr); continue
        peaks[name] = read_peaks(pf, fmt)
        print(name, len(peaks[name]), file=sys.stderr)
    ref = peaks[REF]
    ref_idx = interval_index(ref)
    ref_top = ref.head(NTOP)
    idx = {k: interval_index(v) for k, v in peaks.items()}

    # backgrounds for KO test: random CpG-rich promoters (not within 1 kb of any MAZ peak in any set) and CTCF ref summits
    allmaz = pd.concat([v for k, v in peaks.items() if "mock" not in k and "CTCF" not in k and "MAZKO" not in k])
    allidx = interval_index(allmaz)
    cand = [(c, int(x)) for c, v in tss.items() if c in CHROMS for x in RNG.choice(v, min(len(v), 400), replace=False)]
    RNG.shuffle(cand)
    bg = []
    for c, x in cand:
        if not near_any(allidx, c, x, 1000) and is_cpg_prom(c, x):
            bg.append((c, x))
        if len(bg) >= NTOP:
            break
    bg = pd.DataFrame(bg, columns=["chrom", "summit"])
    ct = pd.read_csv(CTCF_REF, sep="\t", header=None)
    ct = ct[ct[0].isin(CHROMS)]
    ct = pd.DataFrame({"chrom": ct[0], "summit": ((ct[1] + ct[2]) // 2).astype(int)})
    ct = ct[[not near_any(allidx, c, x, 500) for c, x in zip(ct.chrom, ct.summit)]].sample(min(NTOP, len(ct)), random_state=0)

    hap = {k: bw(k) for k in ["HAP1_abMAZ_parental_rep1", "HAP1_abMAZ_parental_rep2", "HAP1_abMAZ_MAZKO_rep1", "HAP1_abMAZ_MAZKO_rep2"]}

    def ko_ratio(df):
        P = sum(bw_mean(hap[k], df.chrom.values, df.summit.values, HW) for k in ["HAP1_abMAZ_parental_rep1", "HAP1_abMAZ_parental_rep2"])
        K = sum(bw_mean(hap[k], df.chrom.values, df.summit.values, HW) for k in ["HAP1_abMAZ_MAZKO_rep1", "HAP1_abMAZ_MAZKO_rep2"])
        r = np.log2((P + 0.05) / (K + 0.05))
        return float(np.nanmedian(r)), float(np.nanmean(r > 1))

    rows, pos = [], {}
    for name, cell, assay, grp, pf, fmt, bwn in SETS:
        if name not in peaks:
            continue
        p = peaks[name]; top = p.head(NTOP)
        (OUT / "top_summits").mkdir(exist_ok=True)
        pd.DataFrame({"c": top.chrom, "a": top.summit - 100, "b": top.summit + 100, "n": [f"p{i}" for i in range(len(top))],
                      "s": 0, "st": "+"}).to_csv(OUT / "top_summits" / f"{name}.bed", sep="\t", header=False, index=False)
        r = dict(dataset=name, cell_line=cell, assay=assay, group=grp, n_peaks=len(p))
        r["frac_peaks_overlapping_ref"] = frac_hit(p, ref_idx)
        r["frac_ref_peaks_overlapped"] = frac_hit(ref, idx[name])
        r["frac_top_overlapping_ref"] = frac_hit(top, ref_idx)
        r["ref_top1000_recall_200bp"] = frac_hit(ref_top, idx[name], d=200, mid=True)
        r["jaccard_bp_vs_ref"] = bp_jaccard(p, ref)
        r["median_peak_width"] = float(np.median(p.end - p.start))
        r["CpG_promoter_frac_top"] = float(np.mean([is_cpg_prom(c, x) for c, x in zip(top.chrom, top.summit)]))
        r["promoter_1kb_frac_top"] = float(np.mean([near_tss(c, x) for c, x in zip(top.chrom, top.summit)]))
        m = motif_block(p, name)
        pos[name] = m.pop("_maz_pos", np.array([]))
        r.update(m)
        # distal-only MAZ motif (removes the promoter GC confound)
        dist = p[[not near_tss(c, x, 2000) for c, x in zip(p.chrom, p.summit)]]
        r["n_distal"] = len(dist)
        md = motif_block(dist, name + "_distal") if len(dist) >= 50 else {}
        md.pop("_maz_pos", None)
        for k in ("motif_n", f"{MOTIFS[0]}:central_frac", f"{MOTIFS[0]}:central_frac_shuf", f"{MOTIFS[0]}:enrich_vs_shuf", f"{MOTIFS[0]}:centrality",
                  f"{MOTIFS[2]}:enrich_vs_shuf", f"{MOTIFS[7]}:enrich_vs_shuf"):
            r[f"distal:{k}"] = md.get(k, np.nan)
        r["HAP1_log2_parental_over_MAZKO_median"], r["HAP1_frac_log2ratio_gt1"] = ko_ratio(top)
        if bwn:
            r["own_centre_flank"] = centre_flank(bw(bwn), top.chrom.values, top.summit.values)
        rows.append(r)
        print("done", name, file=sys.stderr)
    for nm, df in (("background_CpG_promoters_noMAZpeak", bg), ("background_CTCF_ref_summits_noMAZpeak", ct)):
        a, b = ko_ratio(df)
        rows.append(dict(dataset=nm, group="background", HAP1_log2_parental_over_MAZKO_median=a, HAP1_frac_log2ratio_gt1=b,
                         CpG_promoter_frac_top=float(np.mean([is_cpg_prom(c, x) for c, x in zip(df.chrom, df.summit)]))))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "maz_crossval_summary.tsv", sep="\t", index=False)
    np.savez(OUT / "maz_motif_positions.npz", **{k: v for k, v in pos.items()})

    # reference split: HEK293 ref peaks shared with >=3 of the 5 endogenous-tag non-HEK sets vs not shared with any
    endo = ["K562_LAP-MAZ_ENCSR643JRH", "GM12878_LAP-MAZ_ENCSR903MVU", "HepG2_FLAG-MAZ_ENCSR700PNE", "K562_FLAG-MAZ_ENCSR163IUV", "GM12878_abMAZ_ENCSR000DZA"]
    endo = [e for e in endo if e in idx]
    cnt = np.zeros(len(ref), int)
    for e in endo:
        cnt += np.array([near_any(idx[e], c, x, 200) for c, x in zip(ref.chrom, ref.summit)])
    ref = ref.assign(n_other=cnt)
    ref[["chrom", "start", "end", "summit", "score", "n_other"]].to_csv(OUT / "ref_HEK293_peaks_shared_count.tsv", sep="\t", index=False)
    srows = []
    for lab, sub in (("ref shared with >=3 endogenous non-HEK sets", ref[ref.n_other >= 3]),
                     ("ref shared with 1-2", ref[(ref.n_other >= 1) & (ref.n_other <= 2)]),
                     ("ref in no other set (HEK293-only)", ref[ref.n_other == 0])):
        top = sub.head(NTOP)
        r = dict(subset=lab, n=len(sub), median_rank_in_ref=float(np.median(sub.index.values)) if len(sub) else np.nan,
                 frac_of_ref_top1000=float(np.mean(ref.head(NTOP).index.isin(sub.index))),
                 CpG_promoter_frac_top=float(np.mean([is_cpg_prom(c, x) for c, x in zip(top.chrom, top.summit)])))
        m = motif_block(sub, "refsplit" + str(len(srows))); m.pop("_maz_pos", None)
        r.update({k: v for k, v in m.items() if k.startswith(("motif_n", MOTIFS[0], MOTIFS[2], MOTIFS[7]))})
        r["HAP1_log2_parental_over_MAZKO_median"], r["HAP1_frac_log2ratio_gt1"] = ko_ratio(top)
        r["ref_centre_flank"] = centre_flank(bw(REF), top.chrom.values, top.summit.values)
        srows.append(r)
    pd.DataFrame(srows).to_csv(OUT / "ref_split_by_sharing.tsv", sep="\t", index=False)

    # pairwise Jaccard
    names = list(peaks)
    J = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b2 in names[i + 1:]:
            J.loc[a, b2] = J.loc[b2, a] = bp_jaccard(peaks[a], peaks[b2])
    J.to_csv(OUT / "maz_pairwise_jaccard.tsv", sep="\t")

    # signal at union of top summits
    bwsets = [s[6] for s in SETS if s[6]] + EXTRA_BW
    u = pd.concat([peaks[s[0]].head(NTOP)[["chrom", "summit"]] for s in SETS if s[0] in peaks and s[3] != "control"])
    u = u.sort_values(["chrom", "summit"]).reset_index(drop=True)
    keep, last = [], {}
    for i, (c, x) in enumerate(zip(u.chrom, u.summit)):
        if c in last and x - last[c] < HW:
            continue
        keep.append(i); last[c] = x
    u = u.loc[keep].reset_index(drop=True)
    M = pd.DataFrame({b2: bw_mean(bw(b2), u.chrom.values, u.summit.values, HW) for b2 in bwsets})
    M.insert(0, "chrom", u.chrom); M.insert(1, "summit", u.summit)
    M.to_csv(OUT / "maz_union_summit_signal.tsv.gz", sep="\t", index=False)
    C = np.log1p(M[bwsets].clip(lower=0)).corr(method="spearman")
    C.to_csv(OUT / "maz_union_signal_spearman.tsv", sep="\t")
    print(S.to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
