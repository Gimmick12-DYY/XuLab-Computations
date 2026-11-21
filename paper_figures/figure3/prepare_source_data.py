"""Collect the source data of every Figure 3 panel into source_data/ (run on Longleaf after the analysis pipelines).

Three kinds of entries, all recorded in source_data/MANIFEST.tsv with origin, the script that made it, and sha256:
  SOURCES  pipeline outputs in the XuLab repo ($XULAB), copied unchanged (gzip-compressed when marked .gz);
  LOCAL    tables produced on the analysis workstation, already placed in source_data/ (origin = workstation path, relative
           to the project folder); only checksummed here;
  DERIVED  small tables computed here from pipeline outputs (functions below).
Panel scripts read only source_data/, so the figure can be rebuilt from the published source data alone.
  python prepare_source_data.py
"""
import gzip
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

X = Path(os.environ.get("XULAB", "/work/users/d/y/dyy12/XuLab"))
GENCODE_GTF = Path(os.environ.get("GENCODE_GTF", "/vast/som/xujie_lab/Ref/hg38/gencode.v50.primary_assembly.annotation.gtf.gz"))
OUT = Path(__file__).resolve().parent / "source_data"
RB = "cobinding/results/RBBP4_sweep_k50_i10000_s555"
LP = "hic/work_wt293t/loops"
RBL = f"{LP}/union_5k_10k_0.7/RBBP4_sweep_k50_i10000_s555_pad30kb"
HF = "hub_figures/basic/descriptive"
SE = "results_3d_model/se_sedb"
GO = "figures/Figure3/checks/go_great_regions"

SOURCES = {   # name : (origin in $XULAB, made by)
    "rbbp4_peak_edges.tsv.gz": ("cobinding/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv", "Cicero (k = 50) on the RBBP4 peak matrix, cobinding/"),
    "rbbp4_modules.tsv": (f"{RB}/cliques.tsv", "cobinding/scripts/cluster_peaks.py (cliques of the Cicero pair graph)"),
    "rbbp4_pairs_on_loops.tsv": (f"{RBL}/RBBP4_pairs_as_loops.0.7.tsv", "hic/scripts/compare_union_pairs_and_clique_edges.py"),
    "rbbp4_module_edges_on_loops.tsv": (f"{RBL}/RBBP4_clique_edges_as_loops.0.7.tsv", "hic/scripts/compare_union_pairs_and_clique_edges.py"),
    "peakachu_5000.loops.0.7.bedpe.gz": (f"{LP}/peakachu_5000.loops.0.7.bedpe", "Peakachu, GSE216399 wild-type HEK293T Hi-C, score >= 0.7"),
    "peakachu_10000.loops.0.7.bedpe.gz": (f"{LP}/peakachu_10000.loops.0.7.bedpe", "Peakachu, GSE216399 wild-type HEK293T Hi-C, score >= 0.7"),
    "d_module_edge_bins_1000kb.tsv.gz": ("figures/Figure3/checks/d_module_edge_bins_1000kb.tsv.gz", "source_scripts/d_module_edge_bins.py"),
    "go_surviving_per_tf.tsv.gz": (f"{GO}/surviving_per_TF.tsv", "source_scripts/go_great_controls.py"),
    "go_per_tf_controls.tsv.gz": (f"{GO}/per_TF_controls.tsv.gz", "source_scripts/go_great_controls.py"),
    "go_element_counts.tsv": (f"{GO}/distal_fractions.tsv", "source_scripts/go_great_regions.py"),
    "go_null_calibration.tsv": (f"{GO}/null_calibration.tsv", "source_scripts/go_great_null.py"),
    "go_pooled_specific.tsv.gz": ("figures/Figure3/checks/go_per_tf/pooled_specific.tsv", "source_scripts/go_per_tf.py"),
    "go_pooled_conserved.tsv.gz": ("figures/Figure3/checks/go_per_tf/pooled_conserved.tsv", "source_scripts/go_per_tf.py"),
    "go_summary_specific.tsv.gz": ("figures/Figure3/checks/go_per_tf/summary_specific.tsv", "source_scripts/go_per_tf.py"),
    "go_summary_conserved.tsv.gz": ("figures/Figure3/checks/go_per_tf/summary_conserved.tsv", "source_scripts/go_per_tf.py"),
    "go_top_term_jaccard.tsv.gz": ("figures/Figure3/checks/go_per_tf/top_term_jaccard.tsv", "source_scripts/go_term_jaccard.py"),
}
LOCAL = {   # name : (origin on the workstation, made by)
    "class_annotation_counts_chromhmm.tsv": (f"{HF}/annotation_counts_chromhmm.tsv", "cobinding/hub_figures/descriptive/annot_chromhmm.py"),
    "class_module_regions_chromhmm.tsv.gz": (f"{HF}/hub_regions_chromhmm.tsv.gz", "cobinding/hub_figures/descriptive/rebuild_hub_regions_chromhmm.py"),
    "d_module_loop_coverage.tsv.gz": (f"{HF}/clique_loop_coverage.tsv", "cobinding/hub_figures/descriptive/clique_coverage.py (gzip -n)"),
    "d_per_tf_null_summary.tsv": (f"{HF}/s1d_per_tf_null.tsv", "cobinding/hub_figures/descriptive/perm_tf_long.py (summary; selects TFs for the per-edge null)"),
    "refseq_select_tss.tsv.gz": (f"{HF}/chromhmm_rebuild_inputs/rs/*.json", "UCSC API ncbiRefSeqSelect per chromosome (hg38); TSS = txStart (+) / txEnd (-)"),
    "hek293t_chromhmm18.bed.gz": (f"{HF}/chromhmm_rebuild_inputs/HEK293T_chromHMM18.bed.gz", "HEK293T ChromHMM 18-state (hg38)"),
    "f_top_shared_module_edges.tsv": ("hub_figures/basic/clique_edge_overlap/top_shared_clique_edges.tsv", "cobinding/hub_figures/clique_edge_overlap/edge_overlap.py"),
    "se_hub_edges.tsv": (f"{SE}/hub_edges_all.tsv", f"{SE}/scripts/09_hub_edges.py"),
    "se_partners.tsv": (f"{SE}/edges_exact_se.tsv", f"{SE}/scripts/02_edges_exact.py"),
    "se_partner_bands.tsv": (f"{SE}/figure/edges_in_structure.tsv", f"{SE}/scripts/06_structure_flower.py"),
    "se_enhancer_names.tsv": (f"{SE}/figure_chord_tf/enhancer_map.tsv", f"{SE}/scripts/33_enhancer_map.py"),
    "se_model_beads.tsv": (f"{SE}/model/inputs/beads.tsv", f"{SE}/scripts/03_prep_model_se.py"),
    "se_hic_10kb.npz": (f"{SE}/model/inputs/hic_10kb.npz", f"{SE}/scripts/03_prep_model_se.py (ICE-balanced GSE216399 wild-type Hi-C)"),
    "se_tf_coverage_2kb.tsv": ("results_3d_model/se_hub/figure_v5_tftracks/tf_cov_2kb.tsv", "single-cell TF fragment coverage in 2 kb bins (se_hub track step)"),
    "se_chromhmm_window.tsv": ("results_3d_model/se_hub/figure_v7_refseq/chromhmm_window.tsv", "HEK293T ChromHMM 18-state, chr19:0.43-2.25 Mb"),
    "se_refseq_select.json": ("results_3d_model/se_hub/figure_v7_refseq/refseq_select_chr19_0.43-2.25Mb.json", "UCSC API ncbiRefSeqSelect, hg38"),
    "se_hic_test_10kb.json": (f"{SE}/hic_test/results_10kb.json", f"{SE}/scripts/05_hic_test.py"),
    "se_rosette_render.png": (f"{SE}/figure_rosette_own_beads/render.png", f"{SE}/scripts/47_rosette_own_beads.py (PyMOL render step)"),
    "se_rosette_nodes.tsv": (f"{SE}/figure_rosette_own_beads/rosette_nodes.tsv", f"{SE}/scripts/47_rosette_own_beads.py (element pixel positions in the render)"),
}


def derive_module_edge_null(dst):
    """Per-edge shuffle null of panel d: for each TF with >= 100 module edges > 25 kb (d_per_tf_null_summary.tsv), the 10,000 null
    loop fractions and the observed fraction, from figures/fig3_v6/checks/null_per_edge/<TF>_<k>.npz (10 files of 1,000)."""
    T = pd.read_csv(OUT / "d_per_tf_null_summary.tsv", sep="\t").set_index("tf")
    nd = X / "figures/fig3_v6/checks/null_per_edge"
    arr = {}
    for tf in T.index[T.edges >= 100]:
        fs = sorted(nd.glob(f"{tf}_*.npz"), key=lambda f: int(f.stem.rsplit("_", 1)[1]))
        if not fs:
            continue
        arr[f"null_{tf}"] = np.concatenate([np.load(f)["null"] for f in fs])
        arr[f"obs_{tf}"] = np.array(float(np.load(fs[0])["obs"]))
        assert len(arr[f"null_{tf}"]) == 10000 and abs(arr[f"obs_{tf}"] - T.edge_obs[tf]) < 1e-12, tf
    np.savez_compressed(dst, **arr)
    print(f"module-edge null: {len(arr) // 2} TFs")
    return "$XULAB/figures/fig3_v6/checks/null_per_edge/*.npz", "per-edge module-edge shuffle (10,000 per TF); stacked here"


def derive_gene_symbols(dst):
    """GENCODE gene id (version stripped) -> gene name for the promoter genes named in f_top_shared_module_edges.tsv."""
    d = pd.read_csv(OUT / "f_top_shared_module_edges.tsv", sep="\t")
    ids = {x for c in ("region1_promoter_gene_ids", "region2_promoter_gene_ids") for v in d[c].dropna() for x in str(v).split(",") if x}
    sym = {}
    with gzip.open(GENCODE_GTF, "rt") as fh:
        for ln in fh:
            if ln.startswith("#"):
                continue
            f = ln.split("\t", 9)
            if f[2] != "gene":
                continue
            gid = re.search(r'gene_id "([^"]+)"', f[8]).group(1).split(".")[0]
            if gid in ids:
                sym[gid] = re.search(r'gene_name "([^"]+)"', f[8]).group(1)
    pd.DataFrame(sorted(sym.items()), columns=["gene_id", "symbol"]).to_csv(dst, sep="\t", index=False)
    print(f"gene symbols: {len(sym)} of {len(ids)} ids found")
    return str(GENCODE_GTF), "gene_name of each gene_id (GENCODE v50)"


def derive_tf_panel(dst):
    """The 79 TFs of the panel: every TF run through the MACS2 -> Cicero pipeline (cobinding/work/ourrun_k50/<TF>) plus RBBP4."""
    tfs = sorted({p.name for p in (X / "cobinding/work/ourrun_k50").iterdir() if p.is_dir()} | {"RBBP4"})
    pd.DataFrame({"tf": tfs}).to_csv(dst, sep="\t", index=False)
    return "$XULAB/cobinding/work/ourrun_k50/*/ + RBBP4", f"{len(tfs)} TFs"


def derive_go_themes(dst):
    """GO-slim theme of each GO term (first assignment over the surviving and q < 0.05 theme tables of go_theme_map.py)."""
    TH = pd.concat([pd.read_csv(X / GO / f, sep="\t")[["term", "theme"]] for f in ("surviving_per_TF_themes.tsv", "per_TF_q005_themes.tsv")])
    TH.drop_duplicates("term").to_csv(dst, sep="\t", index=False)
    return f"$XULAB/{GO}/surviving_per_TF_themes.tsv + per_TF_q005_themes.tsv", "source_scripts/go_theme_map.py"


LOCI = {   # Hi-C locus panels: one 3-region module each; windows = module span +- 80 kb at 10 kb resolution
    "ets2": dict(tf="ETS2", module="ETS2 module, cobinding/results/ourrun_k50/ETS2/cliques.tsv",
                 regions=["chr1:180502204-180502470", "chr1:180896339-180896797", "chr1:181172243-181172539"],
                 # circles on the refined Hi-C dot centres (results_3d_model/clique_triangle/scripts/03_refine_null.py: best 10 kb
                 # offset of each anchor within +-20 kb, ref_sA/B/C = +10/-20/-10 kb)
                 circles=[180512337, 180876568, 181162391],
                 genes=dict(ACBD6=0, XPR1=0, MR1=0, KIAA1614=1, STX6=0, IER5=1, LINC01732=1), yticks=[0, 100, 200, 300, 400],
                 xticks=[180.5, 180.7, 180.9, 181.1],
                 # as approved (clique-triangle screen): TF track = MACS2 q <= 0.05 peaks of every TF; promoter = TSS within +-2.5 kb
                 peaks="macs2", promoter_window=2500),
    "rbbp4": dict(tf="RBBP4", module="RBBP4 module C00057, cobinding/results/RBBP4_sweep_k50_i10000_s555/cliques.tsv (top-ranked in "
                                     "source_scripts/hic_rbbp4_locus_screen.py)",
                  regions=None, clique="C00057", circles=None, genes=None, yticks=None, xticks=None,
                  # as approved (RBBP4 locus screen): TF track = Cicero peaks of every TF; promoter = TSS within +-1 kb
                  peaks="cicero", promoter_window=1000),
}
MCOOL = "hic/work_wt293t/coolers-hg38/HEK293TWT-MboI-R1-filtered.mcool"
MACS2_Q005 = Path(os.environ.get("MACS2_Q005", "/vast/som/xujie_lab/TAGATG-293T/bigwig/peaks"))   # read-only lab share


def derive_hic_loci(dst):
    """Per locus: balanced 10 kb Hi-C window (ICE weights of the wild-type HEK293T mcool), all TFs' Cicero peaks in the window
    (ourrun_k50 + RBBP4 peak_annot.tsv) or MACS2 q <= 0.05 peaks of every TF (per locus config), GENCODE v46 basic protein-coding / lncRNA genes in the window; config in hic_loci.json."""
    import glob
    import json
    import h5py
    mid = lambda r: sum(map(int, r.split(":")[1].split("-"))) // 2
    peaks = []
    files = {Path(f).parent.name: f for f in glob.glob(str(X / "cobinding/work/ourrun_k50/*/*.peak_annot.tsv"))}
    files["RBBP4"] = str(X / "cobinding/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_annot.tsv")
    for tf, f in sorted(files.items()):
        for r in pd.read_csv(f, sep="\t", usecols=["peak"]).peak:
            c, se = r.split(":")
            s, e = map(int, se.split("-"))
            peaks.append((c, s, e, tf))
    PKS = {"cicero": pd.DataFrame(peaks, columns=["chrom", "start", "end", "tf"])}
    mp = []
    for f in sorted(MACS2_Q005.glob("TF.*_peaks.q005.narrowPeak")):
        if f.stat().st_size == 0:          # TFs without any q <= 0.05 peak
            continue
        d = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2])
        mp.append(pd.DataFrame({"chrom": d[0], "start": d[1], "end": d[2], "tf": f.name[3:].split("_peaks")[0]}))
    PKS["macs2"] = pd.concat(mp, ignore_index=True)
    G = []
    with gzip.open(X / "data/gencode_v46/gencode.v46.basic.annotation.gtf.gz", "rt") as fh:
        for ln in fh:
            if ln[0] == "#":
                continue
            x = ln.split("\t")
            if x[2] != "gene":
                continue
            ty = re.search('gene_type "([^"]+)"', x[8]).group(1)
            if ty in ("protein_coding", "lncRNA"):
                G.append((x[0], int(x[3]), int(x[4]), x[6], re.search('gene_name "([^"]+)"', x[8]).group(1), "pc" if ty == "protein_coding" else "lnc"))
    G = pd.DataFrame(G, columns=["chrom", "start", "end", "strand", "gene", "type"])
    M = pd.read_csv(X / RB / "cliques.tsv", sep="\t").set_index("clique_id")
    f = h5py.File(X / MCOOL, "r")
    g = f["resolutions/10000"]
    names = [n.decode() for n in g["chroms/name"][:]]
    co, b1o = g["indexes/chrom_offset"][:], g["indexes/bin1_offset"]
    cfg = {}
    for name, L in LOCI.items():
        regs = L["regions"] or sorted(M.regions[L["clique"]].split(";"), key=mid)
        c = regs[0].split(":")[0]
        ms = [mid(r) for r in regs]
        x0, x1 = (ms[0] - 80000) // 10000 * 10000, (ms[2] + 80000) // 10000 * 10000 + 10000
        ci = names.index(c)
        i0, i1 = int(co[ci]) + x0 // 10000, int(co[ci]) + (x1 - 1) // 10000 + 1
        lo, hi = int(b1o[i0]), int(b1o[i1])
        a, b, v = g["pixels/bin1_id"][lo:hi], g["pixels/bin2_id"][lo:hi], g["pixels/count"][lo:hi]
        k = b < i1
        R = np.zeros((i1 - i0, i1 - i0))
        R[a[k] - i0, b[k] - i0] = v[k]
        R[b[k] - i0, a[k] - i0] = v[k]
        w = g["bins/weight"][i0:i1]
        np.savez_compressed(dst.parent / f"hic_locus_{name}_10kb.npz", balanced=R * np.outer(w, w), start=x0, end=x1, res=10000, chrom=c)
        PK = PKS[L["peaks"]]
        pm = (PK.start + PK.end) // 2
        PK[(PK.chrom == c) & (pm >= x0) & (pm < x1)].to_csv(dst.parent / f"hic_locus_{name}_peaks.tsv", sep="\t", index=False)
        G[(G.chrom == c) & (G.end > x0) & (G.start < x1)].to_csv(dst.parent / f"hic_locus_{name}_genes.tsv", sep="\t", index=False)
        cfg[name] = dict(L, regions=regs, chrom=c, window=[x0, x1], circles=L["circles"] or ms)
    json.dump(cfg, open(dst, "w"), indent=1)
    return f"$XULAB/{MCOOL}; ourrun_k50 + RBBP4 peak_annot.tsv; {MACS2_Q005}; data/gencode_v46", "Hi-C windows, peaks and genes per locus (hic_locus_<name>_*)"


PERM = "hic/work_wt293t/loops/ourrun_tf_loops_0.7_pad30kb_perm10000/null"   # perm_tf_long.py: 10 x 1,000 shuffles per TF


def _perm(tf):
    fs = sorted((X / PERM).glob(f"{tf}_*.npz"), key=lambda f: int(f.stem.rsplit("_", 1)[1]))
    return np.vstack([np.load(f)["null"] for f in fs]), np.load(fs[0])


def derive_rbbp4_null(dst):
    """RBBP4 pair / module-edge shuffle null (10,000 x [pairs, module edges > 25 kb], observed rates, n)."""
    N, z = _perm("RBBP4")
    np.savez_compressed(dst, null=N, obs=z["obs"], n=z["n"], columns=np.array(["pairs", "module_edges"]))
    return f"$XULAB/{PERM}/RBBP4_*.npz", "hic/work_wt293t/loops/ourrun_tf_loops_0.7_pad30kb_perm10000/perm_tf_long.py; stacked"


def derive_null_percentiles(dst):
    """Per TF in d_per_tf_null_summary.tsv: percentiles (2.5, 25, 50, 75, 97.5) of the 10,000-shuffle null, in % on a loop, for pairs
    (column 0) and for module edges with whole-module block shuffles (column 1)."""
    T = pd.read_csv(OUT / "d_per_tf_null_summary.tsv", sep="\t")
    rows = []
    for tf in T.tf:
        N, _ = _perm(tf)
        for j, test in enumerate(["pairs", "module_edges_block"]):
            v = N[:, j] * 100
            rows.append(dict(tf=tf, test=test, n_shuffles=len(v), **dict(zip(["p2_5", "p25", "p50", "p75", "p97_5"], np.percentile(v, [2.5, 25, 50, 75, 97.5])))))
    pd.DataFrame(rows).to_csv(dst, sep="\t", index=False, float_format="%.6f")
    return f"$XULAB/{PERM}/*.npz", "perm_tf_long.py (same-chromosome shuffles keeping spacing); percentiles here"


DERIVED = {"rbbp4_shuffle_null.npz": derive_rbbp4_null, "s_shuffle_null_percentiles.tsv": derive_null_percentiles, "d_module_edge_null.npz": derive_module_edge_null, "f_gene_symbols.tsv": derive_gene_symbols, "tf_panel.tsv": derive_tf_panel,
           "go_term_themes.tsv": derive_go_themes, "hic_loci.json": derive_hic_loci}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for name, (origin, made_by) in SOURCES.items():
        src, dst = X / origin, OUT / name
        if name.endswith(".gz") and not origin.endswith(".gz"):
            with open(src, "rb") as fi, gzip.GzipFile(dst, "wb", mtime=0) as fo:
                shutil.copyfileobj(fi, fo)
        else:
            shutil.copyfile(src, dst)
        rows.append((name, f"$XULAB/{origin}", made_by))
    for name, (origin, made_by) in LOCAL.items():
        if not (OUT / name).exists():
            sys.exit(f"missing workstation source file {name} (from {origin})")
        rows.append((name, f"workstation:{origin}", made_by))
    for name, fn in DERIVED.items():
        origin, made_by = fn(OUT / name)
        rows.append((name, origin, made_by))
        if name == "hic_loci.json":
            rows += [(f.name, origin, "derive_hic_loci() in this script") for f in sorted(OUT.glob("hic_locus_*"))]
    with open(OUT / "MANIFEST.tsv", "w") as fh:
        fh.write("file\torigin\tmade_by\tsha256\n")
        for r in sorted(rows):
            fh.write("\t".join(r) + "\t" + sha256(OUT / r[0]) + "\n")
    print(f"{len(rows)} source files in {OUT}")


if __name__ == "__main__":
    main()
