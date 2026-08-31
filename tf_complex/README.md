# TF co-occupancy → candidate protein complexes

Standalone pipeline (Image-#11 checklist): **A/B compartments → correlate TFs by
co-occupancy → shared motif → protein complex**. Independent of the Cicero
higher-order co-binding module (`cobinding/`) — this is a parallel analysis of the
whole TF panel.

## Method — aligned with the literature

Finding protein complexes from binding maps is a solved, standard problem. The
canonical reference is **Partridge et al. 2020, *Nature*** ("Occupancy maps of 208
chromatin-associated proteins in one human cell type"): build a **loci × protein**
binding matrix, remove high-occupancy artifacts, correlate proteins, and cluster →
complexes (they recovered cohesin, NuRD, POL2/TSS). We follow that method, with two
corrections the literature demands:

1. **Co-occupancy is measured over called PEAKS**, not genome-wide signal bins —
   peaks are far less accessibility-driven. Peaks are called per-TF at MACS3
   **q < 0.05** (the project standard, `downstream/peak_coverage.py`) from **raw**
   pseudobulk.
2. **HOT (high-occupancy-target) loci are removed.** A handful of loci — active
   promoters / super-enhancers — are bound by nearly *every* TF and are largely
   ChIP-seq artifacts (GC/CpG-rich, motif-free; Wreczycka et al. 2019). They create
   a universal "rich-club" that makes every TF pair look co-bound. We drop loci
   bound by > `HOT_FRAC` of the panel, plus the ENCODE blacklist.

**Metric = `pearson`** (phi of the binary peak vectors, default) or **`jaccard`**.
Both recover complexes across the panel's ~1000× peak-count spread *without* a
coverage block, once HOT loci are removed — validated on a synthetic with a deep
(high-peak) and a shallow (low-peak) complex at realistic universe scale. **A
PCA/SVD step was tried and removed**: row-centering the loci made sparse low-peak
TFs collinear, producing a spurious ~36-TF "complex" that just tracked peak count.
The `[genome] coverage check:` log line reports the low-peak-TF block median (near 0
= clean) so any residual depth artifact is visible.

**A/B compartments are used as an ANNOTATION** of the resulting complexes/TFs
(which complex is A- vs B-biased), **not** as the axis that defines co-binding:
every TF binds active chromatin, so every TF is A-leaning and A/B cannot
discriminate complexes. This is why the earlier "A/B compartment correlation"
framing did not work.

### Why RAW binding, not imputed
The imputed matrices are **open-chromatin masked**, which forces every TF into the
same open bins and would inflate co-occupancy. Peaks come from the **raw** per-TF
pseudobulk (`unified/work/<tf>/mm`). See [[imputed-peaks-accessibility-wash]].

## Pipeline

```bash
mamba env create -f tf_complex/environment.yml -p /work/users/d/y/dyy12/conda/envs/tf_complex

# optional but recommended: ENCODE blacklist (skipped automatically if absent)
mkdir -p tf_complex/ref && cd tf_complex/ref
wget https://github.com/Boyle-Lab/Blacklist/raw/master/lists/hg38-blacklist.v2.bed.gz
gunzip hg38-blacklist.v2.bed.gz && cd -

# 1. per-TF raw peaks (array job, one TF per task; set --array=0-<N-1>)
sbatch tf_complex/slurm/01_call_raw_peaks.sbatch

# 2. union peaks -> HOT removal -> pearson correlation -> complexes  (needs hic step 05 A/B)
sbatch tf_complex/slurm/02_tf_complexes.sbatch
#   METRIC=jaccard HOT_FRAC=0.4 CLUSTER_THRESHOLD=0.6 sbatch tf_complex/slurm/02_tf_complexes.sbatch
#   REBUILD=0 sbatch tf_complex/slurm/02_tf_complexes.sbatch   # reuse peak_matrix.npz
```

### 1. `call_raw_peaks.py` — per-TF raw peaks
MACS3 q<0.05 on each TF's raw pseudobulk → `work/peaks/<tf>.bed`. Reuses the vetted
`downstream/peak_coverage.py` standard (synthetic fragments → MACS3 `--nolambda`).

### 2. `build_peak_matrix.py` — union → loci × TF matrix
Merges all per-TF peaks into consensus **loci** (bedtools-merge style); each locus
records which TFs bind it → `work/peak_matrix.npz` (loci × TF bool) + `loci.bed`
(with occupancy) + `tfs.txt` + `loci_ab.tsv` (per-locus A/B label).

### 3. `tf_complexes.py` — HOT removal → correlation → complexes
Drops HOT loci (`--hot-frac`) + blacklist + singletons (`--min-occ`), computes the
TF×TF similarity, clusters → complexes, and annotates A/B. Outputs:
- `tf_similarity_{genome,A,B}.tsv` — TF×TF similarity (chosen metric).
- **`candidate_complexes.tsv`** — clustered TF groups (complex_id, n_tfs, mean_intra_sim, members).
- **`tf_compartment.tsv`** — per-TF A/B preference, size-normalized.
- `complex_heatmap.png` (`--plot`), clustered.

## Knobs (env vars)

| var | default | meaning |
|-----|---------|---------|
| `WORK_ROOT` | `unified/work` | per-TF raw `mm` matrices |
| `AB_BED` | `hic/work/compartments/compartments_25000.AB.bed` | A/B calls (hic step 05) |
| `BLACKLIST` | `tf_complex/ref/hg38-blacklist.v2.bed` | ENCODE blacklist (optional) |
| `MACS_Q` | `0.05` | per-TF peak-calling q-value |
| `METRIC` | `pearson` | `pearson` (phi of binary peak vectors) \| `jaccard` |
| `HOT_FRAC` | `0.5` | drop loci bound by > this fraction of TFs (HOT artifacts) |
| `MIN_OCC` | `2` | keep loci bound by ≥ this many TFs |
| `CLUSTER` | `hierarchical` | `hierarchical` (avg-linkage cut) \| `threshold` (connected comps) |
| `CLUSTER_THRESHOLD` | `0.5` | similarity cut; set near the printed **p90/p95** |
| `REBUILD` | `1` | `0` = reuse `peak_matrix.npz` |

## Reading the result
- **`candidate_complexes.tsv`** = TF groups that co-occupy → protein-complex candidates.
  Validate by checking a known HEK293T complex comes out together.
- The `[genome] … off-diag: median/p90/p95` diagnostic sets `CLUSTER_THRESHOLD`
  (real complexes sit in the p90–p95 tail; median should be near 0 after HOT removal).
- Compare **A vs B** similarity matrices for compartment-specific complexes.
- **`tf_compartment.tsv`** gives each TF's A/B lean (size-normalized).

Next in the checklist: for each candidate complex, test **shared/same motif**
(HOMER/AME on the group's peaks, reuse `downstream/`), then **define the protein
complex**.

## Tracking
`tf_complex/work/` (peaks + matrix) and `results/` are gitignored; scripts / slurm /
env / README tracked.
