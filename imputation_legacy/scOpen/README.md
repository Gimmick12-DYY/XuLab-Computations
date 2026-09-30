# scOpen TF imputation pipeline

A pipeline that uses [scOpen](https://github.com/CostaLab/scopen) to impute a
sparse single-cell transcription-factor binding matrix via TF-IDF weighted
regularized non-negative matrix factorization. Initial target is the CTCF
matrix of 3,031,053 1-kb bins × 10,410 cells, but nothing here is
CTCF-specific.

The workflow shape mirrors the sibling `cisTopic/` and `FITS/` pipelines so
that artefacts can be compared apples-to-apples via the shared `downstream/`
analyses.

## What "imputation" means here

scOpen factorises the binarised regions × cells matrix (after a TF-IDF
transform) as `M ≈ W H` with W (peaks × k) and H (k × cells), then
reconstructs the imputed matrix as `M_hat = W @ H`. The rank `k` is selected
automatically via a kneedle elbow over a grid of ranks (paper recommends
λ = 1 for imputation).

Step 05 in this pipeline reconstructs `W @ H` directly into the same HDF5
schema cisTopic / FITS write (`imputed_Prc_hdf5_all.h5` with
`Prc`/`regions`/`barcodes` datasets, padded to the full pre-filter region
universe), so the shared 07–11 evaluation scripts and
`downstream/compare_pos_neg.py` work unchanged.

## Why no top-N subsetting

Unlike FITS, scOpen accepts a **sparse** input matrix natively — its 10X
loader uses `scipy.io.mmread`. The post-filter universe (~119k regions ×
10,410 cells) fits comfortably in memory after TF-IDF, so we run scOpen on
the full filtered matrix. The paper benchmarks scOpen at ~3 h on
10k cells × 100k peaks with a peak ~16 GB memory footprint, so this should
be tractable on a normal compute node.

## Over-imputation in genome-wide bin data, and how to mitigate it

scOpen was designed for **peak-by-cell** matrices: every row is a candidate
open-chromatin site already filtered by bulk-level peak calling. Genome-wide
1-kb bins violate that assumption — roughly 95 % of the genome is
heterochromatin or gene desert with zero true accessibility, and scOpen's
NMF will smear small TF-IDF signals into those rows ("over-imputation"). On
the CTCF dataset this shows up as a much smoother imputed-signal
distribution than the raw, including in regions that should never carry
CTCF binding.

This pipeline exposes two complementary mitigations:

### 1. Restrict the input to a candidate-region BED (recommended)

`filter.candidate_regions_bed` (also `--candidate-bed PATH` on
`02_prepare_scopen_input.py`) drops bins that don't overlap any interval in
the supplied BED *before* scOpen sees them. The factorisation is computed
only over plausible candidate regions, so unmodeled bins stay zero in the
final imputed matrix and cannot be over-imputed.

`filter.candidate_invert: true` (or `--candidate-invert`) flips the BED
into a blacklist (drop bins that DO overlap), useful for excluding
problematic regions like the ENCODE blacklist.

Recommended candidate BEDs for CTCF in HEK293T (already cached by
`downstream/prepare_pos_neg_bins.R`):

| BED | what it captures |
|---|---|
| `SCREEN-293T-cCRE-ENCFF529BOG.bed` | 293T cCRE universe (all classes) |
| `SCREEN-293-cCRE-ENCFF439DDQ_ENCFF885SUR_ENCFF128UTY.bed` | 293 cCRE universe |
| union of the two | broadest "potentially accessible" mask, recommended for cross-cell-type robustness |

Example:

```yaml
filter:
  binarize_input:        true
  drop_zero_rows:        true
  candidate_regions_bed: /work/.../downstream/cache/SCREEN-293T-cCRE-ENCFF529BOG.bed
  candidate_invert:      false
```

After step 02 the `prepare_meta.json` reports the mask stats:

```json
"candidate_filter": {
  "candidate_regions_bed": ".../SCREEN-293T-cCRE-ENCFF529BOG.bed",
  "candidate_invert": false,
  "n_bed_intervals": 257_345,
  "n_regions_overlap": 119_356,
  "n_regions_kept": 119_356
}
```

so you can sanity-check the overlap fraction before launching the longer
NMF step.

### 2. Tune scOpen to be more conservative

If a candidate BED is not available or you want to additionally tighten the
in-painting, the most useful knobs in `scopen.*` are:

| knob | effect |
|---|---|
| `alpha` (default 1.0) | regularisation weight. Bump to **2.0–5.0** to penalise reconstruction error in low-signal rows; the NMF then puts less mass into truly-negative regions. |
| `max_n_components` (default 30) | top of the rank grid. Smaller `k` = less expressive factorisation = less aggressive in-painting. Try **15–20** for low-heterogeneity data (e.g. one cell type). |
| `estimate_rank: false` + small `n_components` | force a conservative fixed rank, useful when the kneedle picks an over-rich model. |

These can be combined with the candidate-BED filter; the BED mask is the
sharper instrument and should be the first thing to try.

## Pipeline layout

```text
scOpen/
├── environment.yml                 conda env (scopen + scientific stack + R)
├── configs/default.yaml            all knobs
├── scripts/
│   ├── 00_inspect_rds.R            read-only sanity check on the .rds
│   ├── 01_export_rds_to_mm.R       .rds -> Matrix Market + region/cell TSVs
│   ├── 02_prepare_scopen_input.py  cisTopic-style filter, write 10X-format scopen input
│   ├── 03_run_scopen.py            invoke scopen CLI (rank grid + kneedle)
│   ├── 04_select_model.py          parse error curve, plot elbow, record chosen k
│   ├── 05_impute.py                W @ H -> imputed_Prc_hdf5_all.h5 (cisTopic schema)
│   ├── 06_downstream.py            optional summary stats (no UMAP)
│   ├── 07_eval_heldout.py          AUROC/AUPRC vs baselines
│   ├── 08_compare_imputed.py       imputed-vs-original (filtered + lifted)
│   ├── 09_visualize_imputation.py  heatmaps of original vs imputed
│   ├── 10_nonzero_complexity.py    nonzero density vs thresholds
│   ├── 11_per_cell_fragmentation.py per-cell coverage before/after
│   └── _cfg.py                     config loader shared by 02-11
└── slurm/                          sbatch templates for every step
```

## One-time setup

```bash
# 1. Create the environment (installs scopen from PyPI)
conda env create -f scOpen/environment.yml
conda activate scopen

# 2. Edit configs/default.yaml
#    - paths.input_rds   (your .rds file, e.g. CTCF_bin1000_mtx.rds)
#    - paths.work_dir    (scratch directory)
#    - filter.* defaults match cisTopic and FITS (>=2 cells/region, >=5 regions/cell)
#    - scopen.* defaults: estimate_rank=true with grid 5..30 step 5, alpha=1.0
```

## Running the whole pipeline

### Locally (small tests)

```bash
cd <repo-root>
CFG=scOpen/configs/default.yaml

Rscript scOpen/scripts/00_inspect_rds.R      --input $(yq .paths.input_rds $CFG)
Rscript scOpen/scripts/01_export_rds_to_mm.R --input $(yq .paths.input_rds $CFG) \
                                             --outdir $(yq .paths.work_dir $CFG)/mm
python scOpen/scripts/02_prepare_scopen_input.py --config $CFG
python scOpen/scripts/03_run_scopen.py            --config $CFG
python scOpen/scripts/04_select_model.py          --config $CFG
python scOpen/scripts/05_impute.py                --config $CFG
python scOpen/scripts/06_downstream.py            --config $CFG   # optional
python scOpen/scripts/07_eval_heldout.py          --config $CFG   # optional
python scOpen/scripts/08_compare_imputed.py       --config $CFG   # optional
python scOpen/scripts/09_visualize_imputation.py  --config $CFG   # optional
python scOpen/scripts/10_nonzero_complexity.py    --config $CFG   # optional
python scOpen/scripts/11_per_cell_fragmentation.py --config $CFG  # optional
```

### On SLURM

```bash
cd <repo-root>
export CFG=scOpen/configs/default.yaml

sbatch scOpen/slurm/00_inspect.sbatch                            # optional
JID1=$(sbatch --parsable scOpen/slurm/01_export.sbatch)
JID2=$(sbatch --parsable --dependency=afterok:$JID1 scOpen/slurm/02_prepare.sbatch)
JID3=$(sbatch --parsable --dependency=afterok:$JID2 scOpen/slurm/03_scopen.sbatch)
JID4=$(sbatch --parsable --dependency=afterok:$JID3 scOpen/slurm/04_select.sbatch)
JID5=$(sbatch --parsable --dependency=afterok:$JID4 scOpen/slurm/05_impute.sbatch)
sbatch         --dependency=afterok:$JID5 scOpen/slurm/06_downstream.sbatch
sbatch         --dependency=afterok:$JID5 scOpen/slurm/07_eval.sbatch
sbatch         --dependency=afterok:$JID5 scOpen/slurm/08_compare.sbatch
sbatch         --dependency=afterok:$JID5 scOpen/slurm/09_visualize.sbatch
sbatch         --dependency=afterok:$JID5 scOpen/slurm/10_nonzero_complexity.sbatch
sbatch         --dependency=afterok:$JID5 scOpen/slurm/11_per_cell_fragmentation.sbatch

# Or end-to-end in one allocation:
sbatch scOpen/slurm/99_full_pipeline.sbatch
```

## Key outputs

| Path (under `paths.work_dir`) | What it is |
|---|---|
| `mm/matrix.mtx.gz`, `mm/regions.tsv.gz`, `mm/barcodes.tsv.gz` | raw matrix exported from the `.rds` |
| `scopen_input/{matrix.mtx, barcodes.tsv, peaks.bed}` | scOpen 10X input directory |
| `scopen_input/regions.filtered.tsv` | post-filter region IDs in mm coordinate format |
| `scopen_run/scOpen_peaks.txt` | W matrix (peaks × k) |
| `scopen_run/scOpen_barcodes.txt` | H matrix (k × cells) |
| `scopen_run/scOpen_error.{txt,pdf}` | rank vs reconstruction error |
| `select/model_selection.{png,tsv}` | elbow curve plus tabular form |
| `select/selected_model.meta.yaml` | the rank scOpen used |
| `impute/imputed_Prc_hdf5_all.h5` | imputed matrix in cisTopic-compatible schema |
| `eval/heldout_eval.{json,tsv,png}` | AUROC/AUPRC vs baselines (step 07) |
| `eval/compare_imputed_*` | imputed vs original metrics (step 08) |
| `eval/visualize_imputation_*` | heatmaps (step 09) |
| `eval/nonzero_complexity.*` | nonzero density vs thresholds (step 10) |
| `eval/per_cell_fragmentation.*` | per-cell coverage histograms (step 11) |

## Comparing scOpen against cisTopic / FITS

All three pipelines write `imputed_Prc_hdf5_all.h5` with the same schema, so
any of the eval scripts works against any of the HDF5s:

```bash
# Run scOpen's step 08 against the cisTopic HDF5 (and vice versa)
python scOpen/scripts/08_compare_imputed.py \
       --config scOpen/configs/default.yaml \
       --imputed-h5 /work/.../cistopic_ctcf/impute/imputed_Prc_hdf5_all.h5 \
       --out-name compare_cistopic
```

For the bin-level CTCF positive vs negative analysis (the
`Bin_posVSneg.ipynb` notebook), feed all three pipelines into the shared
comparison script:

```bash
python downstream/compare_pos_neg.py \
  --bins-tsv /work/.../downstream/bins/pos_neg_bins.tsv \
  --out-dir  /work/.../downstream/pos_neg \
  --input raw=/work/.../mm \
  --input cisTopic=/work/.../cistopic_ctcf/impute/imputed_Prc_hdf5_all.h5 \
  --input FITS=/work/.../FITS/work/ctcf/impute/imputed_Prc_hdf5_all.h5 \
  --input scOpen=/work/.../scOpen/work/ctcf/impute/imputed_Prc_hdf5_all.h5
```

## Reality check on time and memory

* The post-filter matrix (~119k regions × ~10,410 cells, sparse binary) is
  small enough that scOpen's TF-IDF + NMF should run in a few hours per rank
  and well under 32 GB RAM. With a 6-rank grid and `nc=6` the rank grid runs
  concurrently, so end-to-end Phase-1 wall clock ≈ time for the slowest rank.
* If memory becomes a problem, drop `scopen.max_n_components` (top of the
  rank grid) — runtime and memory both scale with k.
* Step 05 streams `W @ H` row-blocks into HDF5 and never materialises the
  full `R_full × C` dense matrix; gzip-4 chunking keeps the file at 10–20 GB
  even for the 3 M-region universe.

## References

* Li Z. *et al.* Chromatin-accessibility estimation from single-cell ATAC-seq
  data with scOpen. *Nature Communications* **12**, 6386 (2021).
  https://doi.org/10.1038/s41467-021-26530-2
* scOpen: https://github.com/CostaLab/scopen
