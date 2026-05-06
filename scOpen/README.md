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
