# FITS TF imputation pipeline

A pipeline that uses [FITS (Forest of Imputation Trees)](https://github.com/reggenlab/FITSpython)
to impute a sparse single-cell transcription-factor binding matrix. The initial
target is a CTCF matrix of 3,031,053 1-kb bins × 10,410 cells stored as an
`.rds` file, but nothing here is CTCF-specific.

The workflow shape mirrors the sibling `cisTopic/` pipeline so that artefacts
from the two methods can be compared apples-to-apples.

## What "imputation" means here

FITS grows ensembles of imputation trees over the cell × site dense matrix and
returns a per-entry imputed value (Phase1 builds the forest; Phase2 produces
the imputed dense matrix). Step 05 wraps the Phase2 output in the same HDF5
schema cisTopic writes (`imputed_Prc_hdf5_all.h5` with `Prc`/`regions`/
`barcodes` datasets, padded to the full pre-filter region universe), so the
shared evaluation scripts (07–11) work on either pipeline.

## Why the FITS subset

cisTopic models the entire post-filter region universe (~119k regions for
CTCF). FITS expects a dense input and Phase1L grows trees row-wise, so running
on 119k rows is intractable at this scale. Step 02 therefore takes a top-N
subset of the filtered regions (default 10k, configurable in
`prepare.n_regions`) and step 05 pads imputed values back into the full
pre-filter region space; rows outside the FITS subset are stored as zeros and
compress to almost nothing under gzip.

## Pipeline layout

```text
FITS/
├── environment.yml                 conda env
├── configs/default.yaml            all knobs
├── scripts/
│   ├── 00_inspect_rds.R            read-only sanity check on the .rds
│   ├── 01_export_rds_to_mm.R       .rds -> Matrix Market + region/cell TSVs
│   ├── 02_prepare_fits_input.py    cisTopic-style filter, subset top-N for FITS
│   ├── 03_run_fits_phase1.py       FITS Phase1 (forest construction)
│   ├── 04_run_fits_phase2.py       FITS Phase2 (imputation)
│   ├── 05_impute.py                wrap Phase2 output -> imputed_Prc_hdf5_all.h5
│   ├── 06_downstream.py            optional summary stats (no UMAP/topics)
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
# 1. Create the environment
conda env create -f FITS/environment.yml
conda activate fits

# 2. Clone FITSpython
git clone https://github.com/reggenlab/FITSpython
# point `paths.fits_repo` in configs/default.yaml at that clone

# 3. Edit configs/default.yaml
#    - paths.input_rds   (your .rds file, e.g. CTCF_bin1000_mtx.rds)
#    - paths.work_dir    (scratch directory; lots of space needed)
#    - paths.fits_repo
#    - filter.* defaults match cisTopic (>=2 cells per region, >=5 regions/cell)
#    - prepare.n_regions controls the top-N FITS subset (default 10000)
```

## Running the whole pipeline

### Locally (small tests)

```bash
cd <repo-root>
CFG=FITS/configs/default.yaml

Rscript FITS/scripts/00_inspect_rds.R      --input $(yq .paths.input_rds $CFG)
Rscript FITS/scripts/01_export_rds_to_mm.R --input $(yq .paths.input_rds $CFG) \
                                           --outdir $(yq .paths.work_dir $CFG)/mm
python FITS/scripts/02_prepare_fits_input.py    --config $CFG
python FITS/scripts/03_run_fits_phase1.py       --config $CFG
python FITS/scripts/04_run_fits_phase2.py       --config $CFG
python FITS/scripts/05_impute.py                --config $CFG
python FITS/scripts/06_downstream.py            --config $CFG   # optional
python FITS/scripts/07_eval_heldout.py          --config $CFG   # optional
python FITS/scripts/08_compare_imputed.py       --config $CFG   # optional
python FITS/scripts/09_visualize_imputation.py  --config $CFG   # optional
python FITS/scripts/10_nonzero_complexity.py    --config $CFG   # optional
python FITS/scripts/11_per_cell_fragmentation.py --config $CFG  # optional
```

### On SLURM

```bash
cd <repo-root>
export CFG=FITS/configs/default.yaml

sbatch FITS/slurm/00_inspect.sbatch                              # optional
JID1=$(sbatch --parsable FITS/slurm/01_export.sbatch)
JID2=$(sbatch --parsable --dependency=afterok:$JID1 FITS/slurm/02_prepare.sbatch)
JID3=$(sbatch --parsable --dependency=afterok:$JID2 FITS/slurm/03_phase1.sbatch)
JID4=$(sbatch --parsable --dependency=afterok:$JID3 FITS/slurm/04_phase2.sbatch)
JID5=$(sbatch --parsable --dependency=afterok:$JID4 FITS/slurm/05_impute.sbatch)
sbatch         --dependency=afterok:$JID5 FITS/slurm/06_downstream.sbatch
sbatch         --dependency=afterok:$JID5 FITS/slurm/07_eval.sbatch
sbatch         --dependency=afterok:$JID5 FITS/slurm/08_compare.sbatch
sbatch         --dependency=afterok:$JID5 FITS/slurm/09_visualize.sbatch
sbatch         --dependency=afterok:$JID5 FITS/slurm/10_nonzero_complexity.sbatch
sbatch         --dependency=afterok:$JID5 FITS/slurm/11_per_cell_fragmentation.sbatch

# Or end-to-end in one allocation:
sbatch FITS/slurm/99_full_pipeline.sbatch
```

## Key outputs

| Path (under `paths.work_dir`) | What it is |
|---|---|
| `mm/matrix.mtx.gz`, `mm/regions.tsv.gz`, `mm/barcodes.tsv.gz` | raw matrix exported from the `.rds` |
| `fits_input/fits_input.csv` | dense FITS input (top-N filtered regions × cells) |
| `fits_input/regions.filtered.tsv` | region IDs that passed the filter (pre-subset) |
| `fits_input/regions.selected.tsv` | region IDs in the FITS subset |
| `fits_input/barcodes.tsv` | cell barcodes after the cell filter |
| `fits_run/FITS_OUTPUT*` | FITS Phase1 forest artefacts |
| `fits_run/<phase2_output>.csv` | FITS Phase2 imputed dense matrix |
| `impute/imputed_Prc_hdf5_all.h5` | imputed matrix in cisTopic-compatible schema |
| `eval/heldout_eval.{json,tsv,png}` | AUROC/AUPRC vs baselines (step 07) |
| `eval/compare_imputed_*` | imputed vs original metrics (step 08) |
| `eval/visualize_imputation_*` | heatmaps (step 09) |
| `eval/nonzero_complexity.*` | nonzero density vs thresholds (step 10) |
| `eval/per_cell_fragmentation.*` | per-cell coverage histograms (step 11) |

## Comparing FITS against cisTopic

Both pipelines write `imputed_Prc_hdf5_all.h5` with the same dataset/attribute
schema. To compare on the same matrix:

```bash
# Run cisTopic 08 with the FITS HDF5 (and FITS 08 with the cisTopic HDF5)
python cisTopic/scripts/08_compare_imputed.py \
       --config cisTopic/configs/default.yaml \
       --imputed-h5 /work/.../FITS/work/ctcf/impute/imputed_Prc_hdf5_all.h5 \
       --out-name compare_fits
```

The 08 script reports `filtered_space` (where the model predicted) and
`lifted_space` (full original universe) separately, so the two pipelines can
be compared on either footing.

## References

* Mishra & Sengupta. FITS: Forest of Imputation Trees for recovering true
  signals in single-cell open chromatin profiles. *NAR Genomics and
  Bioinformatics* (2021). https://github.com/reggenlab/FITSpython
* Bravo González-Blas C. *et al.* cisTopic: cis-regulatory topic modeling on
  single-cell ATAC-seq data. *Nature Methods* **16**, 397–400 (2019).
