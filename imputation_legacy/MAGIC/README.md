# MAGIC TF imputation pipeline

A pipeline that uses [MAGIC (Markov Affinity-based Graph Imputation of Cells)](https://github.com/KrishnaswamyLab/MAGIC)
to impute a sparse single-cell transcription-factor binding matrix via data
diffusion on a cell-cell graph. Initial target is the CTCF matrix of
3,031,053 1-kb bins × 10,410 cells, but nothing here is CTCF-specific.

The workflow shape mirrors the sibling `cisTopic/`, `FITS/`, and `scOpen/`
pipelines so that the output drops directly into the shared
`downstream/compare_pos_neg.py` cross-pipeline comparison.

## What "imputation" means here

MAGIC builds a kNN affinity matrix between cells (in PCA space), turns it
into a Markov transition matrix `M`, raises it to a small power `t`, and
returns `M^t @ X` — a smoothed, denoised version of the input where each
cell's profile is a weighted average over its diffusion neighbourhood. The
diffusion time `t` is auto-selected via a Procrustes-style residual elbow
(default `t='auto'`).

Step 05 transposes MAGIC's `(cells × features)` output back to the standard
`(regions × cells)` orientation and persists it as
`<work>/impute/matrix.npy + regions.tsv + barcodes.tsv + meta.json`, the
same dense schema FITS uses.

## Pre-processing

Same minimal pre-processing as `scOpen/` and `FITS/`:

* binarise the input (count > 0 → 1)
* drop all-zero rows (uninformative for the diffusion)

We do NOT apply the cisTopic-style `min_cells_per_region` quality filter —
MAGIC's diffusion is designed to handle sparsity directly.

## Pipeline layout

```text
MAGIC/
├── environment.yml                 conda env (magic-impute + scientific stack + R)
├── configs/default.yaml            all knobs
├── scripts/
│   ├── 00_inspect_rds.R            read-only sanity check on the .rds
│   ├── 01_export_rds_to_mm.R       .rds -> Matrix Market + region/cell TSVs
│   ├── 02_prepare_magic_input.py   binarise + drop zero rows -> sparse npz
│   ├── 03_run_magic.py             MAGIC().fit_transform on (cells x regions)
│   ├── 05_impute.py                transpose + rescale -> impute/matrix.npy
│   └── _cfg.py                     config loader shared by 02-05
└── slurm/                          sbatch templates for every step
```

(There is no `04_*` step — MAGIC produces the imputed matrix directly from
`fit_transform`, with no separate model-selection phase.)

## One-time setup

```bash
# 1. Create the environment (installs magic-impute from PyPI)
conda env create -f MAGIC/environment.yml
conda activate magic

# 2. Edit configs/default.yaml
#    - paths.input_rds   (your .rds file, e.g. CTCF_bin1000_mtx.rds)
#    - paths.work_dir    (scratch directory)
#    - magic.* knobs     (knn=5, decay=1, t=auto, n_pca=100 are sane defaults)
```

## Running the whole pipeline

### Locally (small tests)

```bash
cd <repo-root>
CFG=MAGIC/configs/default.yaml

Rscript MAGIC/scripts/00_inspect_rds.R      --input $(yq .paths.input_rds $CFG)
Rscript MAGIC/scripts/01_export_rds_to_mm.R --input $(yq .paths.input_rds $CFG) \
                                            --outdir $(yq .paths.work_dir $CFG)/mm
python MAGIC/scripts/02_prepare_magic_input.py --config $CFG
python MAGIC/scripts/03_run_magic.py           --config $CFG
python MAGIC/scripts/05_impute.py              --config $CFG
```

### On SLURM

```bash
cd <repo-root>
export CFG=MAGIC/configs/default.yaml

sbatch MAGIC/slurm/00_inspect.sbatch                         # optional
JID1=$(sbatch --parsable MAGIC/slurm/01_export.sbatch)
JID2=$(sbatch --parsable --dependency=afterok:$JID1 MAGIC/slurm/02_prepare.sbatch)
JID3=$(sbatch --parsable --dependency=afterok:$JID2 MAGIC/slurm/03_magic.sbatch)
sbatch         --dependency=afterok:$JID3 MAGIC/slurm/05_impute.sbatch

# Or end-to-end in one allocation:
sbatch MAGIC/slurm/99_full_pipeline.sbatch
```

## Key outputs

| Path (under `paths.work_dir`) | What it is |
|---|---|
| `mm/matrix.mtx.gz`, `mm/regions.tsv.gz`, `mm/barcodes.tsv.gz` | raw matrix exported from the `.rds` |
| `magic_input/matrix.npz` | sparse binarised regions × cells (input to step 03) |
| `magic_input/{regions,barcodes}.tsv` | post drop-zero region IDs and cell barcodes |
| `magic_run/imputed_cells_x_regions.npy` | raw MAGIC output (cells × regions, float32) |
| `magic_run/meta.json` | params used + auto-selected `t` |
| `impute/matrix.npy` | final imputed matrix (regions × cells, float32) |
| `impute/{regions,barcodes}.tsv` | row / col labels |
| `impute/meta.json` | scale_factor, source paths, shape |

## Adding to the cross-pipeline comparison

The `downstream/compare_pos_neg.py` script auto-detects the `matrix.npy`
schema, so adding MAGIC to an existing comparison is just one extra
`--input`:

```bash
python downstream/compare_pos_neg.py \
  --bins-tsv .../downstream/bins/pos_neg_bins.tsv \
  --out-dir  .../downstream/pos_neg \
  --input raw=.../mm \
  --input cisTopic=.../cistopic_ctcf/impute \
  --input FITS=.../FITS/work/ctcf/impute \
  --input scOpen=.../scOpen/work/ctcf/impute \
  --input MAGIC=.../MAGIC/work/ctcf/impute
```

## Reality check on time and memory

* **Affinity step**: PCA of cells × regions to `n_pca` (100), then a
  10k × 10k cell-cell affinity. Trivial in time and memory (< 1 min,
  hundreds of MB).
* **Diffusion**: `M^t` for small `t` (auto typically picks 3–7) is a
  handful of 10k × 10k dense matrix multiplications. Seconds.
* **Imputation step**: `M^t @ X` where X is `(cells × regions)`. The
  output is dense `(n_cells × n_regions × 4)` bytes. For ~10k cells:
  - 100k regions → ~4 GB
  - 600k regions (CTCF post drop-zero) → ~24 GB
* **Step 05 storage**: the `matrix.npy` on disk is the same size as the
  in-memory dense output (no compression). Plan disk accordingly.

End-to-end on the CTCF dataset should fit comfortably under 6 h with
128 GB; MAGIC was the fastest imputer in the scOpen paper benchmarks at
this scale.

## References

* Van Dijk D. *et al.* Recovering Gene Interactions from Single-Cell Data
  Using Data Diffusion. *Cell* **174**, 716–729 (2018).
  https://doi.org/10.1016/j.cell.2018.05.061
* MAGIC: https://github.com/KrishnaswamyLab/MAGIC
