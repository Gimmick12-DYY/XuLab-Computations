# Borzoi sequence-prior imputation

Parallel pipeline to `scBasset/` — uses **Borzoi** (Linder, Srivastava,
Kelley et al., 2023) as a **frozen bulk-derived oracle** to score per-TF
binding/accessibility at every genomic bin, then distributes that bulk
prior across cells via the data-derived cell graph.

## What this pipeline is (and isn't)

- **Frozen, pretrained, bulk-derived.** Borzoi weights were trained on
  thousands of bulk tracks (RNA-seq, ATAC, ChIP). This pipeline does **not**
  retrain. The Borzoi predictions are the "sequence prior."
- **Per-TF specificity comes from track selection, not from training.**
  `04_select_tracks.py` filters Borzoi's ~7000 output tracks by name pattern
  ("CTCF", "FOXA1", ...) and aggregates them into a per-bin scalar.
- **Bridges bulk → single-cell via the data, not via Borzoi.** Step 05 takes
  the bulk prior and combines it with `downstream/cell_graph_kneighbour.py`
  output. Where bins were observed in the user's matrix, the cell graph
  carries cell-axis structure; where bins were observed in zero cells, the
  Borzoi prior fills in scaled by per-cell library size.
- **Companion to, not replacement for, `scBasset/`.** scBasset is the
  data-only, no-bulk-priors imputer. Borzoi is the with-bulk-priors
  reference baseline. They answer different questions; both should exist.

## Why this pipeline exists

scBasset (under our no-bulk-priors framing) cannot make truly TF-specific
predictions when the input is generic scATAC — TF-specificity has to come
from somewhere, and the data alone may not carry it. Borzoi inverts the
tradeoff: the bulk prior is rich and TF-specific by construction, but it
ignores the user's single-cell signal. By combining Borzoi's per-bin scores
with the user's cell-graph propagation, we get a TF-aware imputer whose
cell-axis structure still comes from the user's data.

## Pipeline shape

```
00_inspect_rds.R          sanity-check input (shared script)
01_export_rds_to_mm.R     RDS -> mm/{matrix.mtx.gz, regions.tsv.gz, barcodes.tsv.gz}
02_prepare_windows.py     for each bin, compute 524kb Borzoi input window;
                          filter by chrom whitelist + blacklist + bounds
03_borzoi_predict.py      GPU. Slide windows along the genome, run Borzoi,
                          aggregate central output to per-(bin, track) scalar
04_select_tracks.py       Pattern-match Borzoi targets by TF name, aggregate
                          across tracks -> per-bin scalar prior
05_distribute_to_cells.py Mode B: where bins observed, use cell-graph
                          posterior (cell_graph_kneighbour); where unobserved,
                          use Borzoi prior * per-cell library factor
06_impute.py              union with raw, write standard CSR schema
```

Outputs land at `<work>/impute/{matrix_csr.npz, regions.tsv, barcodes.tsv,
meta.json}` — the cross-pipeline schema.

## Setup

```bash
# 1) conda env (PyTorch + borzoi-pytorch + pyfaidx + h5py)
conda env create -f environment.yml
conda activate borzoi

# 2) hg38 FASTA (~3 GB) -- reuse the one fetched for scBasset
ls /work/.../downstream/cache/hg38.fa.fai     # confirm it exists

# 3) Borzoi weights (~2 GB). One-shot fetch:
./scripts/fetch_borzoi_weights.sh             # -> downstream/cache/borzoi/

# 4) (Mode B prerequisite) cell-graph propagated CSR from raw matrix.
#    Run downstream/cell_graph_kneighbour.py once per dataset:
python downstream/cell_graph_kneighbour.py \
    --mm-dir   <work>/mm \
    --embeddings <path/to/scopen/factors.npz> \
    --out-dir  <work>/cell_graph
```

## Run

```bash
# end-to-end (~12-24h on 1 A100 for ~3M bins x 1 TF allowlist)
sbatch slurm/99_full_pipeline.sbatch

# or step-by-step
sbatch slurm/01_export.sbatch
sbatch slurm/02_windows.sbatch       # CPU, 16 GB,  <1 h
sbatch slurm/03_borzoi.sbatch        # GPU, 32 GB, ~6-24 h
sbatch slurm/04_tracks.sbatch        # CPU, 16 GB,  <10 min
sbatch slurm/05_distribute.sbatch    # CPU, 64 GB, <1 h
sbatch slurm/06_impute.sbatch        # CPU, 32 GB, <30 min
```

## Tuning knobs (configs/default.yaml)

| Knob                                | Effect                                                                                       |
| ----------------------------------- | -------------------------------------------------------------------------------------------- |
| `paths.borzoi_weights`              | Directory containing Borzoi checkpoints (auto-resolved to `downstream/cache/borzoi/`).        |
| `windows.input_length: 524288`      | Borzoi receptive field. Fixed by the published model; do not change.                          |
| `windows.central_output_bp: 196608` | Central output span per inference call (used as the sliding stride).                          |
| `windows.chrom_whitelist`           | Autosomes + chrX/Y by default; mirrors scBasset's filter.                                     |
| `windows.exclude_blacklist: true`   | Drop bins overlapping ENCODE hg38 blacklist (same file scBasset uses).                        |
| `borzoi.fold: 0`                    | Borzoi was released as 4 cross-validated folds. Use fold 0 by default; ensemble via `[0,1,2,3]`. |
| `borzoi.batch_size: 2`              | Inference batch size (windows per forward). 2 fits a 40 GB A100; raise for bigger GPUs.        |
| `tracks.pattern: "^CTCF($|_)"`      | Regex on Borzoi `targets_human.txt`. Picks the TF tracks for this run.                        |
| `tracks.aggregate: mean`            | How to collapse `(n_bins, n_tracks)` -> `(n_bins,)`. `mean` / `max` / `sum`.                 |
| `tracks.apply_sigmoid: true`        | Map raw Borzoi output to [0,1] before per-cell distribution.                                  |
| `distribute.mode: B`                | Cell-axis distribution strategy. Mode B = cell-graph posterior + Borzoi prior at zero bins.   |
| `distribute.observed_k: 1`          | Threshold for "observed" — bins with raw nnz >= k take the cell-graph posterior route.        |
| `distribute.threshold_mode: sparsity_match:3` | Same threshold vocab as cisTopic / PUscOpen / scBasset.                              |
| `impute.binarize_output: true`      | Store 1 at every above-threshold entry. Set false to keep continuous values.                  |
| `impute.keep_raw: true`             | Union with raw at the end via element-wise max.                                               |

## What's saved beyond the standard schema

- `<work>/predict/bin_track.h5` — `(n_bins, n_selected_tracks)` f16 Borzoi
  output. Cached for fast re-runs of step 04 with different track patterns.
- `<work>/predict/bin_score.npy` — per-bin aggregated prior, the output of step 04.
- `<work>/predict/cell_factors.npy` — per-cell library-size factor used in step 05.
- `<work>/predict/meta.json` — Borzoi version, fold(s), track allowlist, hashes.

## The honest framing

This pipeline encodes a deliberate bulk prior. That's its whole point and
its only weakness. The "fair baseline" framing means: when comparing to
scBasset and the other data-only imputers, results should be reported as a
separate row, labeled as a frozen-oracle method. Anywhere Borzoi and
scBasset *disagree* is where the science is.

## Where this fits in the broader roadmap

This is the **with-bulk-priors track** companion to `scBasset/` (the
without-bulk-priors track). Together they bracket the design space for
sequence-based per-cell imputation.
