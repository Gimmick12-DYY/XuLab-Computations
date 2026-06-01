# scBasset_TF — TF-focused fork of scBasset

A parallel pipeline to `scBasset/` with two structural changes targeted at
TF-binding-like single-cell data:

1. **Adaptive pooling architecture.** The 7-stage MaxPool chain in the
   original is replaced by a single stem MaxPool plus a **dilated conv
   tower**, ending in **GlobalAveragePooling1D**. The model no longer
   depends on a specific input length — any `seq_length` works with the
   same architecture file. Default `seq_length` drops from 1344 bp to
   **768 bp** because TF-binding context rarely needs >800 bp.
2. **Peak *or* bin input.** The pipeline accepts either a peak matrix
   (variable-width regions) or a bin matrix (fixed 1 kb tiles), with no
   code branch — the mm/ contract is identical. `run.input_kind` is
   recorded for downstream comparison.

Track A's data-only stance is preserved: `predict_bed: "off"` by default,
no bulk priors anywhere in the pipeline.

## What this fork is (and isn't)

- **Architecturally length-agnostic.** Pure-conv tower with dilations, then
  GlobalAvgPool. Same trained model can score windows of any length; same
  config file generalizes to any `seq_length`.
- **Not a replacement for scBasset/.** The original stays as the baseline
  with fixed 1344 bp windows. scBasset_TF is the experimental fork for the
  TF-narrower-window question.
- **Peak and bin inputs ship as separate runs.** Combinatorial training
  (multi-task / late fusion / sequential) is planned but deferred. For now:
  run once with the peak RDS, once with the bin RDS, compare downstream.
- **Per-cell metadata is eval-only.** The TSV at `paths.cell_metadata` is
  validated against barcodes but never enters the loss. Hooks for
  batch-correction or metadata-conditioned `W_c` can be added later.

## Pipeline shape

Identical structure to `scBasset/`:

```
00_inspect_rds.R       sanity-check (shared script)
01_export_rds_to_mm.R  RDS -> mm/{matrix.mtx.gz, regions.tsv.gz, barcodes.tsv.gz}
02_prepare_seqs.py     center-extract `seq_length` bp per region; one-hot
03_train.py            dilated tower + GlobalAvgPool (GPU)
04_predict.py          genome-wide prediction; predict_bed: "off" by default (GPU)
05_impute.py           assemble standard CSR schema
```

## Setup

```bash
conda env create -f environment.yml
conda activate scbasset_tf
# Reuse scBasset's hg38 FASTA -- nothing to refetch.
```

## Run a peak-input experiment

```yaml
# configs/default.yaml
paths:
  input_rds:  /path/to/CTCF_peaks_mtx.rds
  work_dir:   /path/to/work/ctcf_peak
run:
  input_kind: peak
seqs:
  seq_length: 768
```

```bash
sbatch slurm/99_full_pipeline.sbatch
```

## Run a bin-input experiment

Same pipeline; just swap the RDS and the work_dir:

```yaml
paths:
  input_rds:  /path/to/CTCF_bin1000_mtx.rds
  work_dir:   /path/to/work/ctcf_bin
run:
  input_kind: bin
```

## Architecture, in one place

```
Input(shape=(seq_length, 4))               # any length; e.g. 768
  Conv1D(288, 17, padding=same)
  BN -> GELU
  MaxPool1D(3)                              # single fixed downsample
  for (C, d) in zip([288,323,363,407,456,512], [2,4,8,16,32,64]):
      Conv1D(C, 5, padding=same, dilation=d)
      BN -> GELU
  Conv1D(256, 1)
  BN -> GELU
  GlobalAveragePooling1D                    # collapses any spatial dim
  Dropout(0.2)
  Dense(32)                                 # sequence embedding
  BN -> GELU
  Dense(n_cells)                            # cell projection (W: 32 x n_cells)
  Sigmoid
```

Receptive field: stem 17 bp + (5-1)*(2+4+8+16+32+64) at the 1/3 scale
= ~17 + 504 = ~521 bp at the post-stem resolution. Times 3 (stem MaxPool)
= ~1.5 kb effective receptive field at the input. Plenty for TF context;
no longer scales with `seq_length`.

## Tuning knobs (configs/default.yaml)

| Knob                              | Effect                                                                                |
| --------------------------------- | ------------------------------------------------------------------------------------- |
| `run.input_kind: peak`            | Cosmetic tag; recorded in meta.json. No code branches on it.                          |
| `seqs.seq_length: 768`            | Window length. Any value works — architecture is length-agnostic.                     |
| `seqs.predict_bed: "off"`         | Track A inherited. Bulk-prior leak is opt-in only.                                    |
| `model.stem_pool: 3`              | Single fixed downsample after the stem. Set 1 to disable.                             |
| `model.tower_dilations`           | Per-block dilation rates; control receptive field growth.                             |
| `model.pool: global_avg`          | `global_avg` (default) or `global_max` at the head.                                   |
| `paths.cell_metadata: null`       | Optional per-cell TSV. Validated against barcodes; eval-only.                         |
| `predict.threshold_mode: sparsity_match:3` | Same vocabulary as scBasset / cisTopic / PUscOpen.                           |
| `impute.binarize_output: true`    | Store 1 at above-threshold entries.                                                   |
| `impute.keep_raw: true`           | Union with raw via element-wise max.                                                  |

## How this fork is compared to scBasset/

The cross-pipeline harness (`downstream/compare_pos_neg.py`) takes one
`--input scbasset_tf=<work>/impute` arg per run, so peak and bin
experiments are independent rows. Add `--input scbasset=...` to include
the original 1344 bp pipeline in the same comparison.

For sequence-prior cross-check, also add `--input borzoi=...` and look at
disagreements as discussed in `Borzoi/README.md`.

## Status of deferred work

- **Combinatorial training (peak + bin).** Architecture is single-input
  single-head for now. Multi-task wiring (one tower, two heads) and late
  fusion will be added once we have peak-only and bin-only runs to compare.
- **Metadata as conditioning input.** Eval-only today. Wiring to add
  metadata as a batch-correction signal or `W_c` augmentation will be a
  follow-up.
- **Variable-length batching.** All bins in one run currently share
  `seq_length`. The architecture supports variable length within a batch,
  but the data loader does not yet group by length.
