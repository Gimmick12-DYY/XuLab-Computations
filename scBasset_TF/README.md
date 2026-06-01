# scBasset_TF — sparse-TF-binding imputation built on the Basset CNN

## Goal of this pipeline

Given a sparse single-cell TF-binding matrix (regions × cells, either bin- or
peak-level), produce a denoised and augmented matrix that:

1. **Retrieves nonzero entries at regions that were observed in zero cells**
   in the raw matrix — the "new bin" problem that propagation-based imputers
   (cisTopic, scOpen, MAGIC, kNN) cannot reach, because they only redistribute
   existing signal.
2. **Denoises the original matrix** by smoothing sparse positive evidence
   into a calibrated per-cell probability.
3. **Generalizes across TFs in the same dataset** by training one model on the
   AllTF matrix; per-TF behavior emerges from the data, not from external
   priors.

The pipeline must not rely on bulk-derived motif BEDs, pretrained models, or
TF-specific annotations. Sequence is the only external information allowed.

## Why a new pipeline (and not just scBasset)

scBasset (Yuan & Kelley 2022) is the right architectural family — a Basset
CNN over DNA with a per-cell sigmoid head — but it has three properties that
work against the goal above when applied directly to TF binding data:

| scBasset assumption | Problem for sparse TF data |
|---|---|
| Trained on binarized scATAC accessibility | The model learns "is this peak accessible," not "is this TF bound." Architectural bias is identical; training signal is not. |
| Fixed 1344 bp window centered on peak summit | Peak summits are well-defined; bin midpoints are **not**. A 1 kb bin's actual TF binding event could be near either edge or just outside, so 1344 bp centered on the bin midpoint may miss it. |
| BCE loss on a ~99 % zero matrix | Plain BCE is dominated by easy negatives; gradient at the rare 1s is washed out. Sparse TF data sharpens this problem vs. accessibility. |

scBasset_TF addresses all three:

1. **TF-binding-tuned defaults**: `seq_length: 384` (peak-summit-centric) with
   architecture and dilations sized accordingly. For bin input, `seq_length:
   1024` is the recommended override to defeat bin-center ambiguity.
2. **Adaptive pooling**: the MaxPool chain is replaced by a single stem
   `MaxPool(3)` + dilated conv tower + `GlobalMaxPooling1D`. Any `seq_length`
   works; nothing in the layer math is divisibility-bound. `GlobalMaxPool` is
   the right inductive bias for sparse localized signals (a motif hit
   anywhere in the window fires the pooled output).
3. **Sparsity-aware loss**: focal loss with γ=2, α=0.25 by default. Directly
   attacks the gradient-washout problem in extreme class imbalance. `bce` and
   `bce_weighted` are config-switchable alternatives for ablation.

## What this fork is (and isn't)

- **A data-only, sequence-driven imputer.** No bulk priors, no pretrained
  weights, no motif BEDs in the default path (`predict_bed: "off"`).
- **TF-aware via Y, not via architecture.** The model has no TF
  conditioning. Per-TF behavior comes from the AllTF matrix already carrying
  per-TF signal in the cells (verified via metadata grouping in
  `downstream/eval_basic.py`). Multi-task TF-conditioned head is a planned
  follow-up; not implemented yet.
- **Companion to scBasset/, not a replacement.** scBasset stays as the
  original 1344 bp / MaxPool-chain baseline. scBasset_TF is the
  TF-binding-tuned variant.
- **Per-cell metadata is eval-only.** The TSV at `paths.cell_metadata` is
  validated against barcodes but never enters the loss. It enables per-TF
  stratification in downstream evaluation.

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

Ready-to-use preset configs ship in `configs/`. Pick one and pass it via `CFG`:

```bash
# Peak input — 384 bp / 4-stage dilated tower / global_max / focal loss
CFG=configs/alltf_peak.yaml sbatch slurm/99_full_pipeline.sbatch

# Bin input — 1024 bp / 6-stage dilated tower / same head + loss
CFG=configs/alltf_bin.yaml sbatch slurm/99_full_pipeline.sbatch
```

Each preset is a complete drop-in config. The only thing you should ever
need to edit is `paths.work_dir` if you want to keep multiple parallel runs.
`configs/default.yaml` mirrors `alltf_peak.yaml` and is what the SLURM
scripts fall back to if `CFG` is unset.

## Architecture, in one place

```
Input(shape=(seq_length, 4))               # any length; default 384
  Conv1D(288, 17, padding=same)            # stem: 17-bp motif-sized filters
  BN -> GELU
  MaxPool1D(3)                              # single fixed downsample
  for (C, d) in zip([288,323,363,407], [2,4,8,16]):
      Conv1D(C, 5, padding=same, dilation=d)
      BN -> GELU
  Conv1D(256, 1)
  BN -> GELU
  GlobalMaxPooling1D                        # sparse-signal-aware pooling
  Dropout(0.2)
  Dense(32)                                 # sequence embedding
  BN -> GELU
  Dense(n_cells)                            # cell projection (W: 32 x n_cells)
  Sigmoid
```

Receptive field at the default 4-stage dilation `[2, 4, 8, 16]`:
stem 17 bp + 3 × (5-1) × (2+4+8+16) = ~17 + 360 ≈ **390 bp at input scale**,
matched to the peak default `seq_length: 384`. For wider windows, extend
dilations and channels in lockstep so RF ≥ seq_length.

## Tuning knobs (configs/default.yaml)

| Knob                              | Effect                                                                                |
| --------------------------------- | ------------------------------------------------------------------------------------- |
| `run.input_kind: peak`            | Tag recorded in meta.json. No code branches on it; informs per-run comparisons.       |
| `seqs.seq_length: 384`            | Window length. Peak default. Override to 1024+ for bin input (see table below).       |
| `seqs.min_cells_per_peak_train: 3`| Regions observed in fewer cells are still predicted but not in the loss.              |
| `seqs.predict_bed: "off"`         | Track A inherited. Bulk-prior leak is opt-in only.                                    |
| `model.stem_pool: 3`              | Single fixed downsample after the stem. Set 1 to disable.                             |
| `model.tower_dilations: [2,4,8,16]` | Per-block dilation rates; control receptive field growth. Extend for wider windows.|
| `model.pool: global_max`          | `global_max` (default, motif-friendly) or `global_avg` (accessibility-friendly).      |
| `train.loss: focal`               | `focal` (default) / `bce_weighted` / `bce`. The sparsity attack.                      |
| `train.pos_weight: auto`          | Used when `loss == bce_weighted`. `auto` reads pos_weight_auto from seqs/meta.json.   |
| `train.focal_gamma: 2.0`          | Focal loss focusing parameter. 2 is the canonical default (Lin et al. 2017).          |
| `train.focal_alpha: 0.25`         | Focal loss positive-class weight. 0.25 also from the focal-loss paper.                |
| `paths.cell_metadata`             | Optional per-cell TSV. Validated against barcodes; eval-only.                         |
| `predict.threshold_mode: sparsity_match:3` | Same vocabulary as scBasset / cisTopic / PUscOpen.                           |
| `impute.binarize_output: true`    | Store 1 at above-threshold entries.                                                   |
| `impute.keep_raw: true`           | Union with raw via element-wise max.                                                  |

## Per-input-kind tuning quick reference

| Knob                  | Peak input                  | Bin input                              |
|-----------------------|-----------------------------|----------------------------------------|
| `seqs.seq_length`     | 384                         | 1024 (over-cover the 1 kb tile)        |
| `model.tower_channels`| `[288, 323, 363, 407]`      | `[288, 323, 363, 407, 456, 512]`       |
| `model.tower_dilations` | `[2, 4, 8, 16]`            | `[2, 4, 8, 16, 32, 64]`                |
| `model.pool`          | `global_max`                | `global_max`                           |
| `train.loss`          | `focal`                     | `focal`                                |
| `run.input_kind`      | `peak`                      | `bin`                                  |

The peak defaults are in the default config. For the bin variant, copy the
config and override the four model/seqs knobs above.

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
