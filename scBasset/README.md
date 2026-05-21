# scBasset sequence-based imputation

Parallel pipeline to `cisTopic/`, `FITS/`, `scOpen/`, `MAGIC/`, `PUscOpen/`,
`Cicero/`. Implements scBasset (Yuan & Kelley, *Nature Methods*, 2022) — a
CNN that maps **DNA sequence + per-cell embedding -> accessibility**.

## Why this pipeline exists

Every other imputer in this repo (cisTopic, FITS, scOpen, MAGIC, PUscOpen,
Cicero) hits the same wall: their model components — LDA topics, NMF
factors, cell-cell adjacencies, peak-peak co-accessibility — are all
learned from the *observed* matrix, so they cannot put signal at a bin that
was zero in **every** cell. They can only redistribute / propagate signal
among bins that already exist somewhere.

scBasset breaks that ceiling. Once trained, the model takes a bin's DNA
sequence and a cell's 32-d embedding and outputs `sigmoid(W_c · embed(seq))`
— **a real prediction for any bin in the genome, including ones the
training matrix never observed.** This is the only pipeline in our stack
that fills truly-new bins.

## Pipeline shape

```
00_inspect_rds.R      sanity-check input (same script as cisTopic)
01_export_rds_to_mm.R RDS -> mm/{matrix.mtx.gz, regions.tsv.gz, barcodes.tsv.gz}
02_prepare_seqs.py    pad each bin to seq_length, one-hot encode hg38 -> seqs.h5
                      + matrix_train.npz (subset bins with cells >= min_cells_per_peak_train)
03_train.py           Keras scBasset, conv tower -> 32-d embed -> cell-wise sigmoid (GPU)
04_predict.py         predict on every bin in seqs.h5; two-pass threshold via
                      sparsity_match (or quantile / fixed) (GPU)
05_impute.py          persist mm-aligned matrix_csr.npz + regions.tsv + barcodes.tsv + meta.json
```

Outputs land at `<work>/impute/{matrix_csr.npz, regions.tsv, barcodes.tsv,
meta.json}` — the standard cross-pipeline schema.

## Setup

```bash
# 1) conda env (TensorFlow + pyfaidx + samtools + h5py)
conda env create -f environment.yml
conda activate scbasset

# 2) hg38 FASTA (~3 GB). Pick one:
./scripts/fetch_hg38_fasta.sh                    # downloads + builds .fai
#  OR point paths.genome_fa at an existing hg38.fa that already has hg38.fa.fai
```

## Run

```bash
# end-to-end (~24h on 1 GPU + 128 GB for ~119k trainable bins x 10k cells,
# then ~3M-bin genome-wide prediction)
sbatch slurm/99_full_pipeline.sbatch

# or step-by-step
sbatch slurm/01_export.sbatch
sbatch slurm/02_seqs.sbatch         # CPU,  64 GB, ~2-6 h
sbatch slurm/03_train.sbatch        # GPU, 128 GB, ~6-24 h
sbatch slurm/04_predict.sbatch      # GPU, 128 GB, ~2-12 h
sbatch slurm/05_impute.sbatch       # CPU,  64 GB, <1 h
```

## IGV / BigWig

Use the shared converter under `downstream/` (same script as PUscOpen / Cicero CSR imputes):

```bash
cd /work/users/d/y/dyy12/XuLab
INPUT_DIR=scBasset/work/ctcf/impute \
OUTPUT_BW=scBasset/work/ctcf/bigwig/scbasset_ctcf_mean.bw \
AGGREGATE=mean NORMALIZE=p99 \
sbatch downstream/slurm/imputed_to_bigwig.sbatch
```

Or run `downstream/imputed_matrix_to_bigwig.py` directly with `--input` / `--output` /
`--chrom-sizes`. Input must be `<work>/impute/` with `matrix_csr.npz` + `regions.tsv`.

Load in IGV: genome **hg38**, then **File → Load from File** → the `.bw` path.

For a smoke test:
- `seqs.bin_bed: <path>` restricts to a small candidate BED (e.g. a single
  chromosome), shrinking 02/03/04 by 10-100x.
- `train.epochs: 10` + `train.patience: 3` finishes 03 in minutes.

## Hooking into the cross-pipeline comparison

```bash
python downstream/compare_pos_neg.py \
  --input cisTopic=/path/to/cisTopic/work/ctcf/impute \
  --input fits=/path/to/FITS/work/ctcf/impute \
  --input scopen=/path/to/scOpen/work/ctcf/impute \
  --input magic=/path/to/MAGIC/work/ctcf/impute \
  --input puscopen=/path/to/PUscOpen/work/ctcf/impute \
  --input cell_kgraph=/path/to/downstream/cell_kgraph \
  --input cicero=/path/to/Cicero/work/ctcf/impute \
  --input scbasset=/path/to/scBasset/work/ctcf/impute \
  ...
```

## Tuning knobs (configs/default.yaml)

| Knob                            | Effect                                                                                |
| ------------------------------- | ------------------------------------------------------------------------------------- |
| `seqs.seq_length: 1344`         | scBasset paper default. 1kb bin + 172 bp flanks each side.                            |
| `seqs.min_cells_per_peak_train: 2` | Bins with fewer observed cells are still predicted but not in the training loss.   |
| `seqs.bin_bed: null`            | null = predict on all kept bins. Restrict to a cCRE/motif BED for fast smoke tests.   |
| `seqs.predict_bed: null`        | Predict-time motif filter. null = auto-detect `downstream/cache/CTCF_motif_hg38.bed`; "off" disables; or set an explicit BED path. Training is unaffected. |
| `model.latent_dim: 32`          | Sequence embedding + per-cell embedding dimension. scBasset paper default.            |
| `model.tower_channels`          | Conv tower widths after the stem (288 -> 512 geometric).                              |
| `train.epochs: 500` + `patience: 10` | Train with early stopping on val_loss.                                           |
| `train.batch_size: 64`          | Bin batch size; one batch = one matmul (B, latent) @ (latent, n_cells).               |
| `predict.threshold_mode: sparsity_match:3` | Match raw nnz x 3 (same vocabulary as cisTopic / PUscOpen). Raise K for more calls. |
| `predict.predict_all_bins: true` | Run inference on every bin in seqs.h5 (not just trainable ones).                     |
| `impute.binarize_output: true`  | Store 1 at every above-threshold entry. Set false to keep the sigmoid value.          |
| `impute.keep_raw: true`         | Union with raw at the end via element-wise max.                                       |

## What's saved beyond the standard schema

- `<work>/model/cell_embeddings.npz` — `W: (latent_dim, n_cells)`, `b: (n_cells,)`.
  These are the *learned* cell embeddings; they can replace the scOpen-derived
  embeddings as input to `downstream/cell_graph_kneighbour.py`:
  ```
  python downstream/cell_graph_kneighbour.py \
    --mm-dir /.../mm \
    --embeddings /.../scBasset/work/ctcf/model/cell_embeddings.npz \
    --embeddings-barcodes /.../scBasset/work/ctcf/seqs/barcodes.tsv \
    ...
  ```
- `<work>/predict/predictions_csr.npz` — the thresholded sigmoid output before
  the union with raw, useful for ablations.
- `<work>/model/history.json` + `training.csv` — per-epoch loss / AUROC.

## Where this fits in the broader roadmap

This is item **4** of the "discover new bins" roadmap.

1. **Motif-aware propagation** — `downstream/fetch_ctcf_motif_bed.sh`
   feeds `cisTopic/05_impute.py` and `PUscOpen/06_impute.py` via
   auto-detected `target_bed`.
2. **Cell-graph kNN propagation** — `downstream/cell_graph_kneighbour.py`
   propagates along the cell axis.
3. **Cicero co-accessibility** — `Cicero/`. Replaces the fixed-window
   bin-bin adjacency with a learned one.
4. **Sequence-based imputation (scBasset)** — *this pipeline*. The only
   approach that can fill bins observed in **zero** training cells.
