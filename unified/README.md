# unified — single-architecture TF binding imputer

A PyTorch pipeline that imputes sparse single-cell TF / chromatin matrices
with **one jointly trained model**. It is intentionally **not** a cascade of
scBasset → cisTopic → PUscOpen (or similar): each cascade stage is replaced by
a differentiable submodule that shares one loss and one set of gradients.

Typical use: start from a sparse peak×cell (or bin×cell) count matrix, train
on a GPU, and write a binarized imputed CSR in the same schema used by the
other imputers in this repo (`matrix_csr.npz` + `regions.tsv` + `barcodes.tsv`
+ `meta.json`).

---

## What this share pack contains (and what it does not)

Portable archive (repo path):
[`../share/unified_imputation_pipeline.zip`](../share/unified_imputation_pipeline.zip)

| Included | Not included (you must supply) |
|----------|--------------------------------|
| `README.md`, `environment.yml` | Genome FASTA (+ `.fai` index) |
| `configs/` (YAML templates) | Input count matrix (RDS and/or Matrix Market) |
| `scripts/` (`00`–`03`, `_model.py`, helpers) | Open-chromatin BED (optional post-filter) |
| `slurm/` (job templates) | ENCODE blacklist BED (optional; can auto-download) |
| | Trained checkpoints, `work/` outputs, example data |

Unpack:

```bash
unzip unified_imputation_pipeline.zip -d /path/to/dest
cd /path/to/dest/unified
```

All absolute paths in `configs/*.yaml` are **examples from our cluster**. Edit
them for your machine before any run (see [Configuration](#configuration)).

---

## Inputs you must provide

### 1. Sparse count matrix (required)

Preferred on-disk layout under `<work_dir>/mm/`:

| File | Description |
|------|-------------|
| `matrix.mtx.gz` | Matrix Market; rows = genomic bins/peaks, cols = cells; non-negative counts |
| `regions.tsv` or `regions.tsv.gz` | One region per matrix row, format `chr:start-end` (0- or 1-based consistent with your coords; must match FASTA chromosome names, e.g. `chr1`) |
| `barcodes.tsv` or `barcodes.tsv.gz` | One barcode / cell id per matrix column |

If you start from an R `dgCMatrix` RDS, export to that layout first (any
export that writes the three files above is fine). In the full XuLab tree
this is often done with `cisTopic/scripts/01_export_rds_to_mm.R`; that script
is **not** inside this zip—bring your own export or copy that helper.

### 2. Reference genome FASTA (required for ingest)

- Contig names must match `regions` (e.g. `chr1`…`chrY`).
- Indexed with `samtools faidx your_genome.fa` (produces `your_genome.fa.fai`).
- Set `paths.genome_fa` in the YAML to that FASTA.

Ingest takes a centered window of `seqs.seq_length` bp (default **768**) around
each bin and one-hot encodes it. Bins with too many `N`s, off-whitelist
chromosomes, or (optionally) blacklist overlap are dropped from training
tensors; imputation still aligns back to full `mm` row order.

### 3. Optional BEDs

| Config key | Role | If omitted |
|------------|------|------------|
| `seqs.blacklist_bed_gz` | Drop bins overlapping ENCODE-style blacklist during ingest | If `exclude_blacklist: true` and path is null, ingest may download hg38 blacklist v2; set `exclude_blacklist: false` to skip |
| `impute.open_chromatin_bed` | After scoring, zero **new imputed** calls outside open chromatin; raw observations kept when `keep_raw: true` | Set to `null` or `"off"` — recommended when you have no ATAC/DNase peaks for the same system |

Neither FASTA nor any BED is shipped in the zip. Open-chromatin masking is a
post-hoc biological prior, not part of the neural architecture; disable it for
a prior-free run.

---

## Model structure (detailed)

Implementation: [`scripts/_model.py`](scripts/_model.py). Default hyperparameters
live under `model:` / `train:` in [`configs/default.yaml`](configs/default.yaml).

### Design idea

Sparse TF matrices are ~99% zero. A plain dense reconstructor either collapses
to “predict zero everywhere” or needs cascade heuristics (topic models, PU
seeds, NMF postfilters). This model instead:

1. Embeds each bin from **DNA sequence**.
2. Refines that embedding with **multi-scale genomic neighbors** on the same
   chromosome (differentiable; no LDA).
3. Scores each (bin, cell) with a **bilinear cell bank**.
4. Multiplies by a **learned gate** (confidence) so the final score is
   \(\hat y_{rc} = g_{rc}\, p_{rc}\).

Everything is trained jointly from the user’s matrix only (no motif BEDs, no
pretrained sequence weights, no TF identity input).

### Submodules

| Module | Role | Default shape / ops |
|--------|------|---------------------|
| **SequenceEncoder** | Basset-style CNN: one-hot DNA → latent `D` | Input `(B, L, 4)` with `L=768`. Stem `Conv1d(4→288, k=17)` + MaxPool(3); tower channels `[288,323,363,407]` each `k=5` + MaxPool(2); `1×1` to 256 filters; flatten → Dropout(0.2) → Linear → `D=64`. ~spatial length 16 → flat 4096 before the head. |
| **BinContext** | Multi-scale 1-D conv **along genome-ordered bins** within a chrom chunk | Three dilated branches, `k=3`, dilations `[1,4,16]` → receptive fields **3 / 9 / 33 bins** (~3 / 9 / 33 kb if bins are 1 kb). Concat → GELU → `1×1` mixer → residual `e + α ⊙ h` with learnable per-channel `α ∈ R^D` (init 0.1). |
| **Cell bank `U`, bias `b`** | Per-cell embedding in the same `D`-space | `U ∈ R^{C×D}`, `b ∈ R^C`, small Gaussian init on `U`. |
| **Bilinear head** | Binding probability before gating | \(p_{rc} = \sigma(\tilde e_r · u_c + b_c)\). |
| **Gate MLP** | Per-(bin,cell) confidence | Default: `concat[\tilde e_r ; u_c]` → Linear+GELU stack `[128,64,32]` → 1 → sigmoid. Optional **sparsity-aware** extras: `log1p(cell library size)` and/or `log1p(raw count at (r,c))` (`model.sparsity_aware`). |

Approximate trainable size at defaults: **~5.8M** parameters (dominated by the
sequence tower; cell bank scales with `C × D`).

### Forward pass (two-stage)

Training and imputation share the same factorization so the CNN is not
re-run for every sampled pair:

```
# Stage A — once per contiguous chrom chunk (bins in genomic order)
e_r      = SequenceEncoder(seq_r)                 # (n_bins, D)
ẽ_r      = BinContext(e_r)                        # residual multi-scale mix

# Stage B — many (bin, cell) pairs, or full dense block at impute time
p_rc     = sigmoid(ẽ_r · u_c + b_c)
g_rc     = sigmoid(GateMLP(features(ẽ_r, u_c, …)))
ŷ_rc     = g_rc * p_rc
```

API in code: `encode_bins(seq)` then `forward_pairs(...)` (train) or
`predict_chunk(...)` (impute).

### Why multiple dilations

One architecture covers sharp motif factors and broad chromatin factors by
letting channels mix scales via `α`:

| Binding style (examples) | Branch that tends to dominate | Context (1 kb bins) |
|--------------------------|-------------------------------|---------------------|
| Sharp / motif (CTCF, REST) | d=1 | ~3 kb |
| Enhancer-clustered | d=4 | ~9 kb |
| Broad domain (EZH2, …) | d=16 | ~33 kb |

No TF id is fed in; scale selection is learned per channel from the matrix.

### Loss and training mechanics

Configured under `train:`:

- **Focal BCE** (`γ=2`, `α=0.25`) on observed positives vs **uniform-sampled
  negatives** (`n_neg_per_pos: 5`). Dense BCE over all `R×C` is intractable.
- **Pairwise ranking** (RankNet-style softplus on logits), weight
  `ranking_loss_weight` (default 1.0); optional grading of positives by
  `log1p` observation count.
- **L2** weight decay on parameters; **L1** on `α` (`l1_alpha`) to prefer
  sparse scale use.
- **RC augmentation** (`rc_augment_prob: 0.5`): randomly replace the DNA
  window with its reverse complement during training only.
- **Chunked by chromosome**: `chunk_size` contiguous bins (default 4096) so
  BinContext never crosses chrom boundaries.
- Early stopping on held-out **bin** fraction (`val_frac`, `patience`).

Hard-negative mining is implemented but **off** by default
(`hard_negative_frac: 0`): on this task “hard negatives” are often dropouts or
discoverable sites, and mining them hurts coverage/AUROC.

### Imputation post-processing (not part of the net)

After GPU scoring (`02_impute.py`):

1. Threshold scores (`threshold_mode`, default `sparsity_match:3` — same
   vocabulary as other pipelines in this lab).
2. Optional open-chromatin BED mask on **imputed-only** entries.
3. Optional `keep_raw: true` → element-wise max with the observed matrix so
   raw positives are never deleted.

Outputs under `<work_dir>/impute/`.

---

## Pipeline stages

| Step | Script | Compute | What it does |
|------|--------|---------|--------------|
| Export | (external R / your tool) | CPU | RDS or other → `mm/{matrix.mtx.gz,regions*,barcodes*}` |
| 00 | `scripts/00_ingest.py` | CPU | FASTA + `mm/` → `seqs/seqs.h5`, aligned `matrix.npz`, `counts.npz`, `cell_depth.npy`, `chrom_ranges.json` |
| 01 | `scripts/01_train.py` | **GPU** | Train `GatedUnifiedModel`; write `model/ckpt.pt` |
| 02 | `scripts/02_impute.py` | **GPU** | Score all bins×cells; threshold; optional OCR mask; write `impute/` |
| 02b | `scripts/02b_apply_open_chromatin_mask.py` | CPU | Remask an existing CSR from `matrix_csr.npz.pre_openmask` (no retrain) |
| 03 | `scripts/03_export_impute_rds.py` | CPU | Optional: CSR → RDS for R workflows |

Work directory layout (created by the config helper):

```
<work_dir>/
  mm/        # your input Matrix Market bundle
  seqs/      # ingest outputs
  model/     # ckpt.pt
  impute/    # matrix_csr.npz, regions.tsv, barcodes.tsv, meta.json
  logs/
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate unified
# Needs: Python 3.11, numpy/scipy/pandas/h5py/pyyaml/tqdm/pyfaidx, samtools, PyTorch ≥2.2, einops
```

GPU strongly recommended for steps 01–02. Export-from-RDS (if you use the
XuLab R helper) needs a separate R environment with `Matrix` (e.g. the lab’s
`cistopic` env)—not part of `environment.yml`.

---

## Configuration

Edit a YAML under `configs/` (start from `default.yaml` or a per-TF copy).

**Paths you must set:**

```yaml
paths:
  input_rds:  /path/to/YourTF_bin_mtx.rds   # only needed if your SLURM/export step reads RDS
  work_dir:   /path/to/unified_work/your_tf # all stage I/O roots here
  genome_fa:  /path/to/hg38.fa              # or mm10.fa, etc.; must have .fai
```

**Recommended first-run settings when you have no ATAC BED:**

```yaml
impute:
  open_chromatin_bed: null    # or "off"
  keep_raw: true
```

Other useful knobs: `seqs.seq_length`, `seqs.chrom_whitelist`,
`model.latent_dim`, `train.epochs`, `train.n_neg_per_pos`,
`impute.threshold_mode`. Per-TF YAMLs in `configs/` are historically full
copies; they deep-merge onto `default.yaml` for any keys you omit.

---

## Run (manual, path-agnostic)

```bash
cd /path/to/unified
conda activate unified

# 1) Place or export Matrix Market files into $WORK/mm/
# 2) Point configs/default.yaml paths.work_dir and paths.genome_fa

python scripts/00_ingest.py --config configs/default.yaml
python scripts/01_train.py  --config configs/default.yaml   # GPU
python scripts/02_impute.py --config configs/default.yaml   # GPU
```

Overrides without editing YAML:

```bash
python scripts/00_ingest.py --config configs/default.yaml --work-dir /path/to/work/your_tf
```

**SLURM:** templates under `slurm/` assume a conda module and (for
`99_full_pipeline.sbatch`) an R export script next to this tree. Adjust
`#SBATCH` lines, `CONDA_ENV`, `EXPORT_RSCRIPT`, and `CFG=configs/your_tf.yaml`.
Multi-TF driver: `slurm/run_all_tfs.sh`.

Smoke-check the model dimensions (no data needed):

```bash
python scripts/_model.py
```

---

## Output schema

`<work_dir>/impute/`:

| File | Meaning |
|------|---------|
| `matrix_csr.npz` | scipy CSR, shape `(n_bins_mm, n_cells)`, typically binarized |
| `regions.tsv` | Row names in **original mm order** |
| `barcodes.tsv` | Column barcodes |
| `meta.json` | Threshold mode, paths, run metadata |
| `matrix_csr.npz.pre_openmask` | Present if an open-chromatin mask was applied (for 02b remask) |

---

## What this is not

- **Not a cascade.** One loss; every submodule gets gradients.
- **Not TF-conditioned.** No TF embedding or label in the forward pass.
- **Not PU-learning.** The gate is supervised + differentiable (no Elkan–Noto
  seed, no scOpen NMF stage).
- **Not bulk-prior-dependent for training.** Motif/cCRE/ATAC BEDs are not
  required to train; ATAC BED is an optional impute-time filter only.
- **Not a data drop.** The zip is code + configs only—bring genome and matrix.

---

## Relation to other lab pipelines

Cascade bake-offs (scBasset, cisTopic, PUscOpen, Borzoi, …) live under
`imputation_legacy/` in the full XuLab repo. Downstream comparators that
accept `--input unified=<work>/impute` expect the schema above. This README
is the source of truth for architecture and for running from the share zip.
