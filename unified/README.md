# unified — single-architecture TF binding imputer

Parallel pipeline to `scBasset/`, `scBasset_TF/`, `cisTopic/`, `PUscOpen/`,
`Borzoi/`. **Not** a cascade or remix of those — it's a single trainable
model that replaces each cascade stage with a differentiable submodule, all
optimized jointly on the same loss.

## The architectural decomposition

scBasset_cisTopic_PUscOpen has the best peak-coverage and AUROC numbers we've
measured, but at the cost of three sequential pipelines, leak-prone external
seeds, heuristic windows, and no joint optimization. The unified model
extracts the operation each cascade stage performs and rebuilds it as a
learnable submodule:

| Submodule | Replaces (cascade) | Caveat dropped |
|---|---|---|
| SequenceEncoder | scBasset CNN tower | Trained jointly with downstream heads; less BCE-collapse |
| BinContext | cisTopic LDA φ + neighborhood propagation | Differentiable, multi-scale, no heuristic window |
| Cell bank `U` | cisTopic θ + scBasset cell weights | Trained against the same loss as the bin encoder |
| Gate MLP | PU classifier + scOpen NMF + postfilter | Per-cell aware (was per-bin), no external seed, no leak path |
| Bilinear `ẽ·u` head | scBasset cell head + scOpen reconstruction | One operation, one set of gradients |

## Architecture

```
INPUTS
  seq_r ∈ {ACGT}^768          # 768 bp DNA window per bin
  X_rc ∈ {0, 1}                # sparse observed matrix

PARAMETERS  (~5.8M trainable)
  SequenceEncoder θ_seq         Basset-style conv tower → Dense(D=64)
  BinContext θ_ctx              3 dilations [1, 4, 16] + 1×1 mixer
  Per-channel residual α        ∈ R^64, init 0.1
  Cell bank U                   ∈ R^(C × 64)  (learnable)
  Gate MLP θ_gate               concat[ẽ_r ; u_c] → 128 → 64 → 32 → 1
  Per-cell bias b_c             ∈ R^C

FORWARD
  e_r  = SeqEncoder(seq_r)                                # (R, D)
  ẽ_r  = e_r + α ⊙ MultiScaleContext(e_r along chrom)     # (R, D)
  p_rc = sigmoid(ẽ_r · u_c + b_c)                         # bilinear prediction
  g_rc = sigmoid(MLP_gate(concat(ẽ_r, u_c)))              # confidence gate
  ŷ_rc = g_rc * p_rc

LOSS
  focal(γ=2, α=0.25) on observed positives
   + n_neg=5 uniform-sampled negatives per positive
   + L2 on (U, θ_seq), L1 on α
```

## TF generalization

No TF-specific constant is baked in. Multi-scale dilations let one model
handle the spread of TF binding patterns:

| TF class | Active branch | Effective context |
|---|---|---|
| Sharp / motif-defined (CTCF, REST) | branch 1 (k=3, d=1) | ~3 kb |
| Enhancer-clustered (FOXA1, GATA, OCT4) | branch 2 (k=3, d=4) | ~9 kb |
| Promoter-bound (TBP, POL2) | branch 1 | ~3 kb |
| Broad chromatin (EZH2, CBX) | branch 3 (k=3, d=16) | ~33 kb |
| Multi-TF (AllTF runs) | mix learned per channel | varies |

## Pipeline shape

```
00_ingest.py     mm/ + hg38.fa → seqs.h5 + matrix.npz (binary)
                  + counts.npz (raw ints) + cell_depth.npy + chrom_ranges.json
01_train.py      chunked training, focal + ranking loss (GPU).
                  With model.sparsity_aware: gate also sees log1p(cell depth)
                  and log1p(raw count); positives loss-weighted by strength.
02_impute.py     forward over all bins, threshold via sparsity_match,
                  **open-chromatin mask** (drop new calls outside HEK293T
                  OmniATAC IDR optimal peaks), then keep_raw union (GPU)
02b_apply_open_chromatin_mask.py
                  CPU post-hoc remask of an existing impute CSR (no retrain)
```

Open-chromatin mask (default on via `impute.open_chromatin_bed` in
`configs/default.yaml`):

- BED: `data/HEK293T_OmniATAC_idr_optimal.narrowPeak.gz` (~72k IDR optimal)
- Semantics: **imputed-only** — new model calls outside open chromatin are
  deleted; observed raw signal outside ATAC is kept when `keep_raw: true`
- Disable with `open_chromatin_bed: null` (or `"off"`)

Post-hoc remask all existing TFs (array job, CPU):

```bash
N=$(ls -d unified/work/*/impute/matrix_csr.npz | wc -l)
sbatch --array=0-$((N-1)) unified/slurm/02b_apply_open_chromatin_mask.sbatch
```

Outputs land at `<work>/impute/{matrix_csr.npz, regions.tsv, barcodes.tsv,
meta.json}` — same schema as scBasset / cisTopic / PUscOpen, so the existing
`downstream/compare_pos_neg.py` and `peak_coverage.py`
harnesses ingest it directly.

## Setup

```bash
conda env create -f environment.yml
conda activate unified

# Reuse the hg38 FASTA fetched for scBasset
ls /work/.../downstream/cache/hg38.fa.fai
```

## Run

```bash
# SLURM (CTCF bin matrix; exports RDS -> mm, then 00/01/02)
cd /work/.../XuLab/unified
sbatch slurm/99_full_pipeline.sbatch

# Manual (after RDS -> mm export)
conda activate unified
Rscript ../cisTopic/scripts/01_export_rds_to_mm.R \
  --input  /work/.../data/CTCF_bin1000_mtx.rds \
  --outdir /work/.../unified/work/ctcf/mm
python scripts/00_ingest.py --config configs/default.yaml
python scripts/01_train.py  --config configs/default.yaml     # GPU
python scripts/02_impute.py --config configs/default.yaml     # GPU
```

Imputed outputs: `work/ctcf/impute/` (`matrix_csr.npz`, `regions.tsv`,
`barcodes.tsv`, `meta.json`). Plug into downstream with
`--input unified=work/ctcf/impute` (also wired in
`downstream/slurm/_pipeline_paths.sh`).

## What this is not

- **Not a cascade.** Every submodule receives gradients from the same loss.
- **Not TF-conditioned.** No TF identity goes in. Generalization comes from
  per-channel multi-scale mixing, not architectural specialization.
- **Not PU.** The gate is differentiable + supervised. No Elkan-Noto,
  no positive seed, no scOpen NMF.
- **Not bulk-prior-dependent.** No motif BEDs, no cCRE annotations, no
  pretrained sequence weights (the encoder is trained from scratch on the
  user's matrix).
