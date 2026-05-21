# scBasset + cisTopic (LDA-denoise + union)

Use cisTopic's LDA topics — trained on the **raw** matrix, so they reflect
biology — to score every (bin, cell) in scBasset's output, keep only the
ones LDA validates, then union with cisTopic's standalone output for the
"additive" property.

## What this pipeline does

```
RAW matrix
  ├── cisTopic standalone ──> matrix_csr.npz + factors.npz (phi, theta)
  │                           (LDA on RAW, topics reflect biology)
  │
  └── scBasset standalone ──> matrix_csr.npz
                              (sequence-driven, broad recall)
                                            │
                                            ▼
                           ┌───────────────────────────────┐
                           │ 01_denoise.py                 │
                           │ for each (r, c) in scBasset:  │
                           │   if r in cisTopic LDA vocab: │
                           │     score = phi[r,:] @ theta[:,c]
                           │     keep iff score >= T_c     │
                           │   else (truly-new bin):       │
                           │     keep (LDA can't validate) │
                           │ then union with cisTopic CSR  │
                           └───────────────┬───────────────┘
                                            ▼
                                   <work>/impute/matrix_csr.npz
```

Where `T_c` is the per-cell quantile threshold (default: 50th percentile of
LDA scores over modeled bins for cell c) — keep scBasset entries that LDA
also ranks in the top 50% for that cell.

The union with cisTopic's standalone output guarantees every entry cisTopic
itself called survives (additive property), while scBasset's contributions
are gated by LDA agreement (denoising property).

## Why this is the right architecture

Earlier versions failed:

- **Cascade scBasset → cisTopic LDA**: LDA trained on the *scBasset-densified*
  matrix learned topics about scBasset's co-occurrence patterns, not real
  biology. `frac_pos_panel_covered` collapsed from 0.798 → 0.186.
- **Pure union (max only)**: additive but did no denoising — kept all
  scBasset false positives.

The fix: **train cisTopic on raw** (it's already a standalone pipeline), use
its topics to *score* scBasset's matrix without re-training, then union with
cisTopic's own standalone output. cisTopic's role here is two-fold:
1. Provide the LDA model (phi, theta) used to validate scBasset entries.
2. Provide its own thresholded matrix for the additive union.

## Pipeline shape

```
sources.scbasset_impute_dir  ──┐
                                ├──> 01_denoise.py ──> <work>/impute/
sources.cistopic_impute_dir  ──┘
   (must contain factors.npz +
    region_topic_phi.parquet +
    cell_topic_theta.parquet)
```

`01_denoise.py` is the only script. Outputs land at
`<work>/impute/{matrix_csr.npz, regions.tsv, barcodes.tsv, meta.json}` —
same cross-pipeline schema.

## Setup + run

```bash
# 1) Both source pipelines must have finished:
#    /path/to/scBasset/work/ctcf/impute/matrix_csr.npz
#    /path/to/cisTopic/scripts/cistopic_ctcf/impute/{matrix_csr.npz,factors.npz,
#                                                   region_topic_phi.parquet,
#                                                   cell_topic_theta.parquet}
#
#    cisTopic must have been run with impute.drop_factors_npz: false (the
#    default) so factors.npz exists.
#
# 2) Edit configs/default.yaml so sources.* point at those two dirs.

conda activate cistopic
sbatch slurm/99_full_pipeline.sbatch    # ~1-5 min on a 3M x 10K matrix
```

## Tuning knobs

```yaml
denoise:
  # off                 -> no filtering (== pure max-union with cistopic)
  # quantile:Q          -> per-cell threshold, keep score >= Q-th percentile
  # global_quantile:Q   -> single threshold from a global Q-th percentile
  # fixed:T             -> numeric threshold on phi[r,:] @ theta[:,c]
  mode: "quantile:0.5"      # default: drop the bottom 50% per cell

  threshold_sample_size: 5000     # bins sampled to estimate per-cell quantiles
  keep_truly_new_bins:   true     # bins not in LDA's vocabulary are kept as-is

combine:
  union_with_cistopic: true       # additive property
  binarize_output:     true       # store 1 at every nonzero
```

**Direction of travel:**
- Higher `quantile:Q` (e.g. 0.75, 0.90) = stricter denoising, fewer scBasset
  entries kept on modeled bins.
- `mode: off` = pure union (no filtering); equivalent to the old "max-only"
  pipeline.
- `keep_truly_new_bins: false` = drop scBasset's truly-new bins (the ones
  LDA can't validate). Defeats scBasset's main contribution; rarely useful.

## What meta.json tells you

```json
{
  "n_modeled_bins":          119083,      // size of cisTopic's LDA vocabulary
  "n_truly_new_bins_in_mm":  2911970,     // bins scBasset can hit but LDA can't
  "scbasset_nnz_total":      1722004,
  "scbasset_nnz_on_modeled": 642338,      // entries that get LDA-validated
  "kept_modeled_nnz":        321156,      // ~half kept at quantile:0.5
  "kept_truly_new_nnz":      1079666,     // truly-new entries always kept
  "filtered_scbasset_nnz":   1400822,     // = kept_modeled + kept_truly_new
  "cistopic_nnz_total":      1748250,     // added by the union step
  "output_nnz":              2900000      // approx, union of the two
}
```

## Expected effect on the CTCF benchmark

Compared to scBasset alone (frac_pos=0.798, frac_neg=0.439):

- `frac_pos_panel_covered`: should stay close to or above 0.798. cisTopic's
  contribution can add positives; LDA filtering can drop positives whose
  `phi[r,:] @ theta[:,c]` is in the bottom Q.
- `frac_neg_panel_covered`: should drop materially below 0.439 IF LDA
  topics genuinely discriminate true vs spurious bins for the cells in
  question. If neg coverage stays near 0.4, the topics aren't separating
  true from spurious — try a stricter `quantile:Q` or revisit the cisTopic
  topic count.
