# scBasset+cisTopic ∪ scBasset+PUscOpen (additive union)

```
out[r, c] = max(scbasset_cistopic[r, c], scbasset_puscopen[r, c])
```

Purely additive: every (bin, cell) called by either upstream pipeline
survives. No filtering. The two upstream pipelines already each contain
scBasset's truly-new bins; this combines their LDA-denoised and
NMF-denoised contributions.

## Pipeline shape

```
sources.scbasset_cistopic_impute_dir ──┐
                                        ├──> 01_union.py ──> <work>/impute/
sources.scbasset_puscopen_impute_dir ──┘
```

Single step. Outputs land at `<work>/impute/{matrix_csr.npz, regions.tsv,
barcodes.tsv, meta.json}` — same cross-pipeline schema.

## Setup + run

```bash
# Prerequisites: both upstream pipelines must have finished and written
# their <work>/impute/.

conda activate cistopic
sbatch slurm/99_full_pipeline.sbatch
```

~1-5 min on a 3M x 10K matrix, CPU only.

## Hooking into the cross-pipeline comparison

```bash
python downstream/compare_pos_neg.py \
  --input scbasset=/path/to/scBasset/work/ctcf/impute \
  --input scbasset_cistopic=/path/to/scBasset_cisTopic/work/ctcf/impute \
  --input scbasset_puscopen=/path/to/scBasset_PUscOpen/work/ctcf/impute \
  --input scbasset_cistopic_puscopen=/path/to/scBasset_cisTopic_PUscOpen/work/ctcf/impute \
  ...
```

## Tuning knobs

```yaml
sources:
  scbasset_cistopic_impute_dir: <path>
  scbasset_puscopen_impute_dir: <path>

combine:
  method: max                # only `max` supported
  normalize_inputs: true     # rescale each input to mean(nnz)=1 before max,
                             # so binary cisTopic and scOpen-scaled PUscOpen
                             # contribute proportionally instead of PUscOpen
                             # dominating every overlap
  binarize_output: false     # keep peak-strength magnitudes; set true only
                             # if you need a strict 0/1 union for downstream
                             # set logic
```

If you want the legacy "binary union" behaviour back, set both
`normalize_inputs: false` and `binarize_output: true`. But that's exactly
what produced the flat picket-fence IGV view, so it's almost never useful.

## Expectations vs each source

- `frac_pos_panel_covered`: ≥ max(both sources). Adds positive bins one
  caught but the other didn't.
- `frac_neg_panel_covered`: ≥ max(both sources). Same logic for negatives.
- `total_stored_nnz`: roughly `scbasset_cistopic_nnz + scbasset_puscopen_nnz`
  minus overlap.

The union strictly cannot reduce false positives — both sources' false
positives are kept. Use this when sensitivity is the priority and a
downstream filter (e.g. motif overlap, neighborhood consistency) will
handle precision separately.
