# scBasset+cisTopic ⊕ scBasset+PUscOpen (cis-anchored, pus enhances)

```
out[r, c] = cis[r, c] + w · pus_normalized[r, c]    if cis[r, c] > 0
          = 0                                        otherwise
```

**cisTopic side is the backbone** — every `(bin, cell)` entry in
`scbasset_cistopic` is preserved exactly in the output.
**PUscOpen side is the enhancer** — its values amplify cis entries where
they overlap; **pus-only entries are dropped** (entries where cis is 0
do not enter the output, no matter how confident pus was).

This was renamed from "union" to "enhancer" after the legacy max union
let PUscOpen's huge scOpen-scaled values wash out the cis backbone in
IGV (uniform picket-fence peaks). The enhancer mode guarantees the cis
call set is preserved and pus only differentiates strong vs weak entries
within that call set.

The legacy symmetric `max` mode is still available — set
`combine.method: max` — but is no longer the default.

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
  # enhancer (default) -- cis-anchored. Every cis entry survives; pus boosts
  #                       cis values at overlap; pus-only entries dropped.
  # max                -- legacy symmetric union. Element-wise max of both.
  method: enhancer

  # Amplitude of pus boost at overlap entries (only used in enhancer mode).
  # 1.0 -> overlap peak ~= 1 + pus_normalized (~2x cis-only height).
  # Bump to 2-5 for stronger contrast; 0 reduces to raw scbasset_cistopic.
  enhancer_weight: 1.0

  # Rescale each input to mean(nnz)=1 before combining. Without this,
  # scbasset_cistopic (binary) is dwarfed by scbasset_puscopen (scOpen
  # floats, can reach ~1e6). Keep true unless you have a specific reason.
  normalize_inputs: true

  # Keep false. Setting true erases per-(bin, cell) magnitudes -- exactly
  # the information the enhancer is trying to express.
  binarize_output: false
```

## Expectations vs each source

In `enhancer` (default) mode:

- `frac_pos_panel_covered`: equals `scbasset_cistopic`'s value exactly
  (same (bin, cell) entries, only amplitudes differ).
- `frac_neg_panel_covered`: equals `scbasset_cistopic`'s value exactly.
- `total_stored_nnz`: equals `scbasset_cistopic_nnz` exactly.
- `n_cis_entries_boosted_by_pus`: how many cis entries were amplified by
  pus at the same (bin, cell). The remaining cis entries pass through
  with their original cis value.

What changes vs raw `scbasset_cistopic`: **the per-(bin, cell) magnitudes**
are now larger at agreement entries. In IGV (BigWig per-bin mean), bins
where many cells have cis+pus agreement are visibly taller than bins
where cis fires alone.

In `max` mode the old additive coverage semantics apply (output ⊇ both
inputs, pos/neg covered ≥ max(both)).
