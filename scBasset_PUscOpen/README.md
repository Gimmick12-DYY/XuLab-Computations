# scBasset → PUscOpen cascade

Two-stage imputation: **scBasset** broadens the matrix (recovers bins zero in
every cell using only sequence + learned cell embeddings), then **PUscOpen**
denoises that broadened matrix (geometric cCRE mask + PU classifier + scOpen
NMF + neighbourhood postfilter, all jointly favoring annotated CTCF sites).

Sister pipeline to `scBasset_cisTopic/`. They share the same first stage
(scBasset) and differ only in the denoiser.

## Why cascade

scBasset alone (HEK293T CTCF benchmark):

```
n_pos_bins_in_panel:    12,805     true CTCF peaks
n_neg_bins_in_panel:    12,805     random negatives
frac_pos_panel_covered: 0.798      sensitivity
frac_neg_panel_covered: 0.439      ~44 % of negatives also "hit"
```

PUscOpen is the most opinionated denoiser in our stack: every step prunes via
either the cCRE/CTCF annotation, a learned PU model, or local neighbourhood
support. Cascading PUscOpen after scBasset puts each method where it's
strongest:

```
raw matrix
   |
   |  scBasset (sequence-driven recall)
   v
matrix_csr.npz   denser, sensitivity ~0.80, specificity ~0.56
   |
   |  PUscOpen: cCRE/CTCF mask -> PU classifier -> scOpen NMF
   |            -> neighbourhood postfilter (multiply_by_pu)
   v
matrix_csr.npz   refined; expectation = positives near cCRE/CTCF sites
                 amplified, negatives off-annotation demoted hard
```

## Pipeline shape

```
01_ingest_scbasset.py        scBasset impute -> mm/{matrix.mtx.gz, regions.tsv.gz, barcodes.tsv.gz}
[reuse PUscOpen 02 .. 06]    standard PUscOpen pipeline pointed at this mm
```

Steps 02–06 are the **unmodified** PUscOpen scripts (`02_build_mask.R`,
`03_pu_classifier.py`, `04_run_scopen.py`, `05_postfilter.py`, `06_impute.py`).
The cascade config in `configs/default.yaml` contains all PUscOpen knobs
(`mask`, `pu`, `scopen`, `postfilter`, `impute`) plus a new top-level
`cascade:` block telling step 01 where to read scBasset's output.

Outputs land at `<work>/impute/{matrix_csr.npz, regions.tsv, barcodes.tsv,
meta.json}` — the same cross-pipeline schema.

## Setup

No new conda env — uses the existing `puscopen` env.

```bash
conda activate puscopen
```

## Run

```bash
# end-to-end (~12-24h depending on scopen rank sweep)
sbatch slurm/99_full_pipeline.sbatch

# or step-by-step (point each PUscOpen sbatch at our config)
sbatch slurm/01_ingest.sbatch
sbatch ../PUscOpen/slurm/02_build_mask.sbatch  CFG=configs/default.yaml
sbatch ../PUscOpen/slurm/03_pu.sbatch          CFG=configs/default.yaml
sbatch ../PUscOpen/slurm/04_scopen.sbatch      CFG=configs/default.yaml
sbatch ../PUscOpen/slurm/05_postfilter.sbatch  CFG=configs/default.yaml
sbatch ../PUscOpen/slurm/06_impute.sbatch      CFG=configs/default.yaml
```

## Hooking into the cross-pipeline comparison

```bash
python downstream/compare_pos_neg.py \
  --input scbasset=/path/to/scBasset/work/ctcf/impute \
  --input scbasset_cistopic=/path/to/scBasset_cisTopic/work/ctcf/impute \
  --input scbasset_puscopen=/path/to/scBasset_PUscOpen/work/ctcf/impute \
  --input puscopen=/path/to/PUscOpen/work/ctcf/impute \
  ...
```

## Ingest preserves count structure (pseudocount mode)

PUscOpen's PU classifier uses count-derived features (`log10_total_count`,
`log10_max_count`, `log10_n_cells_observed`, `fraction_cells_observed`).
If we naïvely binarised scBasset's output to fit PUscOpen's expected input,
those features would collapse — every nonzero bin would have `max_count = 1`
and `total_count == n_cells_observed`.

So step 01 (`01_ingest_scbasset.py`) defaults to `cascade.output_mode:
pseudocount` instead. It expects scBasset to have been run with
`impute.binarize_output: false`, so its CSR carries:

- **sigmoid probabilities in (0, 1)** at bins scBasset imputed
- **integer raw counts ≥ 1** at bins the raw matrix already covered
  (preserved by scBasset's `keep_raw: true` element-wise-max union)

The ingest rescales the (0, 1) range into integer pseudocounts in [1, K]
(default `K = 10`, `cascade.scale_factor`) and leaves the ≥1 raw counts
untouched. The resulting `mm/matrix.mtx.gz` has the **same dtype and value
range as the original raw CTCF matrix**, so all of PUscOpen's count
features remain informative and `pu.features` can include `log10_max_count`
as usual.

### Required scBasset settings for this cascade

In `scBasset/configs/default.yaml`:

```yaml
impute:
  binarize_output: false   # MUST be false for cascade pseudocount/float modes
  keep_raw:        true    # MUST be true so raw counts survive into the union
```

If you ran scBasset with `binarize_output: true` (its default), step 01 will
detect the all-binary input and fall back to `binary` mode with a warning.
Either re-run scBasset, or set `cascade.output_mode: binary` here and drop
`log10_max_count` from `pu.features` (it's identically 0 in binary mode).

## Why the cCRE/CTCF mask is OFF in this cascade

Standalone PUscOpen turns on `mask.use_cCRE_*` to restrict computation to
~50k annotated bins. That same prior is what the evaluation panel uses to
draw negatives (random non-cCRE bins), so leaving the mask on **trivially**
filters every panel negative out of the output:

| Run                         | n_bins_any_signal | frac_pos_panel | frac_neg_panel |
|-----------------------------|-------------------|----------------|----------------|
| scBasset alone              | 1,326,659         | 0.798          | 0.439          |
| scBasset → PUscOpen, mask ON | 38,713           | 0.798          | 0.002          |

`frac_neg_panel_covered = 0.002` is **definitionally low**, not earned by
denoising. The cascade's PU classifier + scOpen + postfilter have done very
little work — the cCRE mask alone produced almost the whole reduction. To
measure whether the matrix-level denoising is actually pulling weight, we
turn the cCRE flags off:

```yaml
mask:
  use_cCRE_293T:    false
  use_cCRE_293T_CA: false
  use_cCRE_CTCF:    false
  exclude_blacklist: true   # kept (blacklist is QC, not a prior on CTCF)
```

The PU classifier still uses annotation **features** internally
(`in_cCRE_*`, `dist_log_*`, etc.) to learn its scoring function, but the
*mask* is no longer the same set used to draw negatives. If after this
config change `frac_pos_panel_covered` stays high while
`frac_neg_panel_covered` stays meaningfully below scBasset's 0.439, the
denoising is real. If `frac_neg_panel_covered` jumps back near 0.4, the
mask was doing all the work.

### Heads up: memory

scOpen NMF now sees ~1M+ rows (everything nonzero in the cascade input)
instead of ~50k. Bump the slurm script's memory if `04_run_scopen.py` OOMs.
If still too heavy, two non-circular ways to reduce the universe:

- Set `mask.use_motif: true` + `mask.motif_bed: <path>` — uses the JASPAR
  CTCF motif BED (a sequence prior, not the panel's annotation prior).
- Pre-filter rows in `scripts/01_ingest_scbasset.py` against any candidate
  BED that isn't `cCRE * HEK293T`.

## Tuning differences from standalone PUscOpen

The cascade config tightens a few knobs because scBasset's output is already
broad — we want PUscOpen here to denoise, not amplify:

| Knob                          | Standalone | Cascade | Why                                                |
| ----------------------------- | ---------- | ------- | -------------------------------------------------- |
| `mask.use_cCRE_*`             | true       | **false** | Stops the circular evaluation (see above).      |
| `scopen.alpha`                | 1.25       | 1.50    | More conservative NMF; less amplification.         |
| `postfilter.min_neighbors`    | 1          | 2       | Demand more local support; cleans up scBasset's diffuse hits. |
| `postfilter.damp_floor`       | 0.72       | 0.50    | Push harder on unsupported bins.                   |
| `impute.propagate.enabled`    | true       | false   | scBasset already filled neighbours; no compounding. |

Bump or relax any of these depending on what `compare_pos_neg.py` shows for
the precision/recall trade.
