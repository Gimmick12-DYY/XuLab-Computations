# scBasset → cisTopic cascade

Two-stage imputation: **scBasset** broadens the matrix (recovers bins zero in
every cell using only sequence + learned cell embeddings), then **cisTopic**
denoises that broadened matrix (LDA topics over the now-denser binary matrix
concentrate signal on coherent peaks and demote sporadic false positives).

## Why cascade

From our HEK293T CTCF benchmark, scBasset alone scored:

```
n_pos_bins_in_panel:    12,805     true CTCF peaks in evaluation panel
n_neg_bins_in_panel:    12,805     random negatives in evaluation panel
frac_pos_panel_covered: 0.798      sensitivity (good!)
frac_neg_panel_covered: 0.439      specificity loss (false positives)
```

scBasset gains true positives by extrapolating from sequence, but inevitably
also fills negatives. cisTopic by itself does the opposite: tight LDA-thresholded
output is precise but only fills bins that already have signal. Running cisTopic
*on top of* scBasset's output uses each method where it's strongest:

```
raw matrix
   |
   |  scBasset (sequence-driven recall)
   v
matrix_csr.npz   denser, ~1.7 M nnz, sensitivity ~0.80, specificity ~0.56
   |
   |  cisTopic LDA + sparsity_match + propagation (topic-driven denoise)
   v
matrix_csr.npz   refined; expectation = keep most true positives, drop
                 spurious negatives that don't co-occur in any topic
```

## Pipeline shape

```
01_ingest_scbasset.py        scBasset impute -> mm/{matrix.mtx.gz, regions.tsv.gz, barcodes.tsv.gz}
[reuse cisTopic 02 .. 05]    standard cisTopic pipeline pointed at this mm
```

Steps 02–05 are the **unmodified** cisTopic scripts (`02_build_cistopic_obj.py`,
`03_run_lda_mallet.py`, `04_select_model.py`, `05_impute.py`). The cascade
config in `configs/default.yaml` contains all cisTopic knobs (`filter`, `lda`,
`select`, `impute`) plus a new top-level `cascade:` block telling step 01
where to read scBasset's output.

Outputs land at `<work>/impute/{matrix_csr.npz, regions.tsv, barcodes.tsv,
meta.json}` — the same cross-pipeline schema.

## Setup

No new conda env — uses the existing `cistopic` env (MALLET, pycisTopic, etc.).

```bash
conda activate cistopic
```

## Run

```bash
# end-to-end (~12-24h depending on LDA topic sweep)
sbatch slurm/99_full_pipeline.sbatch

# or step-by-step
sbatch slurm/01_ingest.sbatch                       # ~5 min, CPU only
sbatch ../cisTopic/slurm/02_build.sbatch  CFG=configs/default.yaml
sbatch ../cisTopic/slurm/03_lda.sbatch    CFG=configs/default.yaml
sbatch ../cisTopic/slurm/04_select.sbatch CFG=configs/default.yaml
sbatch ../cisTopic/slurm/05_impute.sbatch CFG=configs/default.yaml
```

The cisTopic SLURM scripts all accept `CFG=<path>` so we point them at our
cascade config; no copying needed.

## Hooking into the cross-pipeline comparison

```bash
python downstream/compare_pos_neg.py \
  --input scbasset=/path/to/scBasset/work/ctcf/impute \
  --input scbasset_cistopic=/path/to/scBasset_cisTopic/work/ctcf/impute \
  --input scbasset_puscopen=/path/to/scBasset_PUscOpen/work/ctcf/impute \
  --input cistopic=/path/to/cisTopic/scripts/cistopic_ctcf/impute \
  ...
```

The expected pattern in `frac_neg_panel_covered`: scbasset >>
scbasset_cistopic ≈ scbasset_puscopen > cistopic.

## Tuning knobs

The cascade config inherits cisTopic's full config (see
`../cisTopic/configs/default.yaml` for documentation). The new knob:

```yaml
cascade:
  source_impute_dir: /path/to/scBasset/work/ctcf/impute
  # If true, binarise the scBasset matrix before writing mm (default true).
  # scBasset's default is already binary, but if you ran it with
  # impute.binarize_output: false the values are sigmoid probabilities and we
  # need to threshold them before LDA.
  binarize: true
```

Recommended cisTopic-side tweaks for this cascade:

- `lda.n_topics`: scBasset output is already broad, so smaller K (e.g.
  `[5, 8, 12, 15, 20]`) is usually enough — the model doesn't need to
  reconstruct dropouts, only structure what's there.
- `impute.threshold_mode: sparsity_match:1` (was `:3` for raw input) —
  the goal here is denoising not amplification, so we want output nnz
  roughly equal to scBasset's, not 3x.
- `impute.propagate.enabled: false` — propagation in cisTopic was designed
  to fill bins LDA couldn't reach. scBasset already filled them; propagating
  again risks compounding false positives. Set true only if you want to
  reach bins scBasset *also* didn't fill.
