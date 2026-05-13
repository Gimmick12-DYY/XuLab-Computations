# Downstream cross-pipeline analyses

Scripts that consume compact imputation outputs written by each pipeline
(`factors.npz` or `matrix.npy` + `regions.tsv` + `barcodes.tsv`) plus the raw
`mm/` matrix, and produce comparisons.

## Bin-level positive vs negative CTCF signal

`Bin_posVSneg.ipynb` is the source-of-truth notebook for the analysis. The two
scripts below mirror it for batch / SLURM execution.

### Step 1 — build the bin index (once per `mm/` universe)

`prepare_pos_neg_bins.R` downloads SCREEN cCRE annotations + the ENCODE
hg38 v2 blacklist, then computes:

* **positives**: bins overlapping the top-N CTCF cCREs (by 293 CTCF z-score)
  that are also chromatin-accessible in 293T (`CA` class)
* **negatives**: bins NOT overlapping any GRCh38 CTCF-bound cCRE, sampled to
  match the positive count (seed defaults to 2026, matching the notebook)

Blacklist bins are excluded from both pools. Indices are emitted in the
**original mm row space** (`bin_idx_orig`) and each row also includes
`bin_name` (`chr:start-end`) so imputed matrices can be aligned by region name.

```bash
Rscript downstream/prepare_pos_neg_bins.R \
  --mm-dir   /work/.../cistopic_ctcf/mm \
  --out-dir  /work/.../downstream/bins \
  --cache-dir /work/.../downstream/cache \
  --n-top    10000 \
  --seed     2026
```

Outputs:

| file | contents |
|---|---|
| `pos_neg_bins.tsv` | `bin_idx_orig` (0-based mm row), `bin_idx_post_blacklist`, `label` (`pos`/`neg`), `bin_name` |
| `pos_neg_bins.meta.json` | source URLs, blacklist count, n_pos/n_neg, seed |

`mm/regions.tsv.gz` only needs to be done once per data set — the same bin
index file works for any imputed output derived from that input.

SLURM:

```bash
sbatch --export=ALL,\
MM_DIR=/work/.../cistopic_ctcf/mm,\
OUT_DIR=/work/.../downstream/bins,\
CACHE_DIR=/work/.../downstream/cache \
  downstream/slurm/prepare_pos_neg_bins.sbatch
```

### Step 2 — compare across pipelines

`compare_pos_neg.py` accepts any number of `--input LABEL=PATH` pairs where
`PATH` is one of:

* a directory containing `matrix.mtx.gz` (raw counts),
* a directory containing `factors.npz` + `regions.tsv` (factored imputed output),
* a directory containing `matrix.npy` + `regions.tsv` (dense imputed output),
* or a legacy HDF5 file with a `Prc` dataset.

It computes the per-bin total signal at the positive and negative bins,
reports AUROC / KS / median ratio, and writes multi-panel ECDF + AUROC plots.

```bash
python downstream/compare_pos_neg.py \
  --bins-tsv /work/.../downstream/bins/pos_neg_bins.tsv \
  --out-dir  /work/.../downstream/pos_neg \
  --input raw=/work/.../cistopic_ctcf/mm \
  --input cisTopic=/work/.../cistopic_ctcf/impute \
  --input FITS=/work/.../FITS/work/ctcf/impute \
  --input scOpen=/work/.../scOpen/work/ctcf/impute \
  --input MAGIC=/work/.../MAGIC/work/ctcf/impute
```

Outputs (under `--out-dir`):

| file | contents |
|---|---|
| `compare_pos_neg.png` | one ECDF panel per `--input`, pos vs neg curves overlaid |
| `compare_pos_neg_auroc.png` | AUROC bar chart, one bar per input |
| `compare_pos_neg.json` | per-input AUROC, KS statistic + p-value, pos/neg medians, log2(median ratio), zero fractions |
| `compare_pos_neg.tsv` | long-form per-bin signal table (`input`, `group`, `bin_idx_orig`, `bin_name`, `modeled_in_input`, `in_modeled_union`, `signal`) for replotting |

SLURM:

```bash
sbatch --export=ALL,\
BINS_TSV=/work/.../downstream/bins/pos_neg_bins.tsv,\
OUT_DIR=/work/.../downstream/pos_neg,\
INPUTS="raw=/work/.../cistopic_ctcf/mm cisTopic=/work/.../cistopic_ctcf/impute FITS=/work/.../FITS/work/ctcf/impute scOpen=/work/.../scOpen/work/ctcf/impute MAGIC=/work/.../MAGIC/work/ctcf/impute" \
  downstream/slurm/compare_pos_neg.sbatch
```

### Reading the output

* **AUROC** — how well per-bin total signal separates positive from negative
  bins. 0.5 = no signal; 1.0 = perfect separation. Higher is better.
* **KS statistic** — distance between the two ECDF curves. Higher = larger
  distributional separation; the accompanying p-value tests against the null
  that pos and neg are drawn from the same distribution.
* **log2 median ratio** — log2(median_pos / median_neg). Positive = imputer
  *up-weights* CTCF cCRE bins relative to background.
* **Zero fractions** — for imputed matrices these should be near zero outside
  the modeled rows (FITS subset / cisTopic filtered universe).

If FITS only modeled a 10K-bin subset, most positive bins fall outside the
FITS subset and will show a zero signal. AUROC will collapse toward 0.5; this
is **expected** and is the cost of subsetting. The cleanest A/B compare in
that case is to also rerun the analysis with `--bins-tsv` restricted to
positives that fall inside the FITS subset (filter the TSV before passing it
in).

## Sensitivity / specificity vs raw observed matrix

`sensitivity_specificity.py` treats `raw[r, c] > 0` (binarised) as the ground
truth label and `imputed[r, c]` as a continuous score, sweeps thresholds via
a **streaming histogram pass** through the modeled rows, and reports per
pipeline:

* `TP / TN / FP / FN` at every threshold
* `sensitivity` (= TPR = recall), `specificity` (= TNR), `precision`, `F1`,
  `MCC`, `accuracy`
* **AUROC** and **AUPR** (trapezoidal integration over the curves)
* **max-F1 threshold** and the sens / spec achieved there
* **sparsity-matched threshold** — the threshold where `#(imputed > t)` is
  closest to the raw nonzero count, i.e. the calibration point that produces
  the same number of "calls" as the raw observed matrix

Two reporting views are emitted for every pipeline:

| view | description |
|---|---|
| `modeled` | only rows present in the imputer's `regions.tsv` (per-pipeline universe; fair when comparing imputers that model different region sets) |
| `full`    | all raw rows; unmodeled raw nonzeros are counted as FN at every threshold (penalises pipelines for *not* modeling a row at all — e.g. FITS at 10 K subset) |

The script never materialises the full dense imputed matrix; it streams in
row blocks (default 2000 rows) and accumulates a 2-row histogram (`pos` /
`neg`) per pass. Memory peaks at ~`chunk_rows × n_cells × 4` bytes per
pipeline (≈ 80 MB for 2000 × 10410 float32).

```bash
python downstream/sensitivity_specificity.py \
  --mm-dir   /work/.../cistopic_ctcf/mm \
  --out-dir  /work/.../downstream/sensitivity_specificity \
  --input cisTopic=/work/.../cistopic_ctcf/impute \
  --input FITS=/work/.../FITS/work/ctcf/impute \
  --input scOpen=/work/.../scOpen/work/ctcf/impute \
  --input MAGIC=/work/.../MAGIC/work/ctcf/impute
```

Outputs (under `--out-dir`):

| file | contents |
|---|---|
| `sensitivity_specificity.tsv` | long-form: input, view, threshold, TP/FP/FN/TN, sens/spec/prec/F1/MCC/acc |
| `sensitivity_specificity.json` | per-input summary: AUROC, AUPR, max-F1 + threshold, sparsity-matched threshold + sens/spec |
| `sensitivity_specificity_roc_{modeled,full}.png` | ROC curves overlaid across pipelines, one figure per view |
| `sensitivity_specificity_pr_{modeled,full}.png`  | precision-recall curves overlaid across pipelines |
| `sensitivity_specificity_sens_spec_{modeled,full}.png` | sens / spec / F1 vs threshold, one panel per pipeline |
| `sensitivity_specificity_bars_{modeled,full}.png` | AUROC / AUPR / max-F1 / sens@match / spec@match bar chart |

By default the script picks **per-pipeline quantile-based thresholds** (48
points spanning quantiles 0.001 → 0.999 with right-tail emphasis, plus 0.0
as a top-right ROC anchor). This is the right choice for ROC / AUPR — score
scales differ across pipelines and AUROC is invariant to monotonic
rescaling. Pass `--thresholds 0.01,0.1,0.5,...` to force a fixed grid across
all pipelines.

SLURM:

```bash
sbatch --export=ALL,\
MM_DIR=/work/.../cistopic_ctcf/mm,\
OUT_DIR=/work/.../downstream/sensitivity_specificity,\
INPUTS="cisTopic=/work/.../cistopic_ctcf/impute FITS=/work/.../FITS/work/ctcf/impute scOpen=/work/.../scOpen/work/ctcf/impute MAGIC=/work/.../MAGIC/work/ctcf/impute" \
  downstream/slurm/sensitivity_specificity.sbatch
```

### Reading the output

* **Class imbalance is extreme** (~99.998 % zeros for CTCF at 1 kb).
  Specificity is therefore high almost by default; sensitivity is the
  discriminating axis.
* **AUROC near 0.5** at the modeled view means the imputer is no better than
  random at separating raw positives from raw negatives within rows it
  modeled. **AUROC near 1.0** means imputed values rank raw nonzeros above
  raw zeros almost perfectly.
* **AUPR is the harder bar** under heavy class imbalance — the random
  baseline is `pos_total / total` ≈ 2 × 10⁻⁵, not 0.5.
* **`sparsity_match_threshold`** is the calibration point most users care
  about — *"what sens / spec does the imputer hit when it makes the same
  number of positive calls as the raw matrix did?"* If `sens@match` is
  meaningfully higher than the raw density, the imputer is recovering signal
  beyond the observed positives, which is the imputation goal.
* **Caveat on "FP"**: a raw zero that becomes a high imputed value can be
  either a wrong call OR a recovered dropout. The ROC / PR metrics treat
  every raw zero as a negative; for biologically-grounded validation
  combine with the `compare_pos_neg.py` bin-level CTCF cCRE analysis.

## Bin-level sensitivity / specificity (cCRE-derived ground truth)

`bin_sensitivity_specificity.py` is the bin-level analogue of
`sensitivity_specificity.py`. The two answer different questions:

| script | ground truth | what it asks |
|---|---|---|
| `sensitivity_specificity.py` | `raw[r, c] > 0`; every (region, cell) entry | "Does the imputer recover the entries the raw matrix saw?" |
| `bin_sensitivity_specificity.py` | `pos_neg_bins.tsv` (cCRE-derived); each **bin** is one example | "Does the imputer separate true CTCF-regulatory bins from background?" |

The bin-level version uses the same positive / negative bin set the notebook
produces (top-10 K HEK293 CTCF cCREs ∩ HEK293T-CA, plus a size-matched random
sample of bins that overlap zero CTCF-bound cCREs in SCREEN). Each bin is
one classification example; the per-bin score is the sum-of-signal across
cells (same as `compare_pos_neg.py` uses). At a given threshold:

```
sensitivity = #(positive bins with signal > t) / n_pos
specificity = #(negative bins with signal <= t) / n_neg
```

```bash
python downstream/bin_sensitivity_specificity.py \
  --bins-tsv /work/.../downstream/bins/pos_neg_bins.tsv \
  --out-dir  /work/.../downstream/bin_sensitivity_specificity \
  --input raw=/work/.../mm \
  --input cisTopic=/work/.../cistopic_ctcf/impute \
  --input FITS=/work/.../FITS/work/ctcf/impute \
  --input scOpen=/work/.../scOpen/work/ctcf/impute \
  --input MAGIC=/work/.../MAGIC/work/ctcf/impute
```

Per pipeline (and per view: `full` + `modeled`) the script reports

* per-threshold **TP / FP / FN / TN**, sens / spec / precision / F1 / MCC /
  accuracy / Youden's J
* **AUROC** and **AUPR** via `sklearn.metrics`
* two natural operating points:
  * **max-F1** threshold and its sens / spec
  * **Youden J** threshold (max `sens + spec − 1`)

Outputs (under `--out-dir`):

| file | contents |
|---|---|
| `bin_sensitivity_specificity.tsv` | long-form per-threshold table |
| `bin_sensitivity_specificity.json` | per-input AUROC, AUPR, max-F1 + Youden-J operating points |
| `bin_sensitivity_specificity_roc_{full,modeled}.png` | ROC curves overlaid across pipelines |
| `bin_sensitivity_specificity_pr_{full,modeled}.png` | precision-recall curves overlaid |
| `bin_sensitivity_specificity_sens_spec_{full,modeled}.png` | sens / spec / F1 vs threshold (one panel per pipeline, with max-F1 and Youden-J markers) |
| `bin_sensitivity_specificity_bars_{full,modeled}.png` | AUROC / AUPR / max-F1 / sens@max-F1 / spec@max-F1 bar chart |

Run cost: only a few thousand bins per evaluation set, so the whole
multi-pipeline run takes seconds once the per-bin signals are computed; the
slow part is reading the imputed matrices for the lookup, which is the same
cost as `compare_pos_neg.py`.

SLURM:

```bash
sbatch downstream/slurm/bin_sensitivity_specificity.sbatch
```

(uses the same default paths as `compare_pos_neg_all4.sbatch`; override
`MM_DIR`, `CISTOPIC_DIR`, `FITS_DIR`, `SCOPEN_DIR`, `MAGIC_DIR`, or `INPUTS`
to point elsewhere)

### Reading the output

* **AUROC** here is *bin-level* — how well the per-bin imputed total ranks
  CTCF cCRE bins above background bins. It is **not** comparable in absolute
  terms to the AUROC reported by `sensitivity_specificity.py`, which is
  entry-level over (region, cell) pairs.
* **`full` vs `modeled` view**: in `full`, bins that fall outside the
  imputer's modeled universe contribute signal=0 (so the imputer is
  penalised for not modeling those bins at all). In `modeled`, those bins
  are excluded; this is the fair view for imputers that work on a restricted
  region set (e.g. FITS at 10 K, or scOpen with a candidate-region BED).
  For the `raw=mm` input there is no modeling so only the `full` view exists.
* **max-F1 vs Youden J**: F1 weights positives more heavily (good when
  imbalance matters); Youden J is the threshold maximising the diagnostic
  trade-off and is the natural choice when pos / neg counts are balanced
  (which they are here by construction).

## Per-cell UMI / signal-sum distribution

`umi_distribution.py` computes per-cell totals (column sums) for the raw
matrix and every imputed pipeline, aligns them to a common barcode order,
and emits the distribution as a long-form TSV, a wide TSV (for replotting),
a summary JSON, and three figures (overlaid log-x histogram, ECDF,
boxplot).

Per-cell sums are computed without materialising the dense matrix:

| input format | per-cell sum recipe |
|---|---|
| `mm/matrix.mtx.gz` | scipy `mat.sum(axis=0)` on the loaded sparse CSR |
| `factors.npz` (cisTopic / scOpen) | `(W.sum(axis=0) @ H) * per_cell_factor` |
| `matrix.npy` (FITS / MAGIC) | chunked `arr[s:e].sum(axis=0)` over an mmap |
| `matrix_csr.npz` (PUscOpen) | scipy `mat.sum(axis=0)` |
| legacy HDF5 `/Prc` | chunked `ds[s:e, :].sum(axis=0)` |

`--align-to LABEL` picks the barcode-order anchor (default: the first
`--input`, which the SLURM wrapper sets to `raw`). Cells absent from a given
input are reported as `NA` in the wide TSV and excluded from that input's
distribution.

```bash
python downstream/umi_distribution.py \
  --out-dir /work/.../downstream/umi_distribution \
  --input raw=/work/.../mm \
  --input cisTopic=/work/.../cistopic_ctcf/impute \
  --input FITS=/work/.../FITS/work/ctcf/impute \
  --input scOpen=/work/.../scOpen/work/ctcf/impute \
  --input MAGIC=/work/.../MAGIC/work/ctcf/impute \
  --input PUscOpen=/work/.../PUscOpen/work/ctcf/impute
```

Outputs (under `--out-dir`):

| file | contents |
|---|---|
| `umi_distribution.tsv` | long-form: input, kind, barcode, umi_total |
| `umi_distribution_wide.tsv` | barcode × pipeline matrix of per-cell totals (NA for missing barcodes) |
| `umi_distribution_stats.json` | per-input mean / median / p10 / p90 / p99 / std + missing-barcode counts |
| `umi_distribution_hist.png` | overlaid log-x histogram |
| `umi_distribution_ecdf.png` | overlaid ECDF |
| `umi_distribution_box.png` | boxplot across pipelines |

Reading note: imputed scales are not directly comparable to raw counts —
each pipeline's `05_impute.py` rescales per-cell sums to a target
`scale_factor` (defaults to `1e6`, CPM-like), so the y-axes track sums of
*scaled* signal rather than UMIs. The pipelines should agree closely after
rescaling; large gaps between them point to either over-imputation
(rescaled sums far below the others) or aggressive masking (PUscOpen's per-
cell sum is over the refined-mask rows only).

SLURM:

```bash
sbatch downstream/slurm/umi_distribution.sbatch
```

## Raw vs imputed coordinate diff (lost positives)

`raw_vs_imputed_diff.py` answers: **how many of the raw `1`s did each
imputer downgrade?** At every raw-nonzero `(region, cell)` coordinate it
looks up the imputed value from each input and reports the distribution of
`delta = imputed - raw` plus per-threshold "lost positive" counts. A raw=1
position with imputed value below threshold `t` counts as lost.

The lookup is name-aligned (regions by `chr:start-end`, cells by barcode),
so any pipeline that drops rows (FITS top-N, scOpen drop-zero, PUscOpen
refined mask) is fairly penalised: raw positives outside the imputed input's
modeled space are counted as `imputed = 0` (and therefore lost at every
positive threshold).

Per-coordinate value lookup is vectorised:

| input format | how the lookup works |
|---|---|
| `factors.npz` | `np.einsum('ik,ki->i', W[rows], H[:, cols]) * per_cell_factor` (chunked) |
| `matrix.npy` | `arr[rows, cols]` (chunked, mmap-backed) |
| `matrix_csr.npz` | `mat[rows, cols]` (paired CSR advanced indexing, chunked) |
| HDF5 `/Prc` | row-batched: load unique rows once, then index columns |

For context the script also samples `--n-zero-sample` (default 200 K) raw
zero coordinates and reports the imputed-value distribution there, which
catches "imputed-only gains" that didn't exist in the raw matrix.

```bash
python downstream/raw_vs_imputed_diff.py \
  --mm-dir   /work/.../mm \
  --out-dir  /work/.../downstream/raw_vs_imputed_diff \
  --thresholds 0,0.001,0.01,0.05,0.1,0.5,1.0 \
  --input cisTopic=/work/.../cistopic_ctcf/impute \
  --input FITS=/work/.../FITS/work/ctcf/impute \
  --input scOpen=/work/.../scOpen/work/ctcf/impute \
  --input MAGIC=/work/.../MAGIC/work/ctcf/impute \
  --input PUscOpen=/work/.../PUscOpen/work/ctcf/impute
```

Outputs (under `--out-dir`):

| file | contents |
|---|---|
| `raw_vs_imputed_diff.tsv` | per-pipeline per-threshold table: TP/FN, lost / recovered counts and fractions |
| `raw_vs_imputed_diff_per_cell.tsv` | per-cell lost-positive counts at the smallest threshold (only cells with loss > 0) |
| `raw_vs_imputed_diff.json` | summary: imp@raw=1 stats, delta stats, imp@raw=0 stats, per-threshold rows, per-cell + per-bin loss stats |
| `..._imp_at_raw_pos.png` | overlaid `log1p(imputed)` histograms at raw=1 coords |
| `..._delta.png` | overlaid `imputed - raw` histograms (negative = imputer underestimated) |
| `..._per_cell_lost.png` | boxplot of per-cell lost counts at the default threshold |
| `..._imp_at_raw_zero.png` | imputed-value histogram at sampled raw=0 coords (skipped if `--n-zero-sample 0`) |

Reading the metrics:

* `frac_eq_zero` (in `imp_at_raw_pos_stats`) is the fraction of raw-positive
  coordinates where the imputer literally returned 0. For pipelines that
  restrict the modeled region set, this floor is exactly the fraction of
  raw positives that fell *outside* the modeled rows. PUscOpen's refined
  mask trades a higher `frac_eq_zero` against a tighter false-positive
  surface; a high `frac_eq_zero` is not by itself a defect.
* `delta_stats.frac_negative` is the fraction of raw-positive coords where
  `imputed < raw`. With binarised raw and imputed values in `[0, 1]` (or a
  CPM-like rescale that puts most values much greater than 1) this is
  almost always close to 1 — what matters is the *distribution* of the
  delta, not its sign. Read the `delta` histogram for the spread.
* `n_lost / n_raw_positive` at small thresholds (`t = 0` or `t = 0.001`) is
  the cleanest "did the imputer keep raw-observed signal alive" number;
  larger thresholds shift the question toward "did the imputer assign
  notable signal to the raw-observed coords".

SLURM:

```bash
sbatch downstream/slurm/raw_vs_imputed_diff.sbatch
```

## Per-cell complexity gain (fragments-gained-per-cell)

`complexity_gain.py` measures the imputation goal directly: **how many more
accessible bins per cell do we recover by imputing?** For every cell:

```
raw_complexity[c]  =  #(raw[r, c] > 0)
imp_complexity[c]  =  #(imp[r, c] > t)
gain[c]            =  imp_complexity[c]  -  raw_complexity[c]
```

The threshold `t` is picked per pipeline because imputed scales differ across
methods. Four modes are supported via `--threshold-mode`:

| mode | meaning |
|---|---|
| `sparsity_match` (default) | Pick `t` per pipeline so `#(imp > t)` on the modeled rows equals `#(raw > 0)` on those same rows. Sum-of-gain across cells is then **≈0** when the threshold is exact; finite random sampling can shift it slightly. **PUscOpen `csr_full_mm`:** the matrix is full-mm shaped with **almost all entries structurally zero**; the matched quantile lands at a **very high** `t`, so `imp_complexity` mostly counts only the strongest calls and `total_gain` can look strongly negative even when peaks look good. Use `sparsity_match:2` or `quantile:0.995` for a kinder readout vs raw. |
| `sparsity_match:K` | Same but each pipeline is allowed K× as many calls as raw. `K = 2` gives sum-of-gain ≈ `1 × raw_nnz`; useful when you want absolute positive gain. |
| `quantile:Q` | Top `(1 - Q)` fraction of imputed values genome-wide on the modeled rows (e.g. `quantile:0.99` keeps the top 1%). |
| `fixed:T` | Numeric, same value across pipelines. Only meaningful when pipelines are on a shared scale. |

The threshold for `sparsity_match` and `quantile` is estimated from a 4 M-coord
random sample of imputed values within the modeled subspace (`--sample-size`).
The per-cell counts above the chosen threshold are then computed in one
streaming pass: chunks of `chunk_rows` (default 2000) materialise a small
dense block per pipeline, threshold, and accumulate column sums.

```bash
python downstream/complexity_gain.py \
  --mm-dir   /work/.../mm \
  --out-dir  /work/.../downstream/complexity_gain \
  --threshold-mode sparsity_match \
  --input cisTopic=/work/.../cistopic_ctcf/impute \
  --input FITS=/work/.../FITS/work/ctcf/impute \
  --input scOpen=/work/.../scOpen/work/ctcf/impute \
  --input MAGIC=/work/.../MAGIC/work/ctcf/impute \
  --input PUscOpen=/work/.../PUscOpen/work/ctcf/impute
```

Outputs (under `--out-dir`):

| file | contents |
|---|---|
| `complexity_gain.tsv` | per-cell long-form: `input, kind, barcode, threshold, raw_complexity, imp_complexity, gain` (one row per (pipeline, cell)) |
| `complexity_gain_summary.json` | per-pipeline: chosen threshold + origin string, raw/imp/gain stats (mean / median / p10 / p90 / min / max / std), `total_gain` summed across cells |
| `complexity_gain_gain_hist.png` | overlaid histograms of per-cell gain (dotted line at 0) |
| `complexity_gain_scatter.png` | per-pipeline panel of `raw_complexity` vs `imp_complexity` per cell, log-log with `y = x` diagonal — shows whether low-raw cells gain disproportionately |
| `complexity_gain_bars.png` | median per-cell gain bar chart |

### Reading the output

* `total_gain ≈ 0` under `sparsity_match` is **expected when the threshold is
  exact**; sampling noise can move it slightly. **PUscOpen** full-mm CSR often
  shows **large negative `total_gain`**: the chosen `t` sits near the top of the
  imputed distribution, so mid-level imputed mass is under-counted vs raw
  nonzeros (see table note). Use `sparsity_match:2` when benchmarking PUscOpen
  on this script. The metric of interest is still the *shape* of the per-cell
  gain distribution (dropout recovery vs smoothing).
* `total_gain > 0` under `sparsity_match:K` (`K > 1`) measures absolute
  uplift — `total_gain ≈ (K − 1) × raw_nnz_on_modeled`. Plotting it under
  `K = 2` alongside `K = 1` lets you see "what does another `raw_nnz` worth
  of calls do to the per-cell distribution?"
* The **scatter plot** is the most informative diagnostic: a healthy
  imputer's cloud sits *above* the diagonal for the low-x (low-raw) region
  (those cells gain complexity) and tracks the diagonal for the high-x
  region (high-raw cells already see what they should). A method that's
  flatlining horizontally (similar `imp_complexity` for every raw level) is
  collapsing per-cell variation.
* Per-cell gain comparisons across pipelines are only fair when the
  threshold mode is the same. `sparsity_match` keeps cross-pipeline totals
  identical so the distribution shapes are directly comparable.

SLURM:

```bash
sbatch downstream/slurm/complexity_gain.sbatch
# defaults: sparsity_match across the five pipelines.
# For 2x uplift:
sbatch --export=ALL,THRESHOLD_MODE=sparsity_match:2 downstream/slurm/complexity_gain.sbatch
# For top-1% calls per pipeline:
sbatch --export=ALL,THRESHOLD_MODE=quantile:0.99 downstream/slurm/complexity_gain.sbatch
```

## Conda env

All seven scripts (`prepare_pos_neg_bins.R`, `compare_pos_neg.py`,
`sensitivity_specificity.py`, `bin_sensitivity_specificity.py`,
`umi_distribution.py`, `raw_vs_imputed_diff.py`, `complexity_gain.py`) work
in any of the pipeline conda envs:

* `cistopic` — already includes R + Bioconductor (`GenomicRanges`,
  `rtracklayer`) needed by `prepare_pos_neg_bins.R`, plus h5py / scipy for
  the Python steps.
* `fits` / `scopen` / `magic` — also have h5py / scipy / scikit-learn; for
  the R step you need the Bioconductor packages, so prefer `cistopic` there.
  The `CONDA_ENV` env var on each SLURM template selects which env to
  activate.
