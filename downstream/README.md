# Downstream cross-pipeline analyses

Scripts that consume the `imputed_Prc_hdf5_all.h5` written by either pipeline
(cisTopic or FITS) plus the raw `mm/` matrix, and produce comparisons.

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
**original mm row space** (i.e. directly index into `imputed_Prc_hdf5_all.h5`'s
`Prc` rows).

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
index file works for any imputed HDF5 derived from that input.

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
`PATH` is either a directory containing `matrix.mtx.gz` (raw counts) or an
HDF5 file with a `Prc` dataset (imputed). It computes the per-bin total signal
at the positive and negative bins, reports AUROC / KS / median ratio, and
writes a multi-panel ECDF figure plus a single-panel AUROC bar chart.

```bash
python downstream/compare_pos_neg.py \
  --bins-tsv /work/.../downstream/bins/pos_neg_bins.tsv \
  --out-dir  /work/.../downstream/pos_neg \
  --input raw=/work/.../cistopic_ctcf/mm \
  --input cisTopic=/work/.../cistopic_ctcf/impute/imputed_Prc_hdf5_all.h5 \
  --input FITS=/work/.../FITS/work/ctcf/impute/imputed_Prc_hdf5_all.h5
```

Outputs (under `--out-dir`):

| file | contents |
|---|---|
| `compare_pos_neg.png` | one ECDF panel per `--input`, pos vs neg curves overlaid |
| `compare_pos_neg_auroc.png` | AUROC bar chart, one bar per input |
| `compare_pos_neg.json` | per-input AUROC, KS statistic + p-value, pos/neg medians, log2(median ratio), zero fractions |
| `compare_pos_neg.tsv` | long-form per-bin signal table (`input`, `group`, `bin_idx_orig`, `signal`) for replotting |

SLURM:

```bash
sbatch --export=ALL,\
BINS_TSV=/work/.../downstream/bins/pos_neg_bins.tsv,\
OUT_DIR=/work/.../downstream/pos_neg,\
INPUTS="raw=/work/.../cistopic_ctcf/mm cisTopic=/work/.../cistopic_ctcf/impute/imputed_Prc_hdf5_all.h5 FITS=/work/.../FITS/work/ctcf/impute/imputed_Prc_hdf5_all.h5" \
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

## Conda env

Both scripts work in either pipeline's conda env:

* `cistopic` — already includes R + Bioconductor (`GenomicRanges`,
  `rtracklayer`) needed by step 1, plus h5py/scipy for step 2.
* `fits` — also has h5py/scipy/scikit-learn; for step 1 you need the
  Bioconductor packages, so prefer `cistopic` for the R step. The env
  `CONDA_ENV` variable on the SLURM template selects which env to activate.
