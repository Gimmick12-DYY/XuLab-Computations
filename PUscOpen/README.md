# PUscOpen — PU-refined, mask-restricted scOpen for genome-wide TF data

A two-stage pipeline that addresses the **over-imputation** problem we
observed when running scOpen on genome-wide 1-kb bins: scOpen was designed
for peak-by-cell matrices where every row is a candidate accessible site,
but ~95 % of the genome is heterochromatin / gene desert with zero true
accessibility. PUscOpen restricts the NMF to bins that *are plausibly
positive* and applies a sequence of biological priors so the imputed matrix
respects the structure of CTCF binding.

## What it does, in order

1. **Conservative candidate mask** *(step 02)*
   Build the union of ENCODE SCREEN cCREs (HEK293T-CA classes by default),
   CTCF-bound cCREs across the SCREEN registry, and optional CTCF motif
   hits. Subtract the ENCODE blacklist and (optionally) low-mappability
   regions.

2. **Positive-Unlabeled classifier** *(step 03)*
   Treat known-positive bins (top-N HEK293 CTCF cCREs ∩ HEK293T-CA, the
   same recipe as `Bin_posVSneg.ipynb`) as positives **`s = 1`**; treat
   every other bin as **unlabeled** rather than as a confirmed negative.
   Train a binary classifier with the Elkan-Noto (2008) calibration trick
   so the output is a calibrated `P(true positive | features)` per bin. A
   `hold_out_fraction` (default 20 %) of seed positives is reserved for the
   step-07 validation pass and is *not* used in training.

3. **scOpen on the refined mask** *(step 04)*
   Use the geometric mask ∩ `PU_score ≥ threshold` as the input row set.
   scOpen factorises only this restricted matrix, so bins outside the mask
   stay at zero in the final output and cannot be over-imputed.

4. **Neighborhood + PU post-filter** *(step 05)*
   CTCF binding is genomically structured. For each refined-mask bin, count
   co-located bins within ±10 kb whose imputed total exceeds the seed
   positives' median. Bins with too few qualifying neighbors are damped by
   ×0.25. Optionally multiply the per-(bin, cell) value by the bin's PU
   score so the final output is "scOpen says high signal **and** PU prior
   says plausibly accessible".

5. **Persist** *(step 06)* in the standard FITS/MAGIC schema
   `<work>/impute/{matrix.npy, regions.tsv, barcodes.tsv, meta.json}`, so
   the cross-pipeline `downstream/compare_pos_neg.py`,
   `sensitivity_specificity.py`, and `bin_sensitivity_specificity.py`
   scripts all consume it unchanged.

6. **Held-out validation** *(step 07)*
   Score the held-out seed positives + a size-matched random control set
   (drawn from bins *outside* the geometric mask) against the final imputed
   matrix. Report AUROC, AUPR, max-F1, recall@k, and full ROC curves.

## Pipeline layout

```text
PUscOpen/
├── environment.yml                conda env (scopen + sklearn + R + Bioconductor)
├── configs/default.yaml           all knobs, with the recommended defaults
├── scripts/
│   ├── 00_inspect_rds.R           read-only sanity check on the .rds
│   ├── 01_export_rds_to_mm.R      .rds -> Matrix Market + region/cell TSVs
│   ├── 02_build_mask.R            cCREs + motif + blacklist  ->  mask + bin features
│   ├── 03_pu_classifier.py        Elkan-Noto PU model  ->  refined mask + holdout
│   ├── 04_run_scopen.py           scOpen on the refined mask
│   ├── 05_postfilter.py           neighborhood-consistency + PU prior damping
│   ├── 06_impute.py               materialise final imputed matrix
│   ├── 07_validate.py             recovery of held-out positives
│   └── _cfg.py                    config loader (mm, mask, pu, scopen_run, ...)
└── slurm/                         00..07 sbatches + 99_full_pipeline
```

## One-time setup

```bash
# 1. Create the conda env (installs scopen from PyPI + R/Bioconductor)
conda env create -f PUscOpen/environment.yml
conda activate puscopen

# 2. Make sure the annotation BEDs are available. The easiest way is to
#    run downstream/prepare_pos_neg_bins.R once -- it downloads everything
#    into <downstream-cache-dir>/.
Rscript downstream/prepare_pos_neg_bins.R \
   --mm-dir   /work/.../cisTopic/scripts/cistopic_ctcf/mm \
   --out-dir  /work/.../downstream/bins \
   --cache-dir /work/.../downstream/cache

# 3. Edit configs/default.yaml
#    - paths.input_rds       (your .rds file)
#    - paths.work_dir        (scratch directory)
#    - paths.annotation_dir  (= downstream/cache/ from step 2)
```

## Running locally

```bash
cd <repo-root>
CFG=PUscOpen/configs/default.yaml

Rscript PUscOpen/scripts/00_inspect_rds.R      --input $(yq .paths.input_rds $CFG)
Rscript PUscOpen/scripts/01_export_rds_to_mm.R --input $(yq .paths.input_rds $CFG) \
                                               --outdir $(yq .paths.work_dir $CFG)/mm
Rscript PUscOpen/scripts/02_build_mask.R       --config $CFG
python  PUscOpen/scripts/03_pu_classifier.py   --config $CFG
python  PUscOpen/scripts/04_run_scopen.py      --config $CFG
python  PUscOpen/scripts/05_postfilter.py      --config $CFG
python  PUscOpen/scripts/06_impute.py          --config $CFG
python  PUscOpen/scripts/07_validate.py        --config $CFG
```

## Running on SLURM

```bash
cd <repo-root>
export CFG=PUscOpen/configs/default.yaml

JID1=$(sbatch --parsable PUscOpen/slurm/01_export.sbatch)
JID2=$(sbatch --parsable --dependency=afterok:$JID1 PUscOpen/slurm/02_build_mask.sbatch)
JID3=$(sbatch --parsable --dependency=afterok:$JID2 PUscOpen/slurm/03_pu.sbatch)
JID4=$(sbatch --parsable --dependency=afterok:$JID3 PUscOpen/slurm/04_scopen.sbatch)
JID5=$(sbatch --parsable --dependency=afterok:$JID4 PUscOpen/slurm/05_postfilter.sbatch)
JID6=$(sbatch --parsable --dependency=afterok:$JID5 PUscOpen/slurm/06_impute.sbatch)
sbatch         --dependency=afterok:$JID6 PUscOpen/slurm/07_validate.sbatch

# Or end-to-end in one allocation:
sbatch PUscOpen/slurm/99_full_pipeline.sbatch
```

## Key outputs

| Path (under `paths.work_dir`) | What it is |
|---|---|
| `mm/matrix.mtx.gz`, `regions.tsv.gz`, `barcodes.tsv.gz` | raw matrix exported from the `.rds` |
| `mask/mask.bed`, `mask/mask.tsv` | geometric candidate mask |
| `mask/bin_features.tsv` | per-bin features fed to the PU classifier |
| `mask/build_meta.json` | rule counts (cCRE / motif / blacklist) |
| `pu/pu_scores.tsv` | per-bin calibrated PU score |
| `pu/refined_mask.bed`, `pu/refined_mask.tsv` | mask ∩ (PU ≥ threshold) |
| `pu/holdout_positives.tsv` | seed positives reserved for step 07 |
| `pu/pu_meta.json` | classifier choice, c-hat, threshold |
| `scopen_input/` | 10X-format input fed to scOpen |
| `scopen_run/scOpen_peaks.txt`, `scOpen_barcodes.txt` | W, H from scOpen |
| `scopen_run/scOpen_error.{txt,pdf}` | rank vs reconstruction error |
| `postfilter/bin_signal.tsv` | per-bin imputed total, neighborhood support, damp |
| `postfilter/damping.npy` | the damping vector applied to W @ H |
| `postfilter/postfilter_meta.json` | thresholds and per-rule counts |
| **`impute/matrix.npy`** | **final imputed matrix (regions × cells, float32)** |
| `impute/regions.tsv`, `barcodes.tsv` | row / col labels |
| `impute/meta.json` | shape, scale_factor, source paths |
| `validate/validate.{tsv,json}` | per-threshold recall/spec on held-out positives |
| `validate/validate_roc.png`, `_recall_at_k.png` | curves |

## How to add PUscOpen to the cross-pipeline comparison

The `impute/` directory uses the same dense-`matrix.npy` schema as FITS and
MAGIC, so the downstream scripts work unchanged:

```bash
# Bin-level CTCF positive vs negative (Bin_posVSneg)
python downstream/compare_pos_neg.py \
  --bins-tsv /work/.../downstream/bins/pos_neg_bins.tsv \
  --out-dir  /work/.../downstream/pos_neg_PUscOpen \
  --input raw=/work/.../mm \
  --input cisTopic=/work/.../cistopic_ctcf/impute \
  --input FITS=/work/.../FITS/work/ctcf/impute \
  --input scOpen=/work/.../scOpen/work/ctcf/impute \
  --input MAGIC=/work/.../MAGIC/work/ctcf/impute \
  --input PUscOpen=/work/.../PUscOpen/work/ctcf/impute

# Bin-level sens/spec using the cCRE-derived ground truth
python downstream/bin_sensitivity_specificity.py \
  --bins-tsv /work/.../downstream/bins/pos_neg_bins.tsv \
  --out-dir  /work/.../downstream/bin_sens_spec_PUscOpen \
  --input raw=/work/.../mm \
  --input PUscOpen=/work/.../PUscOpen/work/ctcf/impute
```

## Tuning notes

- `mask.use_cCRE_293T_CA: true` (default) is the strictest geometric mask.
  Switch to `use_cCRE_293T: true` for a more permissive cCRE universe.
- `pu.classifier: rf` (random forest) is more expressive than the default
  `logistic` but less calibrated; if you switch, expect Elkan-Noto `c_hat`
  to drift.
- `pu.refined_mask_threshold: auto` picks the median PU score over seed
  positives, which is a reasonable "everything as positive-ish as the
  median seed bin". Pass a float in `[0, 1]` to override.
- `postfilter.multiply_by_pu: true` is the strongest hammer against
  over-imputation but couples the bin-level prior to per-cell predictions.
  Set false if you want scOpen's per-(bin, cell) values preserved
  unmultiplied.
- The step-07 validation report quantifies the recall/precision trade-off
  on the *held-out* seed positives. Tune `pu.refined_mask_threshold` and
  `postfilter.neighbor_threshold` against the validation AUROC, not against
  the bin-level CTCF cCRE analysis (which uses the *full* seed set and is
  therefore over-optimistic).

## References

* Elkan C., Noto K. *Learning Classifiers from Only Positive and Unlabeled
  Data*. SIGKDD 2008.
* Li Z. *et al.* *Chromatin-accessibility estimation from single-cell
  ATAC-seq data with scOpen*. Nat. Comm. 12, 6386 (2021).
  https://doi.org/10.1038/s41467-021-26530-2
* ENCODE SCREEN cCRE registry V4: https://screen.wenglab.org
