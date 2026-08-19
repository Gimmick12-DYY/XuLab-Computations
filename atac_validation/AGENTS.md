# Agent guide: `atac_validation/`

Orientation for an AI coding agent (Cursor) adapting these scripts. Read this
before editing. Full runbook is in `README.md`; this file is conventions + gotchas.

## What this module does
Validates/extends the imputer's open-chromatin mask using independent WT HEK293T
ATAC-seq: download → per-dataset MACS2 peaks → bin-level overlap vs the current
OmniATAC mask → union mask (drop-in `open_chromatin_bed`) → per-TF open-positive
recovery. Runs on Longleaf SLURM; not run locally.

## Non-negotiable conventions
- **Overlap = binary bin membership**, computed by `unified/scripts/_regions.py`
  (`bins_overlap_bed`), imported via `scripts/_maskutil.py`. Do NOT reimplement
  overlap or swap in `bedtools coverage` for the concordance step — the mask that
  gates imputation is defined by this exact function, and the numbers must match it.
- **Bin universe** is the unified `regions.tsv` (`chr:start-end`, ~3.03M bins);
  any TF's `impute/regions.tsv` works (CTCF is canonical). Never hardcode a bin list.
- **`open_chromatin_bed` is an interval BED, not a bin list.** The union output
  (`build_union_mask.py`) merges peak *intervals* so it drops straight into the
  unified config; keep it that way.
- **Paths flow through `scripts/lib.sh`.** Every path is an overridable env var
  (`XULAB`, `GENOME_FA`, `BT2_INDEX`, `BLACKLIST_BED`, `REGIONS_TSV`,
  `OMNIATAC_PEAKS`, `MACS_Q`, `CONDA_ENV`). Add new paths there, don't inline them.
- **SLURM style** mirrors `unified/slurm/*`: `#SBATCH` header, `set -euo pipefail`,
  `source .../lib.sh`, array index → item, conda env with a
  `/work → /nas/longleaf/home` fallback.

## Data flow (what feeds what)
```
samples.tsv ─(00 download)→ work/<ds>/fastq
           ─(01 align)→ data/HEK293T_ATAC_<ds>_MACS2.narrowPeak.gz
OmniATAC + 2 new peaks ─(02 peak_overlap)→ results/mask_overlap.tsv
                        ─(02 build_union)→ data/HEK293T_ATAC_union3.bed.gz
union + downstream bins ─(03 recovery)→ results/open_positive_recovery_*.tsv
```

## Common adaptations
- **Add a dataset:** append `keep=yes` rows to `metadata/samples.tsv` (cols:
  `dataset gsm srr layout group keep note`), add the tag to the `DATASETS=(...)`
  array in `01_align_call_peaks.sbatch`, and add a `--mask tag=...narrowPeak.gz`
  line in `02`/`03`. Download array size = count of `keep=yes` rows.
- **Change peak stringency:** set `MACS_Q` (default `0.01`) — don't edit the
  macs2 line directly.
- **Score different positives:** `03` reads `bins_<tf>_<MODE>/pos_neg_bins.tsv`;
  override `BINS_DIR`, `MODE`, `TFS`. The recovery script auto-detects the region
  and label columns (`--region-col`/`--label-col`/`--pos-value` to force).
- **Peak-level (bp/coverage) overlap** is intentionally absent. If asked for it,
  add a *separate* `bedtools jaccard`/`coverage` step — do not replace the
  bin-level analysis.

## Gotchas
- GSE152177: only `group=AA` / rs4420550 A/A is WT (`keep=yes`); G/G is
  `keep=no`. Pooling reads `dataset_runs` → changing `keep` rebuilds the pool
  via `01b` without re-aligning.
- After `samtools merge`, always `samtools sort` before `index` (merged BAMs can
  claim `SO:coordinate` but fail with "Chromosome blocks not continuous").
- Bioconda `macs2` can fail on RHEL9 (`undefined symbol: __log_finite`). Fix:
  `pip install --no-binary MACS2 'MACS2==2.2.9.1'` inside the atac env.
- `_maskutil.py` finds `_regions.py` via `parents[2]/unified/scripts`; keep the
  `atac_validation/scripts/` depth or fix that path.
- SLURM copies the `.sbatch` into a spool dir, so never
  `source "$(dirname "$0")/../scripts/lib.sh"`. Source via
  `${XULAB:-/work/users/d/y/dyy12/XuLab}/atac_validation/scripts/lib.sh`.
- `data/` and `atac_validation/{work,ref}/` are gitignored. Peaks/union BED land
  in `data/` (untracked, like the existing OmniATAC symlink). Only `results/*.tsv`
  + code are tracked (`*.png/*.pdf` ignored).
- Blacklist filter self-skips with a warning if `BLACKLIST_BED` is missing — a
  silent no-blacklist run is possible; check the log if peak counts look high.
- All runs in `samples.tsv` are PAIRED; the aligner assumes `_1/_2` FASTQ. Handle
  layout explicitly if you ever add a SINGLE-end run.
