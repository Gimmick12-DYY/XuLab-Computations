# data_prep — upstream input prep for the scBasset / scBasset_TF pipelines

Builds the `.rds` files that the imputation pipelines consume at
`paths.input_rds`. Standard MACS3 peak matrix workflow — no extra filtering.

## What it does

1. **MACS3 pseudo-bulk peak calling** on the fragment BED with the standard
   CUT&Tag parameters:
   `--nomodel --shift -75 --extsize 150 -q 0.01 --keep-dup all`.
2. **bedtools intersect** to assign each fragment to peaks per barcode.
   Streamed via subprocess pipe — no intermediate file on disk.
3. **Sparse matrix build** with binarized counts.
4. **mm intermediate** → R subprocess → `dgCMatrix` RDS with
   `chr:start-end` rownames and barcode colnames.

No blacklist filter, no chromosome whitelist, no peak/cell QC thresholds.
Downstream pipelines (e.g. `scBasset_TF/scripts/02_prepare_seqs.py`) apply
their own chrom + blacklist + N-content filtering, so doing it here too
would be redundant.

## Dependencies

```bash
conda env create -f environment.yml
conda activate data_prep
```

The env brings `macs3`, `bedtools`, `samtools`, `r-base`, `r-Matrix`,
`r-optparse`, plus numpy/scipy/pandas. If your cluster has its own
`macs3`/`bedtools` modules, drop those from `environment.yml` and
`module load` them instead.

## Usage

Wired inputs live in `config.yaml` (`data/TF.CTCF.bed`, `data/TF.ZNF282.bed` →
`data/<TF>.pmat.mtx.rds`). Run one TF:

```bash
bash data_prep/run_build_peak_matrix.sh CTCF
# or on the cluster:
TF=ZNF282 sbatch data_prep/slurm/build_peak_matrix.sbatch
```

Low-level CLI (paths resolved manually):

```bash
python data_prep/build_peak_matrix.py \
    --fragments /path/to/TF.CTCF.bed.gz \
    --out       /path/to/data/CTCF.pmat.mtx.rds \
    --tf-name   CTCF
```

CLI knobs:

| Flag | Default | Effect |
|---|---|---|
| `--fragments` | required | 5-col BED(.gz): `chr, start, end, barcode, count` |
| `--out` | required | Output `.rds` path |
| `--tf-name` | `CTCF` | Used in MACS3 sample name (`<tf>_pseudo_peaks.narrowPeak`) |
| `--work-dir` | `<out>.work/` | Intermediate files (MACS3 outputs, peaks BED, mm/) |

Final stdout block:

```
================================================================
  Output:   /path/to/data/CTCF_peak_matrix.rds
  Matrix:   <n_peaks> peaks x <n_cells> cells
  nnz:      <count>
  Sparsity: 0.99xxxx   (x.xxxx% nonzero)
================================================================
```

## Integration with scBasset_TF

Point a config at the output RDS:

```yaml
# scBasset_TF/configs/alltf_peak.yaml (or a new ctcf_peak.yaml)
paths:
  input_rds: /path/to/data/CTCF_peak_matrix.rds
run:
  input_kind: peak
```

Then run the pipeline as usual:

```bash
CFG=configs/alltf_peak.yaml sbatch slurm/99_full_pipeline.sbatch
```

Step 01 (`01_export_rds_to_mm.R`) reads the RDS, writes the mm tier, and
the rest of the pipeline takes over. Filters that this script no longer
applies (chrom whitelist, blacklist) are applied at step 02.

## File layout

```
data_prep/
├── README.md                 (this file)
├── config.yaml               (TF -> fragments / output RDS under data/)
├── environment.yml           (conda env)
├── run_build_peak_matrix.sh  (driver: config -> build_peak_matrix.py)
├── slurm/build_peak_matrix.sbatch
├── build_peak_matrix.py      (MACS3 -> bedtools -> mm -> RDS)
└── save_peak_matrix.R        (mm -> dgCMatrix RDS)
```

## Naming convention

The default name `CTCF_peak_matrix.rds` follows the user's spec. The newer
repo-internal convention seen in `data/` is `<TF>.pmat.mtx.rds` (parallel
to `<TF>.bin1000.mtx.rds`). Both names work — the downstream pipeline only
looks at the file's contents, not its name.
