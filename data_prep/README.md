# data_prep — upstream input prep for the scBasset / scBasset_TF pipelines

Builds the `.rds` files that the imputation pipelines consume at
`paths.input_rds`. Today it ships one entry point: a CUT&Tag fragment file
→ peak × cell binary `dgCMatrix` RDS.

## What it does

1. **MACS2 pseudo-bulk peak calling** on the fragment BED with the standard
   CUT&Tag parameters:
   `--nomodel --shift -75 --extsize 150 -q 0.01 --keep-dup all`.
2. **Peak filter**: drop peaks not on `chr1..22 / chrX / chrY` and any peak
   overlapping the ENCODE hg38 blacklist (Boyle Lab v2). Blacklist is
   auto-discovered under `<repo>/downstream/cache/` (the same cache scBasset
   uses) and downloaded on first run if missing.
3. **bedtools intersect** to assign each fragment to peaks per barcode.
   Streamed via subprocess pipe — no intermediate file on disk.
4. **Sparse matrix build** with binarized counts.
5. **Two-pass filter**: peaks present in `< min_cells_per_peak` cells are
   dropped first; then cells with `< min_peaks_per_cell` remaining peaks
   are dropped.
6. **mm intermediate** → R subprocess → `dgCMatrix` RDS with
   `chr:start-end` rownames and barcode colnames. The RDS drops directly
   into `paths.input_rds` of any pipeline that uses
   `scBasset/scripts/01_export_rds_to_mm.R`.

## Dependencies

```bash
conda env create -f environment.yml
conda activate data_prep
```

The env brings `macs2`, `bedtools`, `samtools`, `r-base`, `r-Matrix`,
`r-optparse`, plus numpy/scipy/pandas. If your cluster has its own
`macs2`/`bedtools` modules, you can drop those from `environment.yml` and
`module load` them instead.

## Usage

```bash
python data_prep/build_peak_matrix.py \
    --fragments /path/to/TF.CTCF.bed.gz \
    --out       /path/to/data/CTCF_peak_matrix.rds \
    --tf-name   CTCF
```

CLI knobs:

| Flag | Default | Effect |
|---|---|---|
| `--fragments` | required | 5-col BED(.gz): `chr, start, end, barcode, count` |
| `--out` | required | Output `.rds` path |
| `--tf-name` | `CTCF` | Used in MACS2 sample name (`<tf>_pseudo_peaks.narrowPeak`) |
| `--blacklist` | auto | Falls back to `<repo>/downstream/cache/hg38-blacklist.v2.bed.gz`; downloads if missing |
| `--work-dir` | `<out>.work/` | Intermediate files (MACS2 outputs, filtered peaks BED, mm/) |
| `--min-cells-per-peak` | 2 | Drop peaks present in fewer cells |
| `--min-peaks-per-cell` | 200 | Drop cells with fewer remaining peaks |

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

Step 01 (`01_export_rds_to_mm.R`) will read the RDS, write the mm tier,
and the rest of the pipeline takes over.

## File layout

```
data_prep/
├── README.md                 (this file)
├── environment.yml           (conda env)
├── build_peak_matrix.py      (Python driver: MACS2 -> filter -> bedtools -> mm -> RDS)
└── save_peak_matrix.R        (mm -> dgCMatrix RDS subprocess)
```

## Naming convention

The default name `CTCF_peak_matrix.rds` follows the user's spec. The newer
repo-internal convention seen in `data/` is `<TF>.pmat.mtx.rds` (parallel
to `<TF>.bin1000.mtx.rds`). Both names work — the downstream pipeline only
looks at the file's contents, not its name.
