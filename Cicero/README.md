# Cicero co-accessibility imputation

Parallel pipeline to `cisTopic/`, `FITS/`, `scOpen/`, `MAGIC/`, `PUscOpen/`.
Replaces the fixed `+/- window_bp` neighbour rule used inside the other
imputers' propagation steps with a **learned**, biologically meaningful
peak-peak adjacency derived from Cicero (Pliner et al. 2018).

## What this gives you

For each cell `c`, raw observations propagate to bins linked by Cicero
co-accessibility:

```
prop_sum[r2, c]   = sum_{r1 ~ r2 in cell c, raw[r1,c]>0} coaccess(r1, r2)
prop_count[r2, c] = #{r1 ~ r2 with raw[r1,c] > 0}
value[r2, c]      = weight * prop_sum / prop_count    (count >= min_links)
```

where `r1 ~ r2` means `coaccess(r1, r2) >= coaccess_threshold` (Pliner et
al. recommend 0.05). By default we only fill bins where `raw[r2, c] = 0`
and union with raw, so the output is "raw + extra calls".

**Limitation (shared with cisTopic/scOpen/MAGIC/PUscOpen).** Cicero only
links bins that were observed in at least one cell, so the *target* bin
must already exist in the raw matrix somewhere. This pipeline fills
`(r, c)` entries that one cell didn't observe but a co-accessible bin
did — it does **not** introduce signal at bins that were zero across all
cells. That regime needs sequence-based models (scBasset / Enformer);
see the next pipeline in the roadmap.

## Pipeline shape

```
00_inspect_rds.R       sanity-check the input RDS (same script as cisTopic)
01_export_rds_to_mm.R  RDS -> mm/{matrix.mtx.gz, regions.tsv.gz, barcodes.tsv.gz}
02_build_cds.R         mm -> Monocle3 cell_data_set + LSI + UMAP -> cds/cds.rds
03_run_cicero.R        cds -> peak1 x peak2 co-accessibility table
05_impute.py           raw + co-access -> sparse mm-aligned imputed CSR
```

(Step 04 is intentionally skipped to mirror the MAGIC pipeline's gap.)

Outputs land at `<work>/impute/{matrix_csr.npz, regions.tsv, barcodes.tsv,
meta.json}` — the standard cross-pipeline schema.

## Setup

```bash
# 1) conda env
conda env create -f environment.yml
conda activate cicero

# 2) Install monocle3 + cicero from GitHub (not on Bioconductor in their
#    Monocle3-compatible flavour). Run once per env:
Rscript scripts/setup_r.R
```

## Run

```bash
# end-to-end (~1 day on 16 CPUs / 256 GB for ~119k peaks x 10k cells)
sbatch slurm/99_full_pipeline.sbatch

# or step-by-step
sbatch slurm/01_export.sbatch
sbatch slurm/02_build_cds.sbatch    # 128 GB, ~2-4 h
sbatch slurm/03_cicero.sbatch       # 256 GB, ~6-18 h (heavy step)
sbatch slurm/05_impute.sbatch       # 128 GB, ~1-2 h
```

For a smoke test on a fraction of the data, set `cicero.max_cells: 2000` in
the config (or `sbatch --export=ALL,CFG=/path/to/smoke.yaml ...`).

## Hooking into the cross-pipeline comparison

After `05_impute.py` finishes, add it to the universal comparators:

```bash
python downstream/compare_pos_neg.py \
  --input cisTopic=/path/to/cisTopic/work/ctcf/impute \
  --input fits=/path/to/FITS/work/ctcf/impute \
  --input scopen=/path/to/scOpen/work/ctcf/impute \
  --input magic=/path/to/MAGIC/work/ctcf/impute \
  --input puscopen=/path/to/PUscOpen/work/ctcf/impute \
  --input cell_kgraph=/path/to/downstream/cell_kgraph \
  --input cicero=/path/to/Cicero/work/ctcf/impute \
  ...
```

## Tuning knobs (configs/default.yaml)

| Knob                            | Effect                                                                      |
| ------------------------------- | --------------------------------------------------------------------------- |
| `cicero.min_cells_per_peak: 2`  | Drop singleton bins before fitting (stabilises graphical Lasso).            |
| `cicero.k_metacell: 50`         | Cells per metacell. Higher = smoother but lower resolution.                 |
| `cicero.window_bp: 500000`      | Maximum genomic distance Cicero considers for a link.                       |
| `cicero.sample_num: 100`        | Per-window samples. 50 = faster, 200 = more stable.                         |
| `cicero.max_cells: 0`           | 0 = all cells; set 2000 for a fast smoke test.                              |
| `impute.coaccess_threshold: 0.05` | Pliner et al. default. Raise (e.g. 0.25) for stricter, fewer propagations. |
| `impute.min_links: 1`           | Minimum supporting links per (target, cell). Raise to 2-3 for conservative. |
| `impute.weight: 1.0`            | Multiplier on propagated value. < 1 marks them as softer evidence.          |
| `impute.only_zero_targets: true` | Only fill `raw == 0` bins (canonical imputation).                          |
| `impute.keep_raw: true`         | Union the propagated matrix with raw.                                       |

## Where this fits in the broader roadmap

This is item **3** of the "discover new bins" roadmap. Items completed:

1. **Motif-aware propagation** — `downstream/fetch_ctcf_motif_bed.sh`
   feeds `cisTopic/05_impute.py` and `PUscOpen/06_impute.py` via
   auto-detected `target_bed`.
2. **Cell-graph kNN propagation** — `downstream/cell_graph_kneighbour.py`
   propagates along the cell axis (orthogonal to bin-axis approaches).
3. **Cicero co-accessibility** — *this pipeline*. Replaces the fixed-window
   bin-bin adjacency with a learned one.
4. **Sequence-based imputation (scBasset / Enformer)** — *next*. The only
   approach that can fill bins observed in **zero** cells.
