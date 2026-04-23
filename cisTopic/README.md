# cisTopic TF imputation pipeline

A standalone pipeline that uses [pycisTopic](https://github.com/aertslab/pycisTopic)
to impute a sparse single-cell transcription-factor binding matrix via Latent
Dirichlet Allocation. The initial target is a CTCF matrix of 3,031,053 1-kb
bins × 10,410 cells stored as an `.rds` file, but nothing here is CTCF-specific.

## What "imputation" means here

Following Bravo González-Blas *et al.* (Nat. Methods 2019), cisTopic imputes
drop-outs through the **predictive distribution**:

```
P(r | c) = sum_k  P(r | T_k) * P(T_k | c)        (==  phi^T @ theta)
```

where

* `theta` (shape `K x C`) is the topic-per-cell distribution and
* `phi`   (shape `R x K`) is the region-per-topic distribution,

both returned by LDA. So "imputation" is a deterministic product of the two
matrices after LDA has converged — there is no separate imputation model.

## Why pycisTopic (not the original R package)

The paper's R package is effectively frozen. For this scale (≈3×10⁶ regions,
≈10⁴ cells) the original `lda`-based Gibbs sampler is impractical; pycisTopic
wraps **MALLET**, which is multi-threaded and memory-tunable and is what the
authors now recommend for large datasets.

## Pipeline layout

```text
cisTopic/
├── environment.yml                 conda env (pycisTopic + R Matrix)
├── configs/default.yaml            all knobs
├── scripts/
│   ├── 00_inspect_rds.R            read-only sanity check on the .rds
│   ├── 01_export_rds_to_mm.R       .rds -> Matrix Market + region/cell TSVs
│   ├── 02_build_cistopic_obj.py    filter & build CistopicObject (pickle)
│   ├── 03_run_lda_mallet.py        MALLET LDA over a topic grid
│   ├── 04_select_model.py          metrics + plots + pick K
│   ├── 05_impute.py                save theta/phi (+ optional HDF5 P(r|c))
│   ├── 06_downstream.py            optional: UMAP + topic binarisation
│   └── _cfg.py                     config loader shared by 02-06
└── slurm/                          sbatch templates for every step
```

## One-time setup

```bash
# 1. Create the environment
conda env create -f cisTopic/environment.yml
conda activate cistopic

# 2. Download MALLET (Java)
wget https://github.com/mimno/Mallet/releases/download/v202108/Mallet-202108-bin.tar.gz
tar -xf Mallet-202108-bin.tar.gz
# point `paths.mallet_path` in configs/default.yaml at Mallet-202108/bin/mallet

# 3. Edit configs/default.yaml
#    - paths.input_rds   (your .rds file, e.g. CTCF_bin1000_mtx.rds)
#    - paths.work_dir    (scratch directory; lots of space needed)
#    - paths.mallet_path
#    - filter.* defaults are reasonable; raise/lower per your QC
#    - lda.n_topics grid (the paper's grids were 15-50)
```

## Running the whole pipeline

### Locally (small tests)

```bash
cd <repo-root>
CFG=cisTopic/configs/default.yaml

# Optional but recommended the first time around: inspect the .rds.
Rscript cisTopic/scripts/00_inspect_rds.R \
        --input $(yq .paths.input_rds $CFG)

Rscript cisTopic/scripts/01_export_rds_to_mm.R \
        --input  $(yq .paths.input_rds $CFG) \
        --outdir $(yq .paths.work_dir  $CFG)/mm
python  cisTopic/scripts/02_build_cistopic_obj.py --config $CFG
python  cisTopic/scripts/03_run_lda_mallet.py     --config $CFG
python  cisTopic/scripts/04_select_model.py       --config $CFG
python  cisTopic/scripts/05_impute.py             --config $CFG
python  cisTopic/scripts/06_downstream.py         --config $CFG   # optional
```

### On SLURM

```bash
cd <repo-root>
export CFG=cisTopic/configs/default.yaml

# Optional: one-shot .rds inspection.
sbatch cisTopic/slurm/00_inspect.sbatch

JID1=$(sbatch --parsable cisTopic/slurm/01_export.sbatch)
JID2=$(sbatch --parsable --dependency=afterok:$JID1 cisTopic/slurm/02_build.sbatch)

# Either serial over the whole K grid...
JID3=$(sbatch --parsable --dependency=afterok:$JID2 cisTopic/slurm/03_lda.sbatch)

# ...or an array, one job per K (faster if the cluster allows)
# JID3=$(sbatch --parsable --dependency=afterok:$JID2 cisTopic/slurm/03_lda_array.sbatch)

JID4=$(sbatch --parsable --dependency=afterok:$JID3 cisTopic/slurm/04_select.sbatch)
JID5=$(sbatch --parsable --dependency=afterok:$JID4 cisTopic/slurm/05_impute.sbatch)
sbatch         --dependency=afterok:$JID5 cisTopic/slurm/06_downstream.sbatch
```

## Key outputs

| Path (under `paths.work_dir`) | What it is |
|---|---|
| `mm/matrix.mtx.gz`, `mm/regions.tsv.gz`, `mm/barcodes.tsv.gz` | raw matrix exported from the `.rds` |
| `obj/cistopic_obj.pkl` | filtered CistopicObject (with the selected model attached after step 04) |
| `models/Topic_*.pkl`, `models/models.pkl` | trained MALLET LDA models |
| `select/model_selection.png`, `select/selected_model.pkl` | model-selection curves and winning K |
| `impute/cell_topic_theta.(npy\|parquet)` | `theta` (K × C) |
| `impute/region_topic_phi.(npy\|parquet)` | `phi` (R × K) |
| `impute/imputed_Prc_*.h5` | full `P(r\|c)` matrix (only when `impute.mode != theta_phi_only`) |
| `downstream/umap.(tsv\|png)` | topic-cell UMAP |
| `downstream/binarized_regions_*.pkl` | topic-specific region sets |

## Reality check: storage and time at 3M × 10K

* The raw count matrix (nnz ≈ 1–5 × 10⁸, binary) fits comfortably in tens of GB.
* After `filter.min_counts_per_region=30` typically only 5–30 % of the 3 M bins
  survive (TF binding is very sparse across the genome).
* MALLET training scales roughly linearly in nnz × iterations × K. On
  16 cores with K=30 and 500 iterations, expect several hours to a day per K
  at this scale — use the array sbatch.
* A full `P(r|c)` at R ≈ 1 M, C = 10,410 in float32 is ~40 GB dense; with gzip-4
  chunking in HDF5 it usually lands at 10–20 GB. If that's still too much,
  leave `impute.mode: theta_phi_only` and reconstruct slices on demand via
  `phi[rows] @ theta[:, cols]`.

## Reconstructing P(r|c) from theta/phi without the HDF5 dump

```python
import numpy as np
theta = np.load("impute/cell_topic_theta.npy")   # K x C
phi   = np.load("impute/region_topic_phi.npy")   # R x K
# Any slice, cheap in RAM:
P_slice = phi[row_idx] @ theta[:, col_idx]       # (len(row_idx), len(col_idx))
```

## References

* Bravo González-Blas C. *et al.* cisTopic: cis-regulatory topic modeling on
  single-cell ATAC-seq data. *Nature Methods* **16**, 397–400 (2019).
  https://doi.org/10.1038/s41592-019-0367-1
* Bravo González-Blas C. *et al.* SCENIC+: single-cell multiomic inference of
  enhancers and gene regulatory networks. *Nature Methods* (2023).
* pycisTopic: https://github.com/aertslab/pycisTopic
* MALLET: https://github.com/mimno/Mallet
