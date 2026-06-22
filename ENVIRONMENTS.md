# Conda environments

We deliberately do **not** use one mega-env. A single env is impossible here:
Python pins conflict (scOpen needs 3.9; scBasset/MAGIC/PUscOpen/Cicero want 3.10;
cisTopic/unified want 3.11) and TensorFlow vs PyTorch drag conflicting CUDA
wheels. Instead, envs are grouped by **compatible dependency stack**, and
redundant ones have been merged.

## Active environments

| Env | Build from | Python | Covers |
|-----|-----------|--------|--------|
| **cistopic** | `cisTopic/environment.yml` | 3.11 | cisTopic, scBasset_cisTopic(_PUscOpen) cascades, **all of `downstream/`** (peak_coverage, prepare_pos_neg_bins desert mode, build_peak_matrix, compare/sensitivity/bigwig, ...). Universal CPU analysis env: R + Bioconductor + numpy/scipy + **macs3 + bedtools + samtools**. |
| **scbasset** | `scBasset/environment.yml` | 3.10 | scBasset **and** scBasset_TF (TensorFlow GPU + R + samtools). |
| **unified** | `unified/environment.yml` | 3.11 | unified model (PyTorch GPU). Kept lean and isolated. |
| **puscopen** | `PUscOpen/environment.yml` | 3.10 | PUscOpen and scBasset_PUscOpen (PU classifier + scOpen NMF + heavy Bioconductor). |
| **magic** | `MAGIC/environment.yml` | 3.10 | MAGIC (magic-impute). |
| **scopen** | `scOpen/environment.yml` | 3.9 | scOpen (hard 3.9 pin). |
| **borzoi** | `Borzoi/environment.yml` | 3.11 | Borzoi (torch + borzoi-pytorch). |
| **fits** | `FITS/environment.yml` | 3.11 | FITS (clones FITSpython into the env). |
| **cicero** | `Cicero/environment.yml` | 3.10 | Cicero (R 4.4 + monocle3). |

## Merges already done

- **`data_prep` -> `cistopic`** — macs3 + bedtools + samtools folded into cistopic,
  so the entire `downstream/` stack runs in one env (no more "tool X not on PATH").
- **`scbasset_tf` -> `scbasset`** — the two env files were identical.

The merged-away `data_prep/environment.yml` and `scBasset_TF/environment.yml` are
kept as signpost stubs. Every sbatch's `CONDA_ENV` default was repointed.

## Could be merged later (need a cluster test-solve first)

- `magic` + `puscopen` -> one py3.10 analysis env (magic-impute + heavy
  Bioconductor; risk of a slow/failing solve, so not done blindly).
- `borzoi` -> `unified` (both py3.11 torch) — left separate to keep the
  actively-developed `unified` env clean.

`scopen` (3.9) and `cicero` (R4.4/monocle3) cannot practically merge.

## Build

```bash
conda env create -f cisTopic/environment.yml     # cistopic  (analysis + downstream)
conda env create -f scBasset/environment.yml     # scbasset  (+ scBasset_TF)
conda env create -f unified/environment.yml      # unified
# ...others only if you run those pipelines
```

Every sbatch takes `CONDA_ENV=<name>` as an override, so you can always point a
job at a different env without editing the script.
