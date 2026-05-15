#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# setup_r.R
#
# One-shot bootstrap: install monocle3 + cicero (cole-trapnell-lab GitHub
# branches) on top of the conda env defined in ../environment.yml. Both
# packages are NOT on Bioconductor in their Monocle3-compatible flavour, so
# we have to pull them from GitHub once per environment.
#
# Run after creating the conda env:
#   conda env create -f environment.yml
#   conda activate cicero
#   Rscript scripts/setup_r.R
# -----------------------------------------------------------------------------

if (!requireNamespace("remotes", quietly = TRUE))
  install.packages("remotes", repos = "https://cloud.r-project.org")
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager", repos = "https://cloud.r-project.org")

# Bioconductor deps that may be missing depending on the resolver.
BiocManager::install(
  c("BiocGenerics", "S4Vectors", "IRanges", "GenomicRanges",
    "DelayedArray", "DelayedMatrixStats", "limma",
    "Biobase", "lme4", "batchelor", "HDF5Array",
    "terra", "ggrastr"),
  update = FALSE, ask = FALSE
)

remotes::install_github("cole-trapnell-lab/monocle3", upgrade = "never")
remotes::install_github("cole-trapnell-lab/cicero-release",
                        ref = "monocle3", upgrade = "never")

suppressPackageStartupMessages({
  library(monocle3)
  library(cicero)
})

cat("[setup_r] monocle3 + cicero are importable.\n")
