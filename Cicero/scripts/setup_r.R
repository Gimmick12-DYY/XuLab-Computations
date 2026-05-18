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
#   module unload r    # if you previously `module load r` on Longleaf
#   Rscript scripts/setup_r.R
# -----------------------------------------------------------------------------

# Longleaf users often `module load r`, which prepends a read-only system
# library to .libPaths() *before* the conda env. BiocManager then tries to
# install into /nas/longleaf/rhel9/apps/r/... and fails even in (cicero).
use_conda_r_lib <- function() {
  prefix <- Sys.getenv("CONDA_PREFIX", unset = "")
  if (!nzchar(prefix)) {
    stop(
      "CONDA_PREFIX is not set. Activate the cicero env first:\n",
      "  conda activate cicero\n",
      "If you use `module load r`, run `module unload r` before setup."
    )
  }
  lib <- file.path(prefix, "lib", "R", "library")
  if (!dir.exists(lib)) {
    stop("Expected conda R library at ", lib, " but it is missing.")
  }
  .libPaths(c(lib))
  invisible(lib)
}

assert_lib_writable <- function() {
  lib <- .libPaths()[1L]
  if (!dir.exists(lib) || file.access(lib, 2L) != 0L) {
    stop(
      "R library is not writable: ", lib, "\n",
      "This usually means the cluster R module is shadowing conda. Fix:\n",
      "  module unload r\n",
      "  conda deactivate && conda activate cicero\n",
      "  Rscript scripts/setup_r.R"
    )
  }
}

use_conda_r_lib()
assert_lib_writable()
message("[setup_r] Using R library: ", .libPaths()[1L])

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
