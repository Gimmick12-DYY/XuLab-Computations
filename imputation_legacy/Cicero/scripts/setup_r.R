#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# setup_r.R — install cicero@monocle3 after conda env (environment.yml).
#
#   module unload r
#   conda activate cicero
#   conda env update -f environment.yml --prune
#   Rscript scripts/setup_r.R
# -----------------------------------------------------------------------------

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
      "Usually `module load r` is shadowing conda. Fix:\n",
      "  module unload r\n",
      "  conda deactivate && conda activate cicero\n",
      "  Rscript scripts/setup_r.R"
    )
  }
}

ensure_pkg <- function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    return(invisible(TRUE))
  }
  message("[setup_r] Installing missing dependency: ", pkg)
  install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(
      "Still missing package: ", pkg, "\n",
      "Run: conda activate cicero && conda install -c conda-forge r-",
      tolower(pkg), "\n",
      "Then: Rscript scripts/setup_r.R"
    )
  }
  invisible(TRUE)
}

use_conda_r_lib()
assert_lib_writable()
message("[setup_r] Using R library: ", .libPaths()[1L])

# Everything cicero needs besides monocle3 (installed via conda + GitHub).
for (pkg in c(
  "monocle3", "Gviz", "glasso", "VGAM",
  "VariantAnnotation", "biovizBase", "sf", "BPCells", "batchelor", "terra"
)) {
  ensure_pkg(pkg)
}
message("[setup_r] All cicero dependencies present.")

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", repos = "https://cloud.r-project.org", quiet = TRUE)
}

message("[setup_r] Installing cicero from GitHub (dependencies=FALSE) ...")
remotes::install_github(
  "cole-trapnell-lab/cicero-release",
  ref = "monocle3",
  upgrade = "never",
  dependencies = FALSE,
  force = TRUE
)

suppressPackageStartupMessages({
  library(monocle3)
  library(cicero)
})

cat("[setup_r] monocle3 + cicero are importable.\n")
