#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# check_env.R — fail fast if monocle3 / cicero are not installed.
# Run once after: conda env create -f environment.yml && Rscript scripts/setup_r.R
# -----------------------------------------------------------------------------

missing <- character(0)
for (pkg in c("monocle3", "cicero", "Matrix", "yaml", "optparse")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    missing <- c(missing, pkg)
}

if (length(missing)) {
  cat(
    "[check_env] Missing R package(s): ", paste(missing, collapse = ", "), "\n",
    "Create the env and install GitHub deps:\n",
    "  conda env create -f environment.yml\n",
    "  conda activate cicero\n",
    "  Rscript scripts/setup_r.R\n",
    sep = ""
  )
  quit(status = 1)
}

suppressPackageStartupMessages({
  library(monocle3)
  library(cicero)
})

cat("[check_env] monocle3 + cicero are available.\n")
