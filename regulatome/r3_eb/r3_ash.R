# ashr per unit: reads <dir>/in/<unit>.tsv (eid, b, s), writes <dir>/out/<unit>.tsv (eid, beta, psd, lfsr) and <dir>/out/<unit>.g.tsv (pi0, tau).
# Normal mixture, point mass at 0, unimodal at 0 (ashr defaults: mode 0, prior "nullbiased"). usage: Rscript r3_ash.R <dir> [cores]
suppressMessages(library(ashr)); a <- commandArgs(TRUE); d <- a[1]; cores <- if (length(a) > 1) as.integer(a[2]) else 1L
dir.create(file.path(d, "out"), showWarnings = FALSE)
fs <- list.files(file.path(d, "in"), pattern = "\\.tsv$")
run <- function(f) {
  x <- read.delim(file.path(d, "in", f)); u <- sub("\\.tsv$", "", f)
  if (nrow(x) < 10) { write.table(data.frame(eid = x$eid, beta = 0, psd = 0, lfsr = 1), file.path(d, "out", f), sep = "\t", quote = FALSE, row.names = FALSE)
    write.table(data.frame(unit = u, pi0 = 1, tau = 0, n = nrow(x)), file.path(d, "out", paste0(u, ".g.tsv")), sep = "\t", quote = FALSE, row.names = FALSE); return(u) }
  r <- ash(x$b, x$s, mixcompdist = "normal", pointmass = TRUE, mode = 0, prior = "nullbiased")
  g <- get_fitted_g(r); pi0 <- sum(g$pi[g$sd == 0]); tau <- sqrt(sum(g$pi * g$sd^2))
  write.table(data.frame(eid = x$eid, beta = get_pm(r), psd = get_psd(r), lfsr = get_lfsr(r)), file.path(d, "out", f), sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(data.frame(unit = u, pi0 = pi0, tau = tau, n = nrow(x)), file.path(d, "out", paste0(u, ".g.tsv")), sep = "\t", quote = FALSE, row.names = FALSE); u }
invisible(parallel::mclapply(fs, run, mc.cores = cores)); cat("ash done", length(fs), "\n")
