# ashr per unit: reads <dir>/in/<unit>.tsv (eid, b, s), writes <dir>/out/<unit>.tsv (eid, beta, psd, lfsr) and <dir>/out/<unit>.g.tsv (pi0, tau, n, status).
# Normal mixture, point mass at 0, unimodal at 0 (ashr defaults: mode 0, prior "nullbiased"). Units with < 10 scored elements, or where ash
# errors, get beta 0, lfsr 1 and pi0 1, with status "too_few" or "ash_error: <message>". usage: Rscript r3_ash.R <dir> [cores]
suppressMessages(library(ashr)); a <- commandArgs(TRUE); d <- a[1]; cores <- if (length(a) > 1) as.integer(a[2]) else 1L
dir.create(file.path(d, "out"), showWarnings = FALSE)
fs <- list.files(file.path(d, "in"), pattern = "\\.tsv$")
put <- function(u, f, eid, beta, psd, lfsr, pi0, tau, status) {
  write.table(data.frame(eid = eid, beta = beta, psd = psd, lfsr = lfsr), file.path(d, "out", f), sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(data.frame(unit = u, pi0 = pi0, tau = tau, n = length(eid), status = status), file.path(d, "out", paste0(u, ".g.tsv")), sep = "\t", quote = FALSE, row.names = FALSE) }
run <- function(f) {
  x <- read.delim(file.path(d, "in", f), colClasses = c("integer", "numeric", "numeric")); u <- sub("\\.tsv$", "", f); n <- nrow(x)
  if (n < 10) { put(u, f, x$eid, rep(0, n), rep(0, n), rep(1, n), 1, 0, "too_few"); return(u) }
  r <- tryCatch(ash(x$b, x$s, mixcompdist = "normal", pointmass = TRUE, mode = 0, prior = "nullbiased"), error = function(e) e)
  if (inherits(r, "error")) { put(u, f, x$eid, rep(0, n), rep(0, n), rep(1, n), 1, 0, paste("ash_error:", gsub("[\t\n]", " ", conditionMessage(r)))); return(u) }
  g <- get_fitted_g(r); put(u, f, x$eid, get_pm(r), get_psd(r), get_lfsr(r), sum(g$pi[g$sd == 0]), sqrt(sum(g$pi * g$sd^2)), "ok"); u }
res <- parallel::mclapply(fs, run, mc.cores = cores, mc.preschedule = FALSE)
bad <- sum(vapply(res, function(z) inherits(z, "try-error"), logical(1))); cat("ash done", length(fs), "worker errors", bad, "\n"); if (bad > 0) quit(status = 1)
