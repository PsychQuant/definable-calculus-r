## microbench-sum-sq.R
## Phase D: isolated comparison of fast_sum_sq (Tier 2d kernel) vs alternatives.

Sys.setenv(VECLIB_MAXIMUM_THREADS = "1")
options(digits = 4)

suppressPackageStartupMessages(library(dat))
have_bench <- requireNamespace("bench", quietly = TRUE)

ns <- c(1e3, 1e6, 1e8)

bench_or_median <- function(thunk) {
  if (have_bench) {
    res <- bench::mark(thunk(), iterations = 50L, check = FALSE,
                       time_unit = "ms", filter_gc = FALSE)
    median(as.numeric(res$time[[1L]]) * 1000)
  } else {
    ts <- replicate(5, { t0 <- proc.time()[["elapsed"]]; thunk(); proc.time()[["elapsed"]] - t0 }) * 1000
    median(ts)
  }
}

cat("## microbench-sum-sq\n\n")
cat(sprintf("- bench::mark active: %s\n", ifelse(have_bench, "YES (iterations=50)", "NO")))
cat("\n| n | fast_sum_sq (ms) | R `sum(v^2)` (ms) | R `crossprod(v,v)[1]` (ms) | DD speedup vs sum(v^2) |\n")
cat("|---|---|---|---|---|\n")

for (n in ns) {
  set.seed(20260525L)
  v <- runif(as.integer(n))
  invisible(fast_sum_sq(v)); invisible(sum(v^2)); invisible(crossprod(v, v)[1L])  # warm up
  fast_ms <- bench_or_median(function() invisible(fast_sum_sq(v)))
  sumsq_ms <- bench_or_median(function() invisible(sum(v^2)))
  cp_ms <- bench_or_median(function() invisible(crossprod(v, v)[1L]))
  cat(sprintf("| %.0e | %.4f | %.4f | %.4f | %.2fx |\n",
              n, fast_ms, sumsq_ms, cp_ms, sumsq_ms / max(fast_ms, 1e-6)))
  rm(v); invisible(gc(verbose = FALSE))
}

cat("\nKernel inspection: `fast_sum_sq` dispatches to vDSP_svesqD (single-pass sum-of-squares)\n")
cat("R `sum(v^2)`: allocates intermediate v^2 vector (O(n) memory), then sums\n")
cat("R `crossprod(v,v)`: BLAS dot product (Accelerate-backed); equivalent precision\n")
cat("\nBenchmark complete.\n")
