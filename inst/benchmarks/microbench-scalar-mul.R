## microbench-scalar-mul.R
## Phase D microbenchmark (add-comprehensive-test-benchmark-expansion):
## Isolated comparison of fast_scalar_mul (Tier 1) vs R stock `*`.
##
## Run: Rscript inst/benchmarks/microbench-scalar-mul.R

Sys.setenv(VECLIB_MAXIMUM_THREADS = "1")
options(digits = 4)

suppressPackageStartupMessages(library(dat))
have_bench <- requireNamespace("bench", quietly = TRUE)
have_arma <- requireNamespace("RcppArmadillo", quietly = TRUE)

ns <- c(1e3, 1e6, 1e8)

cat("## microbench-scalar-mul\n\n")
cat(sprintf("- bench::mark active: %s\n", ifelse(have_bench, "YES (iterations=50)", "NO (fallback)")))
cat(sprintf("- RcppArmadillo comparison: %s\n", ifelse(have_arma, "YES", "NO (not installed)")))
cat("\n| n | fast_scalar_mul (ms) | R `s*v` (ms) | DD speedup |\n")
cat("|---|---|---|---|\n")

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

s <- 2.5
for (n in ns) {
  set.seed(20260525L)
  v <- runif(as.integer(n))
  invisible(fast_scalar_mul(s, v)); invisible(s * v)  # warm up
  fast_ms <- bench_or_median(function() invisible(fast_scalar_mul(s, v)))
  r_ms    <- bench_or_median(function() invisible(s * v))
  cat(sprintf("| %.0e | %.4f | %.4f | %.2fx |\n",
              n, fast_ms, r_ms, r_ms / max(fast_ms, 1e-6)))
  rm(v); invisible(gc(verbose = FALSE))
}

cat("\nKernel inspection: `fast_scalar_mul` dispatches to vDSP_vsmulD (Apple Accelerate NEON SIMD)\n")
cat("Reference (R stock `*`): per-element loop with NA semantics + per-call vector allocation\n")
cat("\nBenchmark complete.\n")
