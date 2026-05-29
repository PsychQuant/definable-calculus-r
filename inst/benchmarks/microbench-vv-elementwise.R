## microbench-vv-elementwise.R
## Phase D: isolated comparison of 6 vForce kernels vs R stock.

Sys.setenv(VECLIB_MAXIMUM_THREADS = "1")
options(digits = 4)

suppressPackageStartupMessages(library(dat))
have_bench <- requireNamespace("bench", quietly = TRUE)

ns <- c(1e3, 1e6, 1e8)
funcs <- c("cos", "sin", "exp", "log", "tanh", "sqrt")

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

cat("## microbench-vv-elementwise\n\n")
cat(sprintf("- bench::mark active: %s\n\n", ifelse(have_bench, "YES (iterations=50)", "NO")))

for (fn in funcs) {
  vv_kernel <- get(paste0("fast_vv_", fn))
  r_stock   <- get(fn)
  cat(sprintf("### %s\n\n", fn))
  cat(sprintf("| n | fast_vv_%s (ms) | R `%s(v)` (ms) | DD speedup |\n", fn, fn))
  cat("|---|---|---|---|\n")
  for (n in ns) {
    set.seed(20260525L)
    v <- runif(as.integer(n)) + 0.1  # positive for log/sqrt
    invisible(vv_kernel(v)); invisible(r_stock(v))  # warm up
    vv_ms <- bench_or_median(function() invisible(vv_kernel(v)))
    r_ms  <- bench_or_median(function() invisible(r_stock(v)))
    cat(sprintf("| %.0e | %.4f | %.4f | %.2fx |\n",
                n, vv_ms, r_ms, r_ms / max(vv_ms, 1e-6)))
    rm(v); invisible(gc(verbose = FALSE))
  }
  cat("\n")
}

cat("Kernel inspection: each `fast_vv_<fn>` dispatches to corresponding vForce kernel\n")
cat("(vvcos, vvsin, vvexp, vvlog, vvtanh, vvsqrt) — hand-tuned NEON SIMD elementwise transcendentals\n")
cat("\nBenchmark complete.\n")
