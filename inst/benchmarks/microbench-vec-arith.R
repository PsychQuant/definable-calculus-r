## microbench-vec-arith.R
## Phase D: isolated comparison of Tier 2b vector-vector arithmetic kernels.

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

cat("## microbench-vec-arith\n\n")
cat(sprintf("- bench::mark active: %s\n\n", ifelse(have_bench, "YES (iterations=50)", "NO")))

# fast_vec_add vs R `+`
cat("### fast_vec_add vs R `v + w`\n\n")
cat("| n | fast_vec_add (ms) | R `v + w` (ms) | DD speedup |\n|---|---|---|---|\n")
for (n in ns) {
  set.seed(20260525L); v <- runif(as.integer(n)); w <- runif(as.integer(n))
  invisible(fast_vec_add(v, w)); invisible(v + w)
  f_ms <- bench_or_median(function() invisible(fast_vec_add(v, w)))
  r_ms <- bench_or_median(function() invisible(v + w))
  cat(sprintf("| %.0e | %.4f | %.4f | %.2fx |\n", n, f_ms, r_ms, r_ms / max(f_ms, 1e-6)))
  rm(v, w); invisible(gc(verbose = FALSE))
}

cat("\n### fast_vec_sub vs R `v - w`\n\n")
cat("| n | fast_vec_sub (ms) | R `v - w` (ms) | DD speedup |\n|---|---|---|---|\n")
for (n in ns) {
  set.seed(20260525L); v <- runif(as.integer(n)); w <- runif(as.integer(n))
  invisible(fast_vec_sub(v, w)); invisible(v - w)
  f_ms <- bench_or_median(function() invisible(fast_vec_sub(v, w)))
  r_ms <- bench_or_median(function() invisible(v - w))
  cat(sprintf("| %.0e | %.4f | %.4f | %.2fx |\n", n, f_ms, r_ms, r_ms / max(f_ms, 1e-6)))
  rm(v, w); invisible(gc(verbose = FALSE))
}

cat("\n### fast_vec_smadd vs R `s * v + w`\n\n")
cat("| n | fast_vec_smadd (ms) | R `s*v+w` (ms) | DD speedup |\n|---|---|---|---|\n")
for (n in ns) {
  set.seed(20260525L); v <- runif(as.integer(n)); w <- runif(as.integer(n)); s <- 2.5
  invisible(fast_vec_smadd(s, v, w)); invisible(s * v + w)
  f_ms <- bench_or_median(function() invisible(fast_vec_smadd(s, v, w)))
  r_ms <- bench_or_median(function() invisible(s * v + w))
  cat(sprintf("| %.0e | %.4f | %.4f | %.2fx |\n", n, f_ms, r_ms, r_ms / max(f_ms, 1e-6)))
  rm(v, w); invisible(gc(verbose = FALSE))
}

cat("\nBenchmark complete.\n")
