## grad-quadratic.R
## Reproducible benchmark for Tier 1 + Tier 2a fast paths.
##
## Tier 1 (add-vdsp-fast-path): strict pattern `<num> * <var>` → vDSP_vsmulD
## Tier 2a (add-vdsp-fast-path-tier2a-normalization): outer-negation
##         pattern `-(<num> * <var>)` normalizes to same kernel
##
## Run: Rscript inst/benchmarks/grad-quadratic.R
##
## Baseline numbers (DD v0.1 pure R `*`, recorded in docs/dd-thesis.md):
##   n=1e3 -> ~0.001 ms, n=1e6 -> ~1 ms, n=1e8 -> ~103 ms
## Tier 1+2a target: n=1e8 < 50 ms (~2x speedup minimum)

Sys.setenv(VECLIB_MAXIMUM_THREADS = "1")
options(digits = 4)

loaded <- tryCatch({
  suppressPackageStartupMessages(devtools::load_all(".", quiet = TRUE))
  "devtools::load_all"
}, error = function(e) {
  suppressPackageStartupMessages(library(dat))
  "library(dat)"
})

patterns <- list(
  list(label = "Tier 1 (sum(v^2))",   f = function(v) sum(v^2)),
  list(label = "Tier 2a (-sum(v^2))", f = function(v) -sum(v^2))
)

# Sanity: each pattern's body must dispatch to fast_scalar_mul on macOS.
for (p in patterns) {
  gf <- grad(p$f)
  b <- body(gf)
  stopifnot(is.call(b),
            identical(b[[1L]], as.name("fast_scalar_mul")))
}

ns <- c(1e3, 1e6, 1e8)
reps <- 5L
baseline_ms <- c(0.001, 1, 103)  # DD v0.1 baseline from docs/dd-thesis.md

cat("\n## Tier 1 + Tier 2a fast-path benchmark\n\n")
cat(sprintf("Loaded via: %s\n", loaded))
cat(sprintf("Platform: %s, %s\n", Sys.info()[["sysname"]], R.version$arch))
cat(sprintf("VECLIB_MAXIMUM_THREADS: %s\n", Sys.getenv("VECLIB_MAXIMUM_THREADS")))
cat(sprintf("Repetitions per (pattern, n): %d (median reported)\n\n", reps))

cat("| pattern | n | wall_ms | baseline_ms (generic R `*`) | speedup |\n")
cat("|---|---|---|---|---|\n")

set.seed(20260525L)
all_pass <- TRUE
for (p in patterns) {
  gf <- grad(p$f)
  for (i in seq_along(ns)) {
    n <- as.integer(ns[i])
    v <- runif(n)
    invisible(gf(v))  # warm up
    ts <- replicate(reps, {
      t0 <- proc.time()[["elapsed"]]
      invisible(gf(v))
      proc.time()[["elapsed"]] - t0
    })
    wall_ms <- median(ts) * 1000
    cat(sprintf("| %-22s | %.0e | %.4f | %.3f | %.2fx |\n",
                p$label, ns[i], wall_ms, baseline_ms[i],
                baseline_ms[i] / max(wall_ms, 1e-6)))
    if (n == 1e8 && wall_ms >= 50) all_pass <- FALSE
    rm(v); invisible(gc(verbose = FALSE))
  }
}

cat("\nFast-path threshold (n=1e8 < 50 ms for ALL patterns): ")
cat(ifelse(all_pass, "PASS", "FAIL"))
cat("\n")
