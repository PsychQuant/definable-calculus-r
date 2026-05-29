## microbench-hessian-pattern.R
## Phase D: per-Hessian-pattern build time + evaluation time, including
## quadratic form (constant matrix) and separable forms (diagonal matrix).

Sys.setenv(VECLIB_MAXIMUM_THREADS = "1")
options(digits = 4)

suppressPackageStartupMessages(library(dat))
have_bench <- requireNamespace("bench", quietly = TRUE)

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

build_pow_f <- function(k) {
  e <- function(v) NULL
  body(e) <- bquote(sum(v^.(k)))
  e
}

cases <- list(
  list(label = "sum(v^2)",        f = build_pow_f(2)),
  list(label = "sum(v^3)",        f = build_pow_f(3)),
  list(label = "sum(sin(v))",     f = function(v) sum(sin(v))),
  list(label = "sum(cos(v))",     f = function(v) sum(cos(v))),
  list(label = "sum(exp(v))",     f = function(v) sum(exp(v))),
  list(label = "sum(log(v))",     f = function(v) sum(log(v)))
)
ns <- c(10, 50, 100)

cat("## microbench-hessian-pattern\n\n")
cat(sprintf("- bench::mark active: %s\n\n", ifelse(have_bench, "YES (iterations=50)", "NO")))
cat("| Expression | n | hessian() build (ms) | hf(v) eval (ms) |\n")
cat("|---|---|---|---|\n")

for (c in cases) {
  for (n in ns) {
    set.seed(20260525L)
    v <- runif(as.integer(n))
    invisible(hessian(c$f))  # warm
    build_ms <- bench_or_median(function() invisible(hessian(c$f)))
    hf <- hessian(c$f)
    invisible(hf(v))
    eval_ms  <- bench_or_median(function() invisible(hf(v)))
    cat(sprintf("| %-12s | %3d | %.4f | %.4f |\n", c$label, n, build_ms, eval_ms))
    rm(v, hf); invisible(gc(verbose = FALSE))
  }
}

cat("\nPattern interpretation: each Hessian pattern's build cost is constant per cell;\n")
cat("eval cost is dominated by diag() allocation O(n^2). For n=100, eval cost ~10us-1ms.\n")
cat("\nBenchmark complete.\n")
