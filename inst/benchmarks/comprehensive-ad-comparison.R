## comprehensive-ad-comparison.R
## Cross-framework benchmark: DD vs PyTorch (.backward + torch.func.grad)
## vs numDeriv across 5 gradient expressions and 4 hessian expressions.
##
## Run: Rscript inst/benchmarks/comprehensive-ad-comparison.R

Sys.setenv(VECLIB_MAXIMUM_THREADS = "1")
options(digits = 4)

loaded <- tryCatch({
  suppressPackageStartupMessages(devtools::load_all(".", quiet = TRUE))
  "devtools::load_all"
}, error = function(e) {
  suppressPackageStartupMessages(library(dat))
  "library(dat)"
})

have_numDeriv <- requireNamespace("numDeriv", quietly = TRUE)
# numDeriv's hessian/grad are UseMethod generics; dat shadows the function
# method via S3 dispatch. Use explicit numDeriv::hessian.default bindings to
# skip dispatch.
nd_grad <- if (have_numDeriv) getFromNamespace("grad.default", "numDeriv") else function(...) NA_real_
nd_hess <- if (have_numDeriv) getFromNamespace("hessian.default", "numDeriv") else function(...) NA_real_

have_jsonlite <- requireNamespace("jsonlite", quietly = TRUE)

pyscript <- file.path("inst", "benchmarks", "comprehensive-ad-comparison.py")

have_pytorch <- tryCatch({
  res <- system2("python3", c("-c", shQuote("import torch; print(torch.__version__)")),
                 stdout = TRUE, stderr = NULL)
  length(res) > 0
}, error = function(e) FALSE, warning = function(w) FALSE)

torch_version <- if (have_pytorch) {
  tryCatch(system2("python3", c("-c", shQuote("import torch; print(torch.__version__)")),
                   stdout = TRUE)[1L], error = function(e) "unknown")
} else NA_character_

build_pow_grad_f <- function(k) {
  e <- function(v) NULL
  body(e) <- bquote(sum(v^.(k)))
  e
}

dd_grad_exprs <- list(
  # Tier 1 strict canonical
  sum_v2          = list(label = "sum(v^2)",         f = function(v) sum(v^2)),
  # Tier 2a outer-negation
  neg_sum_v2      = list(label = "-sum(v^2)",        f = function(v) -sum(v^2)),
  # Tier 1 alt syntax → same canonical
  crossprod_v_v   = list(label = "crossprod(v,v)",   f = function(v) crossprod(v, v)),
  # Tier 2d composite (cos outer factor)
  sin_sum_v2      = list(label = "sin(sum(v^2))",    f = function(v) sin(sum(v^2))),
  # Tier 2c bare elementwise
  sum_sin_v       = list(label = "sum(sin(v))",      f = function(v) sum(sin(v))),
  sum_exp_v       = list(label = "sum(exp(v))",      f = function(v) sum(exp(v))),
  # Tier 2c additional vForce kernels (cos/log/tanh/sqrt)
  sum_cos_v       = list(label = "sum(cos(v))",      f = function(v) sum(cos(v))),
  sum_log_v       = list(label = "sum(log(v))",      f = function(v) sum(log(v))),
  sum_tanh_v      = list(label = "sum(tanh(v))",     f = function(v) sum(tanh(v))),
  sum_sqrt_v      = list(label = "sum(sqrt(v))",     f = function(v) sum(sqrt(v))),
  # Tier 2c scaled elementwise
  scaled_sin      = list(label = "2*sum(sin(v))",    f = function(v) 2 * sum(sin(v))),
  neg_sum_cos     = list(label = "-sum(cos(v))",     f = function(v) -sum(cos(v))),
  # Polynomial gradients across degrees 3-5
  sum_v3          = list(label = "sum(v^3)",         f = build_pow_grad_f(3)),
  sum_v4          = list(label = "sum(v^4)",         f = build_pow_grad_f(4)),
  sum_v5          = list(label = "sum(v^5)",         f = build_pow_grad_f(5)),
  # Tier 2d composite additional patterns
  exp_sum_v2      = list(label = "exp(sum(v^2))",    f = function(v) exp(sum(v^2))),
  cos_crossprod   = list(label = "cos(crossprod(v,v))", f = function(v) cos(crossprod(v, v))),
  tanh_sum_sin    = list(label = "tanh(sum(sin(v)))",f = function(v) tanh(sum(sin(v))))
)

build_pow_f <- function(k) {
  e <- function(v) NULL
  body(e) <- bquote(sum(v^.(k)))
  e
}

# Phase 1: Tier-3-only expressions (recursive .grad_inner walker path).
# These have correctness tests but were never speed-benched. Each entry pairs
# the function with an analytic-gradient lambda for correctness pre-check.
tier3_expressions <- list(
  sum_v_sin_v       = list(label = "sum(v*sin(v))",       f = function(v) sum(v * sin(v)),
                           ag = function(v) sin(v) + v * cos(v)),
  sum_sin_cos       = list(label = "sum(sin(v)*cos(v))",  f = function(v) sum(sin(v) * cos(v)),
                           ag = function(v) cos(v)^2 - sin(v)^2),
  sum_v_over_sin    = list(label = "sum(v/sin(v))",       f = function(v) sum(v / sin(v)),
                           ag = function(v) (sin(v) - v * cos(v)) / sin(v)^2),
  sum_sin_v_plus_1  = list(label = "sum(sin(v+1))",       f = function(v) sum(sin(v + 1)),
                           ag = function(v) cos(v + 1)),
  sum_sin_2v        = list(label = "sum(sin(2*v))",       f = function(v) sum(sin(2 * v)),
                           ag = function(v) 2 * cos(2 * v)),
  sum_v2_plus_sin   = list(label = "sum(v^2+sin(v))",     f = function(v) sum(v^2 + sin(v)),
                           ag = function(v) 2 * v + cos(v)),
  crossprod_v_sin   = list(label = "crossprod(v,sin(v))", f = function(v) crossprod(v, sin(v)),
                           ag = function(v) sin(v) + v * cos(v)),
  exp_log_sum_v2    = list(label = "exp(log(sum(v^2)))",  f = function(v) exp(log(sum(v^2))),
                           ag = function(v) 2 * v)
)

dd_hess_exprs <- list(
  sum_v2_h   = list(label = "sum(v^2)",  f = build_pow_f(2)),
  sum_v3_h   = list(label = "sum(v^3)",  f = build_pow_f(3)),
  sum_sin_h  = list(label = "sum(sin(v))", f = function(v) sum(sin(v))),
  sum_exp_h  = list(label = "sum(exp(v))", f = function(v) sum(exp(v)))
)

grad_ns <- c(1e3, 1e6, 1e8)
hess_ns <- c(10, 50, 100)
REPS <- 5L

have_bench <- requireNamespace("bench", quietly = TRUE)

# Phase C: bench::mark migration.
# Returns list with median_ms, iqr_ms, cv_pct, gc_total, bytes when bench is
# available; falls back to median-of-5 with NA_real_ for the new columns
# when bench is missing (backward-compat for envs without bench installed).
#
# C.5: explicit per-iteration seed reset is implemented by snapshotting `v`
# outside the bench::mark call. bench::mark's `iterations` are over the SAME
# already-allocated `v`, eliminating input-distribution variance.
.bench_or_replicate <- function(label, expr_thunk, reps = REPS) {
  if (have_bench) {
    res <- tryCatch(
      bench::mark(expr_thunk(), iterations = 50L, check = FALSE,
                  time_unit = "ms", filter_gc = FALSE),
      error = function(e) NULL
    )
    if (is.null(res)) {
      return(list(median_ms = NA_real_, iqr_ms = NA_real_, cv_pct = NA_real_,
                  gc_total = NA_integer_, bytes = NA_real_))
    }
    # bench::mark returns difftime in seconds for time columns even when
    # time_unit is set — convert to milliseconds explicitly.
    times_sec <- as.numeric(res$time[[1L]])
    times_ms  <- times_sec * 1000
    q  <- stats::quantile(times_ms, c(0.25, 0.5, 0.75), na.rm = TRUE)
    gc_df <- res$gc[[1L]]
    gc_n  <- if (is.data.frame(gc_df)) sum(gc_df$level0 + gc_df$level1 + gc_df$level2, na.rm = TRUE) else NA_integer_
    bytes <- if (length(res$mem_alloc) > 0L) as.numeric(res$mem_alloc[[1L]]) else NA_real_
    list(median_ms = unname(q[2L]),
         iqr_ms    = unname(q[3L] - q[1L]),
         cv_pct    = 100 * stats::sd(times_ms, na.rm = TRUE) / mean(times_ms, na.rm = TRUE),
         gc_total  = gc_n,
         bytes     = bytes)
  } else {
    ts <- replicate(reps, { t0 <- proc.time()[["elapsed"]]; expr_thunk(); proc.time()[["elapsed"]] - t0 }) * 1000
    list(median_ms = median(ts), iqr_ms = NA_real_, cv_pct = NA_real_,
         gc_total = NA_integer_, bytes = NA_real_)
  }
}

time_dd_grad <- function(f, n) {
  set.seed(20260525L)  # C.5: seed reset per cell for reproducibility
  gf <- grad(f); v <- runif(n)
  invisible(gf(v))  # C.4: explicit warm-up
  .bench_or_replicate("dd_grad", function() invisible(gf(v)))
}
time_dd_hess <- function(f, n) {
  set.seed(20260525L)
  hf <- hessian(f); v <- runif(n)
  invisible(hf(v))
  .bench_or_replicate("dd_hess", function() invisible(hf(v)))
}
time_numDeriv_grad <- function(f, n) {
  if (!have_numDeriv) return(list(median_ms = NA_real_, iqr_ms = NA_real_, cv_pct = NA_real_, gc_total = NA_integer_, bytes = NA_real_))
  set.seed(20260525L)
  v <- runif(n)
  .bench_or_replicate("nd_grad", function() invisible(nd_grad(f, v)))
}
time_numDeriv_hess <- function(f, n) {
  if (!have_numDeriv) return(list(median_ms = NA_real_, iqr_ms = NA_real_, cv_pct = NA_real_, gc_total = NA_integer_, bytes = NA_real_))
  set.seed(20260525L)
  v <- runif(n)
  .bench_or_replicate("nd_hess", function() invisible(nd_hess(f, v)))
}

parse_json_wall <- function(line) {
  if (have_jsonlite) {
    p <- tryCatch(jsonlite::fromJSON(line), error = function(e) list(wall_ms = NA_real_))
    if (!is.null(p$error)) return(NA_real_)
    return(as.double(p$wall_ms))
  }
  m <- regmatches(line, regexec('"wall_ms"\\s*:\\s*([0-9.eE+-]+)', line))[[1L]]
  if (length(m) >= 2L) as.double(m[[2L]]) else NA_real_
}

time_torch <- function(expr_name, n, backend) {
  if (!have_pytorch) return(NA_real_)
  out <- tryCatch(
    system2("python3", c(pyscript, "--expression", expr_name,
                         "--n", format(as.integer(n), scientific = FALSE), "--backend", backend),
            stdout = TRUE, stderr = TRUE),
    error = function(e) NA
  )
  if (length(out) == 0L || any(is.na(out))) return(NA_real_)
  parse_json_wall(out[length(out)])
}

cat("\n# Comprehensive AD comparison\n\n")
cat(sprintf("- Loaded via: %s\n", loaded))
cat(sprintf("- Platform: %s, %s\n", Sys.info()[["sysname"]], R.version$arch))
cat(sprintf("- R version: %s\n", R.version$version.string))
cat(sprintf("- PyTorch: %s\n", ifelse(have_pytorch, torch_version, "NOT AVAILABLE")))
cat(sprintf("- numDeriv: %s\n", ifelse(have_numDeriv, "available", "NOT AVAILABLE")))
cat(sprintf("- VECLIB_MAXIMUM_THREADS: %s\n", Sys.getenv("VECLIB_MAXIMUM_THREADS")))
cat(sprintf("- Reps per cell: %d (median reported)\n\n", REPS))

# Phase C: bench::mark migration enriches DD columns with median/IQR/CV/GC/bytes.
# PyTorch column remains median-only (Python sidecar limitation).
fmt_dd <- function(s) {
  if (is.na(s$median_ms)) return("N/A")
  paste0(sprintf("%.3f", s$median_ms),
         " (IQR ", sprintf("%.3f", s$iqr_ms),
         ", CV ", sprintf("%.1f%%", s$cv_pct),
         ", GC ", s$gc_total,
         ", ", sprintf("%.1fMB", s$bytes / 1024^2), ")")
}
fmt_nd <- function(s, available = TRUE) {
  if (!available || is.na(s$median_ms)) return("N/A (skipped)")
  paste0(sprintf("%.3f", s$median_ms),
         " (IQR ", sprintf("%.3f", s$iqr_ms),
         ", CV ", sprintf("%.1f%%", s$cv_pct), ")")
}

cat("## Gradient comparison\n\n")
cat(sprintf("Statistical detail (DD column): median (IQR, CV%%, GC count, memory MB). bench::mark iterations=50 active: %s\n\n",
            ifelse(have_bench, "YES", "NO (fallback: median over 5 reps)")))
cat("| Expression | n | DD | PyTorch .backward() (ms) | PyTorch torch.func.grad (ms) | numDeriv::grad |\n")
cat("|---|---|---|---|---|---|\n")

for (key in names(dd_grad_exprs)) {
  e <- dd_grad_exprs[[key]]
  for (n in grad_ns) {
    dd_s     <- time_dd_grad(e$f, as.integer(n))
    tback_ms <- time_torch(key, n, "torch_backward")
    tfunc_ms <- time_torch(key, n, "torch_func_grad")
    use_nd   <- n <= 1e4
    nd_s     <- if (use_nd) time_numDeriv_grad(e$f, as.integer(n))
                else list(median_ms = NA_real_, iqr_ms = NA_real_, cv_pct = NA_real_, gc_total = NA_integer_, bytes = NA_real_)
    cat(sprintf("| %-18s | %.0e | %s | %s | %s | %s |\n",
                e$label, n, fmt_dd(dd_s),
                if (is.na(tback_ms)) "N/A" else sprintf("%.3f", tback_ms),
                if (is.na(tfunc_ms)) "N/A" else sprintf("%.3f", tfunc_ms),
                fmt_nd(nd_s, use_nd)))
    invisible(gc(verbose = FALSE))
  }
}

cat("\n## Tier 3 expression benchmark\n\n")
cat("Tier-3-only expressions go through the recursive `.grad_inner` walker (no fast-path match).\n")
cat("Each row prefixed with correctness check vs analytic gradient at n=1000.\n\n")
cat("| Expression | n | Correctness | DD | PyTorch .backward() (ms) | PyTorch torch.func.grad (ms) | numDeriv::grad |\n")
cat("|---|---|---|---|---|---|---|\n")

check_correctness <- function(f, ag) {
  set.seed(20260525L)
  v <- runif(1000L)
  gf <- tryCatch(grad(f), error = function(e) NULL)
  if (is.null(gf)) return(list(ok = FALSE, msg = "grad() failed"))
  dd_val <- tryCatch(gf(v), error = function(e) NULL)
  if (is.null(dd_val)) return(list(ok = FALSE, msg = "gf(v) failed"))
  ag_val <- ag(v)
  cmp <- isTRUE(all.equal(as.numeric(dd_val), as.numeric(ag_val), tolerance = 1e-10))
  if (cmp) list(ok = TRUE, msg = "OK")
  else list(ok = FALSE, msg = sprintf("max diff %.3e", max(abs(dd_val - ag_val))))
}

for (key in names(tier3_expressions)) {
  e <- tier3_expressions[[key]]
  corr <- check_correctness(e$f, e$ag)
  corr_str <- if (corr$ok) "✓" else paste0("✗ ", corr$msg)
  for (n in grad_ns) {
    dd_s     <- time_dd_grad(e$f, as.integer(n))
    tback_ms <- time_torch(key, n, "torch_backward")
    tfunc_ms <- time_torch(key, n, "torch_func_grad")
    use_nd   <- n <= 1e4
    nd_s     <- if (use_nd) time_numDeriv_grad(e$f, as.integer(n))
                else list(median_ms = NA_real_, iqr_ms = NA_real_, cv_pct = NA_real_, gc_total = NA_integer_, bytes = NA_real_)
    cat(sprintf("| %-22s | %.0e | %s | %s | %s | %s | %s |\n",
                e$label, n, corr_str, fmt_dd(dd_s),
                if (is.na(tback_ms)) "N/A" else sprintf("%.3f", tback_ms),
                if (is.na(tfunc_ms)) "N/A" else sprintf("%.3f", tfunc_ms),
                fmt_nd(nd_s, use_nd)))
    invisible(gc(verbose = FALSE))
  }
}

cat("\n## N-size scaling curves\n\n")
cat("Four representatives covering distinct dispatch paths. DD-only timings (cross-framework comparison stays on 3-size grid above).\n\n")

# Phase 2: scaling sweep limited to 4 representatives per design Decision 2.
scaling_representatives <- list(
  sum_v2     = list(label = "sum(v^2)",     f = function(v) sum(v^2),
                    dispatch = "Tier 1 fast-path (fast_scalar_mul)"),
  sum_sin_v  = list(label = "sum(sin(v))",  f = function(v) sum(sin(v)),
                    dispatch = "Tier 2c vForce (fast_vv_cos via grad)"),
  sin_sum_v2 = list(label = "sin(sum(v^2))", f = function(v) sin(sum(v^2)),
                    dispatch = "Tier 2d composite (fast_sum_sq + outer scalar)"),
  sum_v3     = list(label = "sum(v^3)",     f = function(v) sum(v^3),
                    dispatch = "Tier 3b scalar-pow (fast_vec_mul chain)")
)
scaling_ns <- c(1e3, 1e4, 1e5, 1e6, 1e7, 1e8)

for (key in names(scaling_representatives)) {
  e <- scaling_representatives[[key]]
  cat(sprintf("### %s — %s\n\n", e$label, e$dispatch))
  cat("| n | DD ms | Throughput (Mops/sec) |\n|---|---|---|\n")
  for (n in scaling_ns) {
    dd_s <- time_dd_grad(e$f, as.integer(n))
    if (is.na(dd_s$median_ms) || dd_s$median_ms <= 0) {
      mops <- NA_real_
    } else {
      mops <- (n / (dd_s$median_ms / 1000)) / 1e6
    }
    cat(sprintf("| %.0e | %.3f | %.1f |\n",
                n, dd_s$median_ms,
                if (is.na(mops)) NA_real_ else mops))
    invisible(gc(verbose = FALSE))
  }
  cat("\n")
}

cat("\n## JAX framework comparison\n\n")

# Phase 3: JAX sidecar (jax.grad + jax.jit-warmed) — graceful skip if not installed.
jax_script <- file.path("inst", "benchmarks", "sidecar_jax_compare.py")
time_jax <- function(expr_name, n, backend) {
  out <- tryCatch(
    system2("python3", c(jax_script, "--expression", expr_name,
                         "--n", format(as.integer(n), scientific = FALSE),
                         "--backend", backend),
            stdout = TRUE, stderr = TRUE),
    error = function(e) structure("error", class = "try-error")
  )
  if (inherits(out, "try-error") || length(out) == 0L) return(list(wall = NA_real_, missing = TRUE))
  last <- out[length(out)]
  if (grepl("jax not installed", last, fixed = TRUE)) return(list(wall = NA_real_, missing = TRUE))
  list(wall = parse_json_wall(last), missing = FALSE)
}

# Probe once to decide whether to emit the section or the skip message.
probe <- time_jax("sum_v2", 1000L, "jax_grad")
if (isTRUE(probe$missing)) {
  cat("[JAX skipped: jax not installed]\n\n")
} else {
  cat("Four representatives × 3 n sizes. `jax_jit` column includes 3 warm-up calls (amortizes JIT trace + XLA compile).\n\n")
  cat("| Expression | n | DD ms | jax.grad ms | jax.jit warmed ms |\n|---|---|---|---|---|\n")
  jax_representatives <- list(
    sum_v2     = list(label = "sum(v^2)",      f = function(v) sum(v^2)),
    sum_sin_v  = list(label = "sum(sin(v))",   f = function(v) sum(sin(v))),
    sin_sum_v2 = list(label = "sin(sum(v^2))", f = function(v) sin(sum(v^2))),
    sum_v3     = list(label = "sum(v^3)",      f = function(v) sum(v^3))
  )
  for (key in names(jax_representatives)) {
    e <- jax_representatives[[key]]
    for (n in grad_ns) {
      dd_s    <- time_dd_grad(e$f, as.integer(n))
      jg      <- time_jax(key, n, "jax_grad")
      jj      <- time_jax(key, n, "jax_jit")
      cat(sprintf("| %-18s | %.0e | %.3f | %s | %s |\n",
                  e$label, n,
                  dd_s$median_ms,
                  if (is.na(jg$wall)) "N/A" else sprintf("%.3f", jg$wall),
                  if (is.na(jj$wall)) "N/A" else sprintf("%.3f", jj$wall)))
      invisible(gc(verbose = FALSE))
    }
  }
}

cat("\n## Real-world ML/stat patterns\n\n")

# Phase 4: real-world patterns with closure-thesis classification.
source(file.path("inst", "benchmarks", "real_world_patterns.R"))

classify_pattern <- function(key, e, n) {
  # Use [[ to avoid R's $ partial matching (e$f_inline would otherwise match
  # f_inline_builder for nn_forward, returning the builder fn itself).
  if (!is.null(e[["f_inline"]])) {
    f_inline <- e[["f_inline"]]
  } else if (!is.null(e[["f_inline_builder"]])) {
    f_inline <- e[["f_inline_builder"]](n)
  } else {
    return(list(status = "OUT-OF-SCOPE", reason = "no f_inline", f_inline = NULL))
  }
  res <- tryCatch({
    gf <- grad(f_inline)
    invisible(gf(runif(min(50L, as.integer(n)))))
    list(status = "WORKS", reason = "", f_inline = f_inline)
  },
  dat_unknown_generator = function(c) list(status = "OUT-OF-SCOPE",
                                           reason = sprintf("unknown generator: %s", conditionMessage(c)),
                                           f_inline = NULL),
  dat_not_definable = function(c) list(status = "OUT-OF-SCOPE",
                                       reason = conditionMessage(c),
                                       f_inline = NULL),
  error = function(c) list(status = "OUT-OF-SCOPE", reason = conditionMessage(c), f_inline = NULL))
  res
}

cat("| Pattern | Status | n | DD | PyTorch .backward() (ms) | numDeriv (ms) | Notes |\n")
cat("|---|---|---|---|---|---|---|\n")
for (key in names(real_world_patterns)) {
  e <- real_world_patterns[[key]]
  # Classify at small n to decide if benchable
  cls <- classify_pattern(key, e, 100L)
  if (cls$status == "OUT-OF-SCOPE") {
    cat(sprintf("| %-18s | OUT-OF-SCOPE | — | — | — | — | %s |\n",
                e$label, substr(cls$reason, 1, 80)))
    next
  }
  # Correctness verification at n=100
  v_corr <- runif(100L)
  ana <- real_world_analytic[[key]](v_corr)
  dd_corr <- cls$f_inline
  gf_corr <- grad(dd_corr)
  dd_out <- gf_corr(v_corr)
  ok <- isTRUE(all.equal(as.numeric(dd_out), as.numeric(ana), tolerance = 1e-10))
  if (!ok) {
    cat(sprintf("| %-18s | CORRECTNESS-FAIL | — | — | — | — | max diff %.2e |\n",
                e$label, max(abs(dd_out - ana))))
    next
  }
  # Benchmark across grad_ns. For nn_forward need fresh builder per n.
  for (n in grad_ns) {
    if (!is.null(e[["f_inline_builder"]])) {
      f_n <- e[["f_inline_builder"]](as.integer(n))
    } else {
      f_n <- cls[["f_inline"]]
    }
    dd_s <- time_dd_grad(f_n, as.integer(n))
    # PyTorch path: skip (no sidecar entry for these patterns; would require adding)
    use_nd <- n <= 1e4
    nd_s <- if (use_nd) time_numDeriv_grad(f_n, as.integer(n))
            else list(median_ms = NA_real_, iqr_ms = NA_real_, cv_pct = NA_real_, gc_total = NA_integer_, bytes = NA_real_)
    cat(sprintf("| %-18s | WORKS | %.0e | %s | N/A (sidecar gap) | %s | — |\n",
                e$label, n, fmt_dd(dd_s), fmt_nd(nd_s, use_nd)))
    invisible(gc(verbose = FALSE))
  }
}

cat("\n## Hessian comparison\n\n")
cat("| Expression | n | DD | numDeriv::hessian |\n")
cat("|---|---|---|---|\n")

for (key in names(dd_hess_exprs)) {
  e <- dd_hess_exprs[[key]]
  for (n in hess_ns) {
    dd_s <- tryCatch(time_dd_hess(e$f, as.integer(n)),
                     error = function(x) list(median_ms = NA_real_, iqr_ms = NA_real_, cv_pct = NA_real_, gc_total = NA_integer_, bytes = NA_real_))
    nd_s <- time_numDeriv_hess(e$f, as.integer(n))
    cat(sprintf("| %-12s | %4d | %s | %s |\n",
                e$label, n, fmt_dd(dd_s), fmt_nd(nd_s)))
    invisible(gc(verbose = FALSE))
  }
}

cat("\n---\nBenchmark complete.\n")
