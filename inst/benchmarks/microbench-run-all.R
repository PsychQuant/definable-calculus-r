## microbench-run-all.R
## Phase D driver: source each microbench script sequentially, aggregating
## output into a combined markdown stream for inclusion in
## docs/dd-kernel-microbenchmarks.md。

scripts <- c(
  "inst/benchmarks/microbench-scalar-mul.R",
  "inst/benchmarks/microbench-sum-sq.R",
  "inst/benchmarks/microbench-vv-elementwise.R",
  "inst/benchmarks/microbench-vec-arith.R",
  "inst/benchmarks/microbench-grad-dispatch.R",
  "inst/benchmarks/microbench-hessian-pattern.R"
)

cat("# DD per-kernel microbenchmarks\n\n")
cat(sprintf("Run on %s @ %s\n", Sys.info()[["sysname"]], format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
cat(sprintf("R version: %s\n", R.version$version.string))
cat(sprintf("Platform: %s\n", R.version$platform))
cat(sprintf("VECLIB_MAXIMUM_THREADS: %s\n\n", Sys.getenv("VECLIB_MAXIMUM_THREADS", "(unset)")))

for (script in scripts) {
  cat(sprintf("\n---\n\n<!-- source: %s -->\n\n", script))
  src <- tryCatch(source(script, echo = FALSE, local = new.env()),
                  error = function(e) {
                    cat(sprintf("ERROR sourcing %s: %s\n", script, conditionMessage(e)))
                  })
}

cat("\n---\n\nAll microbenchmarks complete.\n")
