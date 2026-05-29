## hessian_normalize.R
## Canonicalize a gradient AST before the recursive Hessian walker reads it.
##
## The grad engine fast-dispatches reduction sub-gradients, so the AST that
## `.grad_expr` returns can contain vDSP fast kernels (`fast_vec_add`,
## `fast_vv_exp`, `fast_scalar_mul`, `fast_sum_sq`, ...) rather than canonical
## arithmetic. The Hessian walker's `.hess_shape` only knows standard ops, so a
## `fast_*` head classifies as "unknown" and the walker raises
## `hessian_not_supported` before any shape rule applies.
##
## `.normalize_fast_kernels` rewrites each fast kernel back to its canonical
## equivalent (value-preserving — the kernels are drop-in SIMD replacements that
## evaluate identically on finite input) and, in the same recursive pass, folds
## additive / multiplicative identities the grad engine leaves behind (e.g. the
## `0 * sum(w^2)` term a quotient rule's `a'` produces when `a' = 0`), so the
## five downstream shape rules see a uniform canonical surface and need no
## `fast_*` awareness. A no-op on ASTs that already contain only standard ops.
##
## Argument order matches the grad-engine emit sites:
##   fast_scalar_div(s, v) = s / v       fast_vec_div(numerator, denominator)
##   fast_vec_sub(a, b)    = a - b       fast_vec_smadd(s, v, w) = s * v + w
##   fast_sum_sq(v)        = sum(v^2)    fast_scalar_mul(s, v)   = s * v

.normalize_fast_kernels <- function(expr) {
  expr <- .strip_paren(expr)
  if (!is.call(expr) || !is.symbol(expr[[1L]])) return(expr)
  a <- lapply(as.list(expr)[-1L], .normalize_fast_kernels)
  switch(as.character(expr[[1L]]),
    "fast_vec_add"    = bquote(.(a[[1L]]) + .(a[[2L]])),
    "fast_vec_sub"    = bquote(.(a[[1L]]) - .(a[[2L]])),
    "fast_vec_mul"    = bquote(.(a[[1L]]) * .(a[[2L]])),
    "fast_vec_div"    = bquote(.(a[[1L]]) / .(a[[2L]])),
    "fast_scalar_mul" = bquote(.(a[[1L]]) * .(a[[2L]])),
    "fast_scalar_div" = bquote(.(a[[1L]]) / .(a[[2L]])),
    "fast_vec_smadd"  = bquote(.(a[[1L]]) * .(a[[2L]]) + .(a[[3L]])),
    "fast_sum_sq"     = bquote(sum(.(a[[1L]])^2)),
    "fast_vv_exp"     = bquote(exp(.(a[[1L]]))),
    "fast_vv_sin"     = bquote(sin(.(a[[1L]]))),
    "fast_vv_cos"     = bquote(cos(.(a[[1L]]))),
    "fast_vv_log"     = bquote(log(.(a[[1L]]))),
    "fast_vv_sqrt"    = bquote(sqrt(.(a[[1L]]))),
    "fast_vv_tanh"    = bquote(tanh(.(a[[1L]]))),
    # Strip shape-coercion wrappers the grad engine inserts purely to avoid
    # the array-vector recycling deprecation (e.g. as.numeric(crossprod(v,v))
    # in the L_0 `/` rule). They are transparent for differentiation, so the
    # walker should classify the inner expression (sum/crossprod -> scalar).
    "as.numeric" = a[[1L]],
    "as.vector"  = a[[1L]],
    # Fold additive / multiplicative identities (0 * x, 0 - x, ...) so no
    # degenerate scalar term reaches the walker. Identity-preserving.
    "+" = if (length(a) == 2L) .smart_add(a[[1L]], a[[2L]]) else a[[1L]],
    "-" = if (length(a) == 2L) .smart_sub(a[[1L]], a[[2L]]) else .smart_neg(a[[1L]]),
    "*" = if (length(a) == 2L) .smart_mul(a[[1L]], a[[2L]]) else as.call(c(expr[[1L]], a)),
    as.call(c(expr[[1L]], a))
  )
}
