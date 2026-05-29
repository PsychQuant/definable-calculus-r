## hessian_inner.R
## Recursive Hessian construction (add-hessian-recursive-walker), generalized
## to multiple vector variables (add-multi-variable-hessian).
##
## The Hessian of a scalar-valued f is the Jacobian of its gradient field:
## H = J(grad f). The grad engine produces grad f as a closed-form vector AST
## via `.grad_expr`; this file walks that gradient AST into an n x n (or, for a
## mixed block of a multi-variable function, n_a x n_b) Hessian matrix AST in a
## small closed matrix sublanguage (`diag`, `outer`, `matrix(0, ...)`, `+`,
## `-`, scalar `*`, `%*%`). It mirrors `.grad_inner`'s closure-preserving
## design one order higher.
##
## All three helpers take `all_vars` — the full set of vector-variable names.
## For single-variable callers `all_vars` defaults to the differentiation
## variable, so a symbol that is not that variable is a scalar constant. For
## multi-variable callers, a symbol naming ANOTHER vector variable is a vector
## constant (vector-grain, derivative zero w.r.t. the differentiation variable)
## rather than a scalar.
##
## `hessian.function` calls `.hessian_recursive` (single variable) or
## `.hessian_block` (one block of a multi-variable function) only as a
## fallback, after the fast-path pattern dispatcher returns no match.

# Elementwise scalar functions whose chain rule the walker knows.
.HESS_ELEM_FNS <- c("sin", "cos", "exp", "log", "tanh", "sqrt", "atan")

# Gradient ASTs are canonicalized by `.normalize_fast_kernels` (R/hessian_normalize.R)
# at the walker entry points before any shape classification.

# .hess_shape(expr, var, all_vars)
#
# Classify a (sub)expression as "scalar" / "vector_grain" / "vector_dense" /
# "unknown" with respect to differentiation variable `var`, knowing the full
# set of vector variables `all_vars`.
.hess_shape <- function(expr, var, all_vars = var) {
  expr <- .strip_paren(expr)

  if (is.numeric(expr) || is.logical(expr)) {
    return(if (length(expr) == 1L) "scalar" else "vector_grain")
  }
  if (is.symbol(expr)) {
    nm <- as.character(expr)
    if (nm == var) return("vector_grain")
    if (nm %in% all_vars) return("vector_grain")  # another vector variable (constant w.r.t. var)
    return("scalar")                              # genuine free scalar constant
  }
  if (!is.call(expr) || !is.symbol(expr[[1L]])) return("unknown")
  op <- as.character(expr[[1L]])

  if (op %in% c("sum", "crossprod")) return("scalar")
  if (op == "%*%") return("vector_dense")
  # rep(value, ...) with a var-free value is a constant vector.
  if (op == "rep") return(if (.contains_var(expr[[2L]], var)) "unknown" else "vector_grain")

  if (op %in% c("-", "+") && length(expr) == 2L) {
    return(.hess_shape(expr[[2L]], var, all_vars))
  }
  if (op %in% c("+", "-") && length(expr) == 3L) {
    a <- .hess_shape(expr[[2L]], var, all_vars)
    b <- .hess_shape(expr[[3L]], var, all_vars)
    if (a == "unknown" || b == "unknown") return("unknown")
    if (a == "scalar" && b == "scalar") return("scalar")
    if ("vector_dense" %in% c(a, b)) return("vector_dense")
    return("vector_grain")
  }
  if (op == "*" && length(expr) == 3L) {
    a <- .hess_shape(expr[[2L]], var, all_vars)
    b <- .hess_shape(expr[[3L]], var, all_vars)
    if (a == "unknown" || b == "unknown") return("unknown")
    if (a == "scalar" && b == "scalar") return("scalar")
    if (a == "scalar" || b == "scalar") {       # scalar * vector
      if (a == "scalar") { s_expr <- expr[[2L]]; w_shape <- b }
      else               { s_expr <- expr[[3L]]; w_shape <- a }
      if (.contains_var(s_expr, var)) return("vector_dense")
      return(w_shape)
    }
    if ("vector_dense" %in% c(a, b)) return("vector_dense")
    return("vector_grain")                      # vector * vector (Hadamard)
  }
  if (op == "^" && length(expr) == 3L) {
    base <- .hess_shape(expr[[2L]], var, all_vars)
    if (!.contains_var(expr[[3L]], var) && base %in% c("scalar", "vector_grain")) {
      return(base)
    }
    return("unknown")
  }
  # Scalar-denominator quotient a / b. A vector denominator is outside the
  # matrix sublanguage (-> "unknown"). A var-DEPENDENT scalar denominator makes
  # the Jacobian dense (the rank-one 1/b correction), mirroring the
  # scalar*vector rule; a var-free scalar denominator inherits the numerator's
  # shape (a pure diagonal scaling).
  if (op == "/" && length(expr) == 3L) {
    num <- .hess_shape(expr[[2L]], var, all_vars)
    if (num == "unknown") return("unknown")
    if (.hess_shape(expr[[3L]], var, all_vars) != "scalar") return("unknown")
    if (.contains_var(expr[[3L]], var)) return("vector_dense")
    return(num)
  }
  if (op %in% .HESS_ELEM_FNS && length(expr) == 2L) {
    return(.hess_shape(expr[[2L]], var, all_vars))
  }

  "unknown"
}

# .hess_diag(expr, var, all_vars)
#
# For a vector-grain (or scalar) expression, return the AST of its diagonal
# derivative vector — the per-coordinate derivative w.r.t. `var`. Symbols other
# than `var` (other vector variables or free constants) are constant w.r.t.
# `var`, so their derivative is 0.
.hess_diag <- function(expr, var, all_vars = var) {
  expr <- .strip_paren(expr)

  # Any expression with no dependence on the differentiation variable has zero
  # diagonal derivative. Guarding here lets var-free scalar-reducing factors
  # (sum(...), crossprod(...)) appear inside product / quotient rules without
  # hitting an unhandled .hess_diag case.
  if (!.contains_var(expr, var)) return(0)

  if (is.numeric(expr) || is.logical(expr)) return(0)
  if (is.symbol(expr)) {
    if (identical(as.character(expr), var)) {
      return(bquote(rep(1, length(.(as.symbol(var))))))
    }
    return(0)
  }
  if (!is.call(expr) || !is.symbol(expr[[1L]])) {
    .hessian_not_supported(expr)
  }
  op <- as.character(expr[[1L]])

  # Constant vector rep(value, ...) with var-free value -> derivative 0.
  if (op == "rep" && !.contains_var(expr[[2L]], var)) return(0)

  if (op == "-" && length(expr) == 2L) {
    return(.smart_neg(.hess_diag(expr[[2L]], var, all_vars)))
  }
  if (op == "+" && length(expr) == 2L) {
    return(.hess_diag(expr[[2L]], var, all_vars))
  }
  if (op %in% c("+", "-") && length(expr) == 3L) {
    da <- .hess_diag(expr[[2L]], var, all_vars)
    db <- .hess_diag(expr[[3L]], var, all_vars)
    return(if (op == "+") .smart_add(da, db) else .smart_sub(da, db))
  }
  if (op == "*" && length(expr) == 3L) {
    # General product rule: d/dvar(a * b) = a' * b + a * b'. Covers both a
    # var-free constant factor and a vector-grain Hadamard product of two
    # vector variables (one factor's derivative is 0 when it is constant).
    a <- expr[[2L]]; b <- expr[[3L]]
    return(.smart_add(
      .smart_mul(.hess_diag(a, var, all_vars), b),
      .smart_mul(a, .hess_diag(b, var, all_vars))
    ))
  }
  if (op == "^" && length(expr) == 3L) {
    base <- expr[[2L]]
    k <- expr[[3L]]
    inner <- .hess_diag(base, var, all_vars)
    return(.smart_mul(bquote(.(k) * .(base)^.(bquote(.(k) - 1))), inner))
  }
  # Scalar-denominator quotient rule: d/dvar (a / b) = (a' b - a b') / b^2.
  # Reached only for a var-free scalar denominator (a var-dependent denominator
  # is routed to the dense .jacobian_inner branch by .hess_shape). The var-free
  # guard on b' keeps a scalar-reducing denominator (e.g. sum(w^2), constant
  # w.r.t. the diff var) from hitting an unhandled .hess_diag case; b' = 0 then
  # collapses the rule to a' / b.
  if (op == "/" && length(expr) == 3L) {
    a <- expr[[2L]]; b <- expr[[3L]]
    da <- .hess_diag(a, var, all_vars)
    db <- if (.contains_var(b, var)) .hess_diag(b, var, all_vars) else 0
    num <- .smart_sub(.smart_mul(da, b), .smart_mul(a, db))
    return(bquote(.(num) / .(b)^2))
  }
  if (op %in% .HESS_ELEM_FNS && length(expr) == 2L) {
    arg <- expr[[2L]]
    outer_deriv <- switch(op,
      "sin"  = bquote(cos(.(arg))),
      "cos"  = bquote(-sin(.(arg))),
      "exp"  = bquote(exp(.(arg))),
      "log"  = bquote(1 / .(arg)),
      "tanh" = bquote(1 - tanh(.(arg))^2),
      "sqrt" = bquote(1 / (2 * sqrt(.(arg)))),
      "atan" = bquote(1 / (1 + .(arg)^2))
    )
    return(.smart_mul(outer_deriv, .hess_diag(arg, var, all_vars)))
  }

  .hessian_not_supported(expr)
}

# .jacobian_inner(expr, var, all_vars, orig)
#
# Walk a vector-valued gradient AST `expr` and return its Jacobian AST w.r.t.
# `var`. For a block of a multi-variable Hessian the result is n_a x n_b, where
# n_a is the length of `expr` (the per-variable gradient) and n_b is the length
# of `var`. Raises `hessian_not_supported` for shapes outside the rules.
.jacobian_inner <- function(expr, var, all_vars = var, orig = expr) {
  expr <- .strip_paren(expr)
  shp <- .hess_shape(expr, var, all_vars)

  # Pure vector-grain expressions have a diagonal Jacobian. When the diagonal
  # derivative is identically zero (expression is constant w.r.t. var), emit a
  # correctly-sized zero block: length(expr) rows x length(var) columns.
  if (shp == "vector_grain") {
    d <- .hess_diag(expr, var, all_vars)
    if (identical(d, 0) || identical(d, 0L)) {
      return(bquote(matrix(0, length(.(expr)), length(.(as.symbol(var))))))
    }
    return(bquote(diag(.(d))))
  }
  if (shp != "vector_dense") {
    .hessian_not_supported(orig)
  }

  op <- if (is.call(expr) && is.symbol(expr[[1L]])) as.character(expr[[1L]]) else ""

  if (op %in% c("+", "-") && length(expr) == 3L) {
    Ja <- .jacobian_inner(expr[[2L]], var, all_vars, orig)
    Jb <- .jacobian_inner(expr[[3L]], var, all_vars, orig)
    return(if (op == "+") bquote(.(Ja) + .(Jb)) else bquote(.(Ja) - .(Jb)))
  }
  if (op == "-" && length(expr) == 2L) {
    return(bquote(-.(.jacobian_inner(expr[[2L]], var, all_vars, orig))))
  }
  if (op == "+" && length(expr) == 2L) {
    return(.jacobian_inner(expr[[2L]], var, all_vars, orig))
  }

  # Scalar-times-vector: J(s * w) = w (x) grad(s) + s * J(w). When the scalar
  # factor is constant w.r.t. the differentiation variable, grad(s) = 0 and the
  # rank-one outer term vanishes, leaving J(s * w) = s * J(w). Returning that
  # directly avoids a degenerate length-1 outer product (non-conformable with
  # the n x n_var Jacobian block) — this arises in multi-variable blocks where
  # the scalar factor depends only on another variable.
  if (op == "*" && length(expr) == 3L) {
    a <- .hess_shape(expr[[2L]], var, all_vars)
    b <- .hess_shape(expr[[3L]], var, all_vars)
    if (a == "scalar")      { s_expr <- expr[[2L]]; w_expr <- expr[[3L]] }
    else if (b == "scalar") { s_expr <- expr[[3L]]; w_expr <- expr[[2L]] }
    else .hessian_not_supported(orig)
    Jw <- .jacobian_inner(w_expr, var, all_vars, orig)
    if (!.contains_var(s_expr, var)) {
      return(bquote(as.numeric(.(s_expr)) * .(Jw)))
    }
    grad_s <- .normalize_fast_kernels(.grad_expr(s_expr, var))
    return(bquote(
      outer(as.numeric(.(w_expr)), as.numeric(.(grad_s))) +
        as.numeric(.(s_expr)) * .(Jw)
    ))
  }

  # Constant matrix-vector product: J(W %*% w) = W %*% J(w). This is where a
  # rectangular block arises (W is n_a x n_b, J(w) is n_b x n_b -> n_a x n_b).
  if (op == "%*%" && length(expr) == 3L && !.contains_var(expr[[2L]], var)) {
    return(bquote(.(expr[[2L]]) %*% .(.jacobian_inner(expr[[3L]], var, all_vars, orig))))
  }

  # Scalar-denominator quotient via the scalar-factor decomposition
  # a / b = (1/b) * a, so J(a/b) = a (x) grad(1/b) + (1/b) * J(a), where
  # grad(1/b) = -grad(b) / b^2. Routes through the same outer(...) + scalar *
  # J(...) shape as the scalar x vector rule, so rectangular block dimensions
  # and the all_vars plumbing carry over unchanged.
  if (op == "/" && length(expr) == 3L) {
    a_expr <- expr[[2L]]
    b_expr <- expr[[3L]]
    if (.hess_shape(b_expr, var, all_vars) != "scalar") .hessian_not_supported(orig)
    Ja <- .jacobian_inner(a_expr, var, all_vars, orig)
    # A var-free denominator makes grad(1/b) = 0, so the rank-one correction
    # vanishes: J(a / b) = J(a) / b. Returning that directly avoids a degenerate
    # length-1 outer product (the dense numerator / constant-denominator case
    # that arises in multi-variable blocks).
    if (!.contains_var(b_expr, var)) {
      return(bquote((1 / as.numeric(.(b_expr))) * .(Ja)))
    }
    grad_b <- .normalize_fast_kernels(.grad_expr(b_expr, var))
    grad_recip <- bquote(-as.numeric(.(grad_b)) / as.numeric(.(b_expr))^2)
    return(bquote(
      outer(as.numeric(.(a_expr)), as.numeric(.(grad_recip))) +
        (1 / as.numeric(.(b_expr))) * .(Ja)
    ))
  }

  .hessian_not_supported(orig)
}

# .hessian_recursive(body_expr, var)
#
# Single-variable recursive Hessian path. Obtains the gradient AST from the
# grad engine (propagating dat_* conditions when the gradient is unsupported),
# then walks it into a Hessian matrix AST.
.hessian_recursive <- function(body_expr, var) {
  grad_ast <- .normalize_fast_kernels(.grad_expr(body_expr, var))
  .jacobian_inner(.strip_paren(grad_ast), var, var, orig = body_expr)
}

# .hessian_block(body_expr, row_var, col_var, all_vars)
#
# One block of a multi-variable Hessian: the Jacobian of the per-variable
# gradient grad_{row_var}(f) with respect to col_var. Returns an
# n_{row_var} x n_{col_var} matrix AST.
.hessian_block <- function(body_expr, row_var, col_var, all_vars) {
  grad_ast <- .normalize_fast_kernels(.grad_expr(body_expr, row_var))
  .jacobian_inner(.strip_paren(grad_ast), col_var, all_vars, orig = body_expr)
}
