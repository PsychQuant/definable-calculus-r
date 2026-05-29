## level.R
## Bottom-up inference of L_i language tier for an R expression.

#' Infer the language tier of an expression
#'
#' Recursively determines the smallest tier (L_0, L_1, L_2, L_3) of the
#' Definable Algebra Theory hierarchy in which the expression is definable,
#' or `"unknown"` if it contains a function name not present in the active
#' catalog. The inference is bottom-up: a composite expression's tier is
#' the maximum tier of any generator appearing in any sub-expression.
#'
#' @param expr An R object: call, expression, function, or atom.
#' @return Character: one of `"L_0"`, `"L_1"`, `"L_2"`, `"L_3"`, `"unknown"`.
#' @export
#' @examples
#' level(quote(v + w))                       # "L_0"
#' level(quote(sum(v^2)))                    # "L_1"
#' level(quote(crossprod(v, A %*% v)))       # "L_2"
#' level(quote(sin(sum(v^2))))               # "L_3"
#' level(quote(unknown_fn(v)))               # "unknown"
level <- function(expr) {
  if (is.function(expr)) {
    expr <- body(expr)
  }
  if (inherits(expr, "expression")) {
    if (length(expr) == 0L) return("L_0")
    expr <- expr[[1L]]
  }
  .infer_level(expr)
}

# Recursive worker. Returns one of the tier strings.
.infer_level <- function(expr) {
  expr <- .strip_paren(expr)
  if (is.symbol(expr) || is.numeric(expr) || is.logical(expr) || is.character(expr)) {
    return("L_0")
  }
  if (!is.call(expr)) return("unknown")

  head_sym <- expr[[1L]]
  if (!is.symbol(head_sym)) return("unknown")
  fname <- as.character(head_sym)

  own <- .lookup_level(fname)
  if (is.na(own)) return("unknown")

  sub_levels <- character(0)
  for (i in seq_along(expr)[-1L]) {
    sub_levels <- c(sub_levels, .infer_level(expr[[i]]))
  }

  .max_level(c(own, sub_levels))
}

# Maximum of a set of tier labels using .level_rank ordering.
.max_level <- function(levels) {
  ranks <- vapply(levels, .level_rank, integer(1))
  if (any(is.na(ranks))) return("unknown")
  if (length(ranks) == 0L) return("L_0")
  idx <- which.max(ranks)
  levels[idx]
}
