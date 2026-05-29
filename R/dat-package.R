#' dat: Definable Algebra Theory R Reference Implementation
#'
#' Symbolic source-transformation differentiation for vector calculus
#' expressions with awareness of a four-tier first-order language hierarchy
#' (L0 pure vector space, L1 inner product, L2 named operators, L3 analytic
#' scalar functions).
#'
#' @section Main exported functions:
#' \itemize{
#'   \item [grad()] - S3 generic for symbolic differentiation
#'   \item [level()] - Infer language tier of an expression
#'   \item [extend_language()] - Register a new generator in the catalog
#'   \item [language_catalog()] - Inspect the active catalog
#'   \item [verify_grad()] - Three-layer verification of a gradient
#' }
#'
#' @keywords internal
"_PACKAGE"
