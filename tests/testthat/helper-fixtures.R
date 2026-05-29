## helper-fixtures.R
## Common test fixtures.

f_sumsq <- function(v) sum(v^2)
f_inner <- function(v) sum(v * w)
f_quad  <- function(v) crossprod(v, A %*% v)[1, 1]

# True gradients for comparison
gf_sumsq_true <- function(v) 2 * v

# Sample vectors used throughout tests
.sample_v3   <- c(0.5, -1.2, 2.3)
.sample_seed <- 1234L

# Tolerance constants
TOL_NUMERIC <- 1e-6
TOL_TIGHT   <- 1e-10
