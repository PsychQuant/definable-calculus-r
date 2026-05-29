## test-edge-numerical.R
## Dimension: edge/numerical. n=1, moderate n, zeros, repeated values, and a
## large-but-finite scalar-denominator quotient.

EN_TOL <- 1e-10

test_that("hessian: length-1 (n=1) returns a correct 1x1 matrix", {
  # Post-fix regression guard for the diag(scalar) bug.
  H2 <- hessian(function(v) sum(v^2))(c(5))
  expect_equal(dim(H2), c(1L, 1L))
  expect_equal(as.numeric(H2), 2, tolerance = EN_TOL)
  expect_equal(as.numeric(hessian(function(v) sum(v^3))(c(2))), 12, tolerance = EN_TOL)
  expect_equal(as.numeric(hessian(function(v) sum(sin(v)))(c(0.7))), -sin(0.7), tolerance = EN_TOL)
  expect_equal(as.numeric(hessian(function(v) sum(exp(v)))(c(0.5))), exp(0.5), tolerance = EN_TOL)
})

test_that("hessian: moderate n=50 separable Hessian stays diagonal and correct", {
  set.seed(7)
  x <- runif(50, 0.5, 1.5)
  H <- hessian(function(v) sum(sin(v)))(x)
  expect_equal(dim(H), c(50L, 50L))
  expect_true(all(abs(H[upper.tri(H)]) < EN_TOL))
  expect_equal(diag(H), -sin(x), tolerance = EN_TOL)
})

test_that("grad: zeros and repeated values in the input vector", {
  expect_equal(as.numeric(grad(function(v) sum(v^2))(c(0, 1, 0))), c(0, 2, 0), tolerance = EN_TOL)
  expect_equal(as.numeric(grad(function(v) sum(exp(v)))(c(1, 1, 1))), rep(exp(1), 3), tolerance = EN_TOL)
})

test_that("hessian: large-but-finite scalar-denominator quotient stays finite and correct", {
  skip_if_not_installed("numDeriv")
  v <- c(5, 6, 7)
  f <- function(v) sum(v * exp(v)) / sum(exp(v))
  H <- hessian(f)(v)
  expect_true(all(is.finite(H)))
  expect_equal(H, nd_hessian(f, v), tolerance = 1e-4)
})
