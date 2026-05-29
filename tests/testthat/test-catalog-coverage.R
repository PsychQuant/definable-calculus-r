## test-catalog-coverage.R
## Dimension: catalog-coverage. One grad (and hessian where supported) case per
## catalog generator, each checked against its closed form.

CC_TOL <- 1e-10

test_that("grad: each elementwise fn matches its closed-form derivative", {
  v <- c(0.5, 0.8, 1.2)
  expect_equal(as.numeric(grad(function(v) sum(sin(v)))(v)),  cos(v),          tolerance = CC_TOL)
  expect_equal(as.numeric(grad(function(v) sum(cos(v)))(v)),  -sin(v),         tolerance = CC_TOL)
  expect_equal(as.numeric(grad(function(v) sum(exp(v)))(v)),  exp(v),          tolerance = CC_TOL)
  expect_equal(as.numeric(grad(function(v) sum(log(v)))(v)),  1 / v,           tolerance = CC_TOL)
  expect_equal(as.numeric(grad(function(v) sum(tanh(v)))(v)), 1 - tanh(v)^2,   tolerance = CC_TOL)
  expect_equal(as.numeric(grad(function(v) sum(sqrt(v)))(v)), 1 / (2*sqrt(v)), tolerance = CC_TOL)
  expect_equal(as.numeric(grad(function(v) sum(atan(v)))(v)), 1 / (1 + v^2),   tolerance = CC_TOL)
})

test_that("grad: power rule sum(v^k) for k = 2, 3, 4", {
  # Exponent must be a literal in the function body (dat's power rule needs a
  # numeric constant, not a captured loop variable).
  v <- c(0.5, 0.8, 1.2)
  expect_equal(as.numeric(grad(function(v) sum(v^2))(v)), 2 * v,      tolerance = CC_TOL)
  expect_equal(as.numeric(grad(function(v) sum(v^3))(v)), 3 * v^2,    tolerance = CC_TOL)
  expect_equal(as.numeric(grad(function(v) sum(v^4))(v)), 4 * v^3,    tolerance = CC_TOL)
})

test_that("grad: arithmetic rules + - * /", {
  v <- c(0.5, 0.8, 1.2)
  expect_equal(as.numeric(grad(function(v) sum(v^2) + sum(sin(v)))(v)), 2*v + cos(v), tolerance = CC_TOL)
  expect_equal(as.numeric(grad(function(v) sum(v^2) - sum(sin(v)))(v)), 2*v - cos(v), tolerance = CC_TOL)
  expect_equal(as.numeric(grad(function(v) 3 * sum(v^2))(v)), 3 * 2 * v, tolerance = CC_TOL)
  expect_equal(as.numeric(grad(function(v) sum(v^2) / sum(v))(v)),
               (2*v * sum(v) - sum(v^2)) / sum(v)^2, tolerance = CC_TOL)
})

test_that("grad: reductions sum, crossprod, constant matmul", {
  v <- c(0.5, 0.8, 1.2)
  expect_equal(as.numeric(grad(function(v) sum(v))(v)), rep(1, 3), tolerance = CC_TOL)
  expect_equal(as.numeric(grad(function(v) crossprod(v))(v)), 2 * v, tolerance = CC_TOL)
  W <- matrix(c(1, 2, 3, 4, 5, 6), 2, 3)
  expect_equal(as.numeric(grad(function(v) sum(W %*% v))(v)),
               as.numeric(t(W) %*% rep(1, 2)), tolerance = CC_TOL)
})

test_that("hessian: supported elementwise + power diagonals vs closed form", {
  v <- c(0.5, 0.8, 1.2)
  expect_equal(diag(hessian(function(v) sum(sin(v)))(v)),  -sin(v),                  tolerance = CC_TOL)
  expect_equal(diag(hessian(function(v) sum(cos(v)))(v)),  -cos(v),                  tolerance = CC_TOL)
  expect_equal(diag(hessian(function(v) sum(exp(v)))(v)),   exp(v),                  tolerance = CC_TOL)
  expect_equal(diag(hessian(function(v) sum(log(v)))(v)),  -1 / v^2,                 tolerance = CC_TOL)
  expect_equal(diag(hessian(function(v) sum(tanh(v)))(v)), -2*tanh(v)*(1-tanh(v)^2), tolerance = CC_TOL)
  expect_equal(diag(hessian(function(v) sum(v^3))(v)),      6 * v,                   tolerance = CC_TOL)
})
