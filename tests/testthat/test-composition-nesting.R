## test-composition-nesting.R
## Dimension: composition-nesting. Deep chains, matmul chains, bias terms, and
## products/sums of forms, grad+hessian vs numDeriv.

test_that("grad: deep nest exp(sin(log(sum(v^2))))", {
  skip_if_not_installed("numDeriv")
  v <- c(0.4, 0.9, 1.1)
  f <- function(v) exp(sin(log(sum(v^2))))
  expect_equal(as.numeric(grad(f)(v)), nd_grad(f, v), tolerance = 1e-8)
})

test_that("grad: 2-layer NN sum(tanh(W2 %*% tanh(W1 %*% v)))", {
  skip_if_not_installed("numDeriv")
  W1 <- matrix(c(0.2, 0.4, -0.1, 0.3, 0.5, -0.2), 3, 2)
  W2 <- matrix(c(0.6, -0.3, 0.4), 1, 3)
  f <- function(v) sum(tanh(W2 %*% tanh(W1 %*% v)))
  x <- c(0.3, -0.5)
  expect_equal(as.numeric(grad(f)(x)), nd_grad(f, x), tolerance = 1e-8)
})

test_that("grad: bias term sum(tanh(W %*% v + b))", {
  skip_if_not_installed("numDeriv")
  W <- matrix(c(0.2, 0.4, -0.1, 0.3, 0.5, -0.2), 3, 2)
  b <- c(0.1, -0.2, 0.3)
  f <- function(v) sum(tanh(W %*% v + b))
  x <- c(0.3, -0.5)
  expect_equal(as.numeric(grad(f)(x)), nd_grad(f, x), tolerance = 1e-8)
})

test_that("hessian: product of quadratic forms crossprod(v) * crossprod(v)", {
  skip_if_not_installed("numDeriv")
  v <- c(0.4, 0.9, 1.1)
  f <- function(v) crossprod(v) * crossprod(v)
  expect_equal(hessian(f)(v), nd_hessian(f, v), tolerance = 1e-5)
})

test_that("hessian: mixed quadratic + elementwise crossprod(v) + sum(sin(v))", {
  skip_if_not_installed("numDeriv")
  v <- c(0.4, 0.9, 1.1)
  f <- function(v) crossprod(v) + sum(sin(v))
  expect_equal(hessian(f)(v), nd_hessian(f, v), tolerance = 1e-5)
})

test_that("hessian: composite outer-scalar sin(sum(v^2))", {
  skip_if_not_installed("numDeriv")
  v <- c(0.4, 0.9, 1.1)
  f <- function(v) sin(sum(v^2))
  expect_equal(hessian(f)(v), nd_hessian(f, v), tolerance = 1e-5)
})
