## test-walker-family2-fastpath.R
## Tier 4 change `add-walker-family2-fastpath`: family-2 walker fast-paths.
## Pattern A: sum(v / f(v)) quotient.  Pattern B: sum/crossprod of two
## vForce calls.  Plus fast_vec_div kernel correctness + swap-regression.

skip_if_no_fast <- function() {
  if (!.fast_path_available()) skip("macOS fast-path required")
}

v_test <- c(0.5, 1.0, 1.5)

# ===== Pattern A: quotient sum(v / f(v)) =====

test_that("Pattern A: sum(v / sin(v)) matches analytic quotient", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v / sin(v)))
  expected <- (sin(v_test) - v_test * cos(v_test)) / sin(v_test)^2
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("Pattern A: sum(v / cos(v)) matches analytic quotient", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v / cos(v)))
  expected <- (cos(v_test) + v_test * sin(v_test)) / cos(v_test)^2
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("Pattern A: sum(v / exp(v)) matches analytic quotient", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v / exp(v)))
  expected <- (exp(v_test) - v_test * exp(v_test)) / exp(v_test)^2
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

# Pattern A body shape — kernels invoked
test_that("Pattern A body for sum(v / sin(v)) uses fast_vec_div + fast_vv_sin/cos", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v / sin(v)))
  body_str <- deparse(body(gf))
  expect_true(any(grepl("fast_vec_div", body_str, fixed = TRUE)))
  expect_true(any(grepl("fast_vv_sin", body_str, fixed = TRUE)))
  expect_true(any(grepl("fast_vv_cos", body_str, fixed = TRUE)))
})

test_that("Pattern A body for sum(v / cos(v)) uses fast_vec_div + fast_vv_cos/sin", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v / cos(v)))
  body_str <- deparse(body(gf))
  expect_true(any(grepl("fast_vec_div", body_str, fixed = TRUE)))
  expect_true(any(grepl("fast_vv_cos", body_str, fixed = TRUE)))
  expect_true(any(grepl("fast_vv_sin", body_str, fixed = TRUE)))
})

test_that("Pattern A body for sum(v / exp(v)) uses fast_vec_div + fast_vv_exp", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v / exp(v)))
  body_str <- deparse(body(gf))
  expect_true(any(grepl("fast_vec_div", body_str, fixed = TRUE)))
  expect_true(any(grepl("fast_vv_exp", body_str, fixed = TRUE)))
})

# ===== Pattern B: product of two vForces (6 distinct pairs) =====

test_that("Pattern B: sum(sin(v) * sin(v)) = 2*sin*cos", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(sin(v) * sin(v)))
  expected <- 2 * sin(v_test) * cos(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("Pattern B: sum(sin(v) * cos(v)) = cos^2 - sin^2", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(sin(v) * cos(v)))
  expected <- cos(v_test)^2 - sin(v_test)^2
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("Pattern B: sum(sin(v) * exp(v)) = (cos+sin)*exp", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(sin(v) * exp(v)))
  expected <- cos(v_test) * exp(v_test) + sin(v_test) * exp(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("Pattern B: sum(cos(v) * cos(v)) = -2*sin*cos (negative-sign case)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(cos(v) * cos(v)))
  expected <- -2 * sin(v_test) * cos(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("Pattern B: sum(cos(v) * exp(v)) = (cos-sin)*exp", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(cos(v) * exp(v)))
  expected <- cos(v_test) * exp(v_test) - sin(v_test) * exp(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("Pattern B: sum(exp(v) * exp(v)) = 2*exp^2", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(exp(v) * exp(v)))
  expected <- 2 * exp(v_test) * exp(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

# Pattern B commutative variants — verify pair normalization
test_that("Pattern B commutative: cos*sin equals sin*cos", {
  skip_if_no_fast()
  gf1 <- grad(function(v) sum(cos(v) * sin(v)))
  gf2 <- grad(function(v) sum(sin(v) * cos(v)))
  expect_equal(as.numeric(gf1(v_test)), as.numeric(gf2(v_test)), tolerance = 1e-10)
})

test_that("Pattern B commutative: exp*sin equals sin*exp", {
  skip_if_no_fast()
  gf1 <- grad(function(v) sum(exp(v) * sin(v)))
  gf2 <- grad(function(v) sum(sin(v) * exp(v)))
  expect_equal(as.numeric(gf1(v_test)), as.numeric(gf2(v_test)), tolerance = 1e-10)
})

test_that("Pattern B commutative: exp*cos equals cos*exp", {
  skip_if_no_fast()
  gf1 <- grad(function(v) sum(exp(v) * cos(v)))
  gf2 <- grad(function(v) sum(cos(v) * exp(v)))
  expect_equal(as.numeric(gf1(v_test)), as.numeric(gf2(v_test)), tolerance = 1e-10)
})

# Pattern B crossprod equivalence
test_that("Pattern B crossprod(sin(v), cos(v)) equals sum(sin(v) * cos(v))", {
  skip_if_no_fast()
  gf_cp <- grad(function(v) crossprod(sin(v), cos(v)))
  gf_sum <- grad(function(v) sum(sin(v) * cos(v)))
  expect_equal(as.numeric(gf_cp(v_test)), as.numeric(gf_sum(v_test)), tolerance = 1e-10)
})

test_that("Pattern B crossprod(cos(v), cos(v)) equals sum(cos(v) * cos(v))", {
  skip_if_no_fast()
  gf_cp <- grad(function(v) crossprod(cos(v), cos(v)))
  gf_sum <- grad(function(v) sum(cos(v) * cos(v)))
  expect_equal(as.numeric(gf_cp(v_test)), as.numeric(gf_sum(v_test)), tolerance = 1e-10)
})

# Pattern B body shape
test_that("Pattern B body for sum(sin(v) * cos(v)) uses fast_vec_sub + fast_vv_sin/cos", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(sin(v) * cos(v)))
  body_str <- deparse(body(gf))
  expect_true(any(grepl("fast_vec_sub", body_str, fixed = TRUE)))
  expect_true(any(grepl("fast_vv_sin", body_str, fixed = TRUE)))
  expect_true(any(grepl("fast_vv_cos", body_str, fixed = TRUE)))
})

# ===== Negative tests — patterns must fall through =====

test_that("Negative: sum(sin(v) / v) (reverse quotient) falls through to walker", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(sin(v) / v))
  # Analytic: d/dv (sin(v)/v) = (cos(v)*v - sin(v)) / v^2 = cos(v)/v - sin(v)/v^2
  expected <- (cos(v_test) * v_test - sin(v_test)) / v_test^2
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
  body_str <- deparse(body(gf))
  expect_false(any(grepl("fast_vec_div", body_str, fixed = TRUE)))
})

test_that("Negative: sum(v / tanh(v)) (tanh unsupported) falls through to walker", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v / tanh(v)))
  # Analytic: d/dv (v/tanh(v)) = (tanh(v) - v*(1-tanh^2)) / tanh(v)^2
  expected <- (tanh(v_test) - v_test * (1 - tanh(v_test)^2)) / tanh(v_test)^2
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
  body_str <- deparse(body(gf))
  expect_false(any(grepl("fast_vec_div", body_str, fixed = TRUE)))
})

test_that("Negative: sum(sin(v) * cos(v) * exp(v)) (three-way product) falls through", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(sin(v) * cos(v) * exp(v)))
  # Product rule: d/dv (sin*cos*exp) = cos*cos*exp + sin*(-sin)*exp + sin*cos*exp
  expected <- cos(v_test) * cos(v_test) * exp(v_test) -
              sin(v_test) * sin(v_test) * exp(v_test) +
              sin(v_test) * cos(v_test) * exp(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("Negative: sum((v + 1) / sin(v)) (non-bare-var numerator) falls through", {
  skip_if_no_fast()
  gf <- grad(function(v) sum((v + 1) / sin(v)))
  # Analytic: d/dv ((v+1)/sin(v)) = (sin(v) - (v+1)*cos(v)) / sin(v)^2
  expected <- (sin(v_test) - (v_test + 1) * cos(v_test)) / sin(v_test)^2
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

# ===== fast_vec_div kernel tests =====

test_that("fast_vec_div: swap-regression — c(6,8,10) / c(2,4,5) = c(3,2,2)", {
  skip_if_no_fast()
  # The critical swap-convention test. Wrong-swap would return c(0.333, 0.5, 0.5).
  out <- fast_vec_div(c(6, 8, 10), c(2, 4, 5))
  expect_equal(as.numeric(out), c(3, 2, 2), tolerance = 4 * .Machine$double.eps)
})

test_that("fast_vec_div: length mismatch raises", {
  skip_if_no_fast()
  expect_error(fast_vec_div(c(1, 2, 3), c(1, 2)), regexp = "length mismatch")
})
