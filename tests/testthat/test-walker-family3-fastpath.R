## test-walker-family3-fastpath.R
## Tier 4 change `add-walker-family3-fastpath`: family-3 walker fast-paths
## (FINAL sub-change in the walker-fast-paths series).
## Pattern A: sum(v^2 +/- f(v)) additive composition (commutative for +,
## non-commutative for -). Pattern B: sum(f(k*v)) chain-through-scalar-multiply
## via Tier 2d-style body block.

skip_if_no_fast <- function() {
  if (!.fast_path_available()) skip("macOS fast-path required")
}

v_test <- c(0.5, 1.0, 1.5)

# ===== Pattern A: 6 (operator × function) combinations =====

test_that("Pattern A: sum(v^2 + sin(v)) = 2*v + cos(v)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v^2 + sin(v)))
  expected <- 2 * v_test + cos(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("Pattern A: sum(v^2 + cos(v)) = 2*v - sin(v) (cos derivative flips sign)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v^2 + cos(v)))
  expected <- 2 * v_test - sin(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("Pattern A: sum(v^2 + exp(v)) = 2*v + exp(v)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v^2 + exp(v)))
  expected <- 2 * v_test + exp(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("Pattern A: sum(v^2 - sin(v)) = 2*v - cos(v)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v^2 - sin(v)))
  expected <- 2 * v_test - cos(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("Pattern A: sum(v^2 - cos(v)) = 2*v + sin(v) (subtracting -sin flips)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v^2 - cos(v)))
  expected <- 2 * v_test + sin(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("Pattern A: sum(v^2 - exp(v)) = 2*v - exp(v)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v^2 - exp(v)))
  expected <- 2 * v_test - exp(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

# Commutative variant
test_that("Pattern A commutative: sum(sin(v) + v^2) matches sum(v^2 + sin(v))", {
  skip_if_no_fast()
  gf1 <- grad(function(v) sum(sin(v) + v^2))
  gf2 <- grad(function(v) sum(v^2 + sin(v)))
  expect_equal(as.numeric(gf1(v_test)), as.numeric(gf2(v_test)), tolerance = 1e-10)
})

# Non-commutative subtraction
test_that("Pattern A non-commutative: sum(sin(v) - v^2) = cos(v) - 2*v (NOT 2*v - cos(v))", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(sin(v) - v^2))
  expected <- cos(v_test) - 2 * v_test
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

# Body-shape assertions
test_that("Pattern A body for sum(v^2 + sin(v)) uses fast_vec_add + fast_scalar_mul + fast_vv_cos", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v^2 + sin(v)))
  body_str <- deparse(body(gf))
  expect_true(any(grepl("fast_vec_add", body_str, fixed = TRUE)))
  expect_true(any(grepl("fast_scalar_mul", body_str, fixed = TRUE)))
  expect_true(any(grepl("fast_vv_cos", body_str, fixed = TRUE)))
})

test_that("Pattern A body for sum(v^2 + cos(v)) uses fast_vec_sub (cos derivative flips sign)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v^2 + cos(v)))
  body_str <- deparse(body(gf))
  expect_true(any(grepl("fast_vec_sub", body_str, fixed = TRUE)))
  expect_true(any(grepl("fast_vv_sin", body_str, fixed = TRUE)))
})

# ===== Pattern B: 3 functions =====

test_that("Pattern B: sum(sin(2*v)) = 2*cos(2*v)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(sin(2 * v)))
  expected <- 2 * cos(2 * v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("Pattern B: sum(cos(3*v)) = -3*sin(3*v) (negative scalar from cos derivative)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(cos(3 * v)))
  expected <- -3 * sin(3 * v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("Pattern B: sum(exp(0.5*v)) = 0.5*exp(0.5*v)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(exp(0.5 * v)))
  expected <- 0.5 * exp(0.5 * v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("Pattern B commutative: sum(sin(v * 2)) matches sum(sin(2 * v))", {
  skip_if_no_fast()
  gf1 <- grad(function(v) sum(sin(v * 2)))
  gf2 <- grad(function(v) sum(sin(2 * v)))
  expect_equal(as.numeric(gf1(v_test)), as.numeric(gf2(v_test)), tolerance = 1e-10)
})

test_that("Pattern B body is a `{` block with kv <- binding", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(sin(2 * v)))
  expect_identical(body(gf)[[1L]], quote(`{`))
  body_str <- deparse(body(gf))
  expect_true(any(grepl("kv <- fast_scalar_mul(2, v)", body_str, fixed = TRUE)))
  # cos case has negative scalar in source
  gf_c <- grad(function(v) sum(cos(3 * v)))
  body_c_str <- deparse(body(gf_c))
  expect_true(any(grepl("-3", body_c_str, fixed = TRUE)))
})

# ===== Negative test =====

test_that("Negative: sum(v^3 + sin(v)) falls through (Pattern A only matches v^2)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v^3 + sin(v)))
  expected <- 3 * v_test^2 + cos(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
  # The body should NOT match Pattern A's signature
  # (i.e., NOT fast_vec_add(fast_scalar_mul(2, v), ...) — would be for v^2)
  body_str <- deparse(body(gf))
  expect_false(any(grepl("fast_vec_add(fast_scalar_mul(2, v)", body_str, fixed = TRUE)))
})
