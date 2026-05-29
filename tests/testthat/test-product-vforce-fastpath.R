## test-product-vforce-fastpath.R
## Tier 4 change `add-product-vforce-fastpath`: family-1 walker fast-path
## for sum(v * f(v)) and crossprod(v, f(v)) where f ∈ {sin, cos, exp}.
## Emits fused gradient AST via existing fast_vec_add/sub/mul + fast_vv_*
## kernels, bypassing the walker's stock-R evaluation path.

skip_if_no_fast <- function() {
  if (!.fast_path_available()) skip("macOS fast-path required")
}

v_test <- c(0.5, 1.0, 1.5)

# ---- 1-3: Forward numeric correctness for the 3 supported functions ----

test_that("sum(v * sin(v)) matches sin(v) + v*cos(v)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v * sin(v)))
  expected <- sin(v_test) + v_test * cos(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("sum(v * cos(v)) matches cos(v) - v*sin(v)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v * cos(v)))
  expected <- cos(v_test) - v_test * sin(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("sum(v * exp(v)) matches exp(v) + v*exp(v)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v * exp(v)))
  expected <- exp(v_test) + v_test * exp(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

# ---- 4: Commutative variant (operand order swapped) ----

test_that("sum(sin(v) * v) commutative variant equals sum(v * sin(v))", {
  skip_if_no_fast()
  gf_swap <- grad(function(v) sum(sin(v) * v))
  gf_orig <- grad(function(v) sum(v * sin(v)))
  expect_equal(as.numeric(gf_swap(v_test)), as.numeric(gf_orig(v_test)), tolerance = 1e-10)
})

# ---- 5-6: crossprod variants (equivalent to sum) ----

test_that("crossprod(v, sin(v)) equals sum(v * sin(v))", {
  skip_if_no_fast()
  gf_cp <- grad(function(v) crossprod(v, sin(v)))
  gf_sum <- grad(function(v) sum(v * sin(v)))
  expect_equal(as.numeric(gf_cp(v_test)), as.numeric(gf_sum(v_test)), tolerance = 1e-10)
})

test_that("crossprod(sin(v), v) commutative crossprod also equals sum(v * sin(v))", {
  skip_if_no_fast()
  gf_cp_swap <- grad(function(v) crossprod(sin(v), v))
  gf_sum <- grad(function(v) sum(v * sin(v)))
  expect_equal(as.numeric(gf_cp_swap(v_test)), as.numeric(gf_sum(v_test)), tolerance = 1e-10)
})

# ---- 7-8: Body-shape assertions (proves fast-path dispatch fired) ----

test_that("Body of grad(sum(v * sin(v))) uses fast_vec_add + fast_vv_sin/cos", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v * sin(v)))
  body_str <- deparse(body(gf))[[1L]]
  expect_match(body_str, "fast_vec_add", fixed = TRUE)
  expect_match(body_str, "fast_vv_sin", fixed = TRUE)
  expect_match(body_str, "fast_vv_cos", fixed = TRUE)
  expect_match(body_str, "fast_vec_mul", fixed = TRUE)
})

test_that("Body of grad(sum(v * cos(v))) uses fast_vec_sub + fast_vv_cos/sin", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v * cos(v)))
  body_str <- deparse(body(gf))[[1L]]
  expect_match(body_str, "fast_vec_sub", fixed = TRUE)
  expect_match(body_str, "fast_vv_cos", fixed = TRUE)
  expect_match(body_str, "fast_vv_sin", fixed = TRUE)
})

# ---- 9-12: Negative tests (patterns must NOT match — fall through to walker
# or other existing paths) ----

test_that("sum(v * tanh(v)) falls through to walker (tanh not in supported set)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v * tanh(v)))
  expected <- tanh(v_test) + v_test * (1 - tanh(v_test)^2)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
  # Body should NOT contain the family-1 fast-path signature
  body_str <- deparse(body(gf))[[1L]]
  expect_false(grepl("fast_vec_add", body_str, fixed = TRUE))
})

test_that("sum((v + 1) * sin(v)) falls through (first operand not bare var)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum((v + 1) * sin(v)))
  # Analytic gradient: d/dv_i sum((v_i+1) * sin(v_i)) = sin(v_i) + (v_i+1)*cos(v_i)
  expected <- sin(v_test) + (v_test + 1) * cos(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("sum(2 * sin(v)) falls through (constant scalar, not var)", {
  skip_if_no_fast()
  # Existing Tier 2c scaled-elementwise path handles this; the family-1
  # check (one side must be bare var) doesn't match.
  gf <- grad(function(v) sum(2 * sin(v)))
  expected <- 2 * cos(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("crossprod(v + 1, sin(v)) falls through (non-bare-var first arg)", {
  skip_if_no_fast()
  gf <- grad(function(v) crossprod(v + 1, sin(v)))
  # crossprod(v+1, sin(v)) = sum((v+1) * sin(v)); same gradient as test 10.
  expected <- sin(v_test) + (v_test + 1) * cos(v_test)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})
