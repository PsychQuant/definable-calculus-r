test_that("level() infers tier across hierarchy", {
  expect_equal(level(quote(v + w)),                       "L_0")
  expect_equal(level(quote(sum(v^2))),                    "L_1")
  expect_equal(level(quote(crossprod(v, A %*% v))),       "L_2")
  expect_equal(level(quote(sin(sum(v^2)))),               "L_3")
  expect_equal(level(quote(unknown_fn(v))),               "unknown")
})

test_that("level() accepts function input via body()", {
  expect_equal(level(function(v) sum(v^2)),               "L_1")
  expect_equal(level(function(v) sin(sum(v^2))),          "L_3")
})

test_that("level() accepts expression input", {
  expect_equal(level(expression(sum(v^2))),               "L_1")
})

test_that("level() bottom-up: max across subexpressions", {
  # exp at outer is L_3, inner is L_1 → overall L_3
  expect_equal(level(quote(exp(crossprod(v, w)))),        "L_3")
  # outer L_1, inner has L_2 (named operator) → overall L_2
  expect_equal(level(quote(sum(A %*% v))),                "L_2")
})

test_that("level() returns 'unknown' for any unrecognised sub-expression", {
  expect_equal(level(quote(sin(unknown_fn(v)))),          "unknown")
  expect_equal(level(quote(sum(unknown_fn(v)))),          "unknown")
})

test_that("level() returns L_0 for symbol or constant", {
  expect_equal(level(quote(v)),                            "L_0")
  expect_equal(level(quote(3.14)),                         "L_0")
})
