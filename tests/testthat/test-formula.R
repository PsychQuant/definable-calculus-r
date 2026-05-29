test_that("one-sided formula returns one-sided formula", {
  result <- grad(~ sum(v^2), "v")
  expect_true(inherits(result, "formula"))
  expect_equal(length(result), 2L)
})

test_that("one-sided formula RHS equals 2 * v", {
  result <- grad(~ sum(v^2), "v")
  expect_equal(result[[2L]], quote(2 * v))
})

test_that("two-sided formula preserves LHS verbatim", {
  result <- grad(y ~ sum(v^2), "v")
  expect_true(inherits(result, "formula"))
  expect_equal(length(result), 3L)
  expect_equal(result[[2L]], quote(y))
  expect_equal(result[[3L]], quote(2 * v))
})

test_that("environment of result is identical to input formula environment", {
  custom_env <- new.env()
  input <- ~ sum(v^2)
  environment(input) <- custom_env
  result <- grad(input, "v")
  expect_identical(environment(result), custom_env)
})

test_that("vars is inferred when RHS has a single free variable", {
  result <- grad(~ sum(v^2))
  expect_equal(result[[2L]], quote(2 * v))
})

test_that("multi-variable RHS without vars defaults to all free variables", {
  # add-multi-variable-gradient: a RHS with multiple free variables now
  # defaults to the full gradient (named list of per-variable formulas)
  # instead of raising. Free vars of sum(v + w) are v and w.
  r <- grad(~ sum(v + w))
  expect_type(r, "list")
  expect_identical(names(r), c("v", "w"))
  expect_true(inherits(r$v, "formula") && inherits(r$w, "formula"))
})

test_that("formula result evaluates numerically equal to grad.function result", {
  # Build a callable from the formula's RHS and compare numerics with the
  # function-dispatch path. Uses the function pathway directly so the
  # comparison touches both methods.
  result_form <- grad(~ sum(v^2), "v")
  gf_func     <- grad(function(v) sum(v^2))

  rhs_expr <- result_form[[2L]]
  rhs_fn   <- as.function(c(alist(v = ), list(rhs_expr)))

  for (vec in list(c(1, 2, 3), c(-1.5, 0.7, 2.2), c(0, 0, 4))) {
    expect_equal(rhs_fn(vec), gf_func(vec))
  }
})
