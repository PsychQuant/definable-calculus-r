## real_world_patterns.R
## Phase 4: Real-world ML/stat patterns + their analytic-gradient direct-R
## evaluators. Sourced by comprehensive-ad-comparison.R and
## tests/testthat/test-real-world-patterns.R.
##
## Each pattern has two forms:
##   f         — natural multi-statement style (uses intermediate vars).
##                Used as the correctness baseline (direct R evaluation).
##   f_inline  — single-expression equivalent that DD can ingest. Tested.
## When f_inline is NULL the pattern needs multi-statement support that DD
## currently does not provide — classified OUT-OF-SCOPE.
## nn_forward uses fixed weights (closure-bound) so the engine sees only v
## as the active variable — keeps single-variable scope intact (Decision 5).

set.seed(20260525L)

# Fixed NN weights — chosen so that for any n, W1 has shape (10, n) and W2
# is (1, 10). Re-built lazily so n can vary at call time.
.nn_weights_cache <- new.env(parent = emptyenv())
.nn_weights <- function(n) {
  key <- as.character(n)
  if (!exists(key, envir = .nn_weights_cache, inherits = FALSE)) {
    local_seed <- 42L + as.integer(n)
    set.seed(local_seed)
    W1 <- matrix(rnorm(10L * n), 10L, n)
    W2 <- matrix(rnorm(10L),     1L, 10L)
    assign(key, list(W1 = W1, W2 = W2), envir = .nn_weights_cache)
    set.seed(20260525L)
  }
  get(key, envir = .nn_weights_cache, inherits = FALSE)
}

# Closure factory: returns f_inline functions that capture fixed weights /
# labels by lexical scope, so the body is a single expression.

.make_logistic_inline <- function() {
  function(v) sum(log(1 + exp(-rep(c(1, -1), length.out = length(v)) * v)))
}

.make_gaussian_inline <- function() {
  # mu=0, sigma=1 → expression collapses to -0.5 * sum(v^2)
  function(v) -0.5 * sum(v^2)
}

.make_softmax_inline <- function() {
  # log(sum(exp(v))) - sum(v * exp(v)) / sum(exp(v))
  # Two sum(exp(v)) calls — DD's walker should handle.
  function(v) log(sum(exp(v))) - sum(v * exp(v)) / sum(exp(v))
}

.make_nn_inline <- function(n) {
  w <- .nn_weights(n)
  W1 <- w$W1
  W2 <- w$W2
  # Single-expression body using lexically-captured W1, W2.
  # Note: this only works for the fixed n — caller must build per-n.
  function(v) sum(tanh(W2 %*% tanh(W1 %*% v)))
}

.make_kld_inline <- function() {
  # mu=0 → 0.5 * sum(v^2)
  function(v) 0.5 * sum(v^2)
}

# ===== Pattern definitions =====

real_world_patterns <- list(
  logistic_loss = list(
    label = "sum(log(1 + exp(-y * v)))",
    f = function(v) {
      y <- rep(c(1, -1), length.out = length(v))
      sum(log(1 + exp(-y * v)))
    },
    f_inline = .make_logistic_inline(),
    description = "Logistic regression loss; y closure-inlined into expression"
  ),
  gaussian_loglik = list(
    label = "-0.5 * sum(v^2)  (mu=0, sigma=1)",
    f = function(v) -0.5 * sum(((v - 0) / 1)^2),
    f_inline = .make_gaussian_inline(),
    description = "Gaussian log-likelihood collapsed to mu=0, sigma=1"
  ),
  softmax_entropy = list(
    label = "log(sum(exp(v))) - sum(v*exp(v))/sum(exp(v))",
    f = function(v) {
      ev <- exp(v)
      sev <- sum(ev)
      log(sev) - sum(v * ev) / sev
    },
    f_inline = .make_softmax_inline(),
    description = "Entropy of softmax(v); inline form has 3 redundant exp/sum calls"
  ),
  nn_forward = list(
    label = "sum(tanh(W2 %*% tanh(W1 %*% v)))",
    f = function(v) {
      w <- .nn_weights(length(v))
      sum(tanh(w$W2 %*% tanh(w$W1 %*% v)))
    },
    # Built per-n at use site via .make_nn_inline(n).
    f_inline_builder = .make_nn_inline,
    description = "2-layer NN forward; W1/W2 closure-bound. Requires per-n builder for inline form"
  ),
  kld_normal = list(
    label = "0.5 * sum(v^2)  (mu=0)",
    f = function(v) 0.5 * sum((v - 0)^2),
    f_inline = .make_kld_inline(),
    description = "KL-divergence simplified (mu=0)"
  )
)

# ===== Analytic-gradient direct-R evaluators =====

real_world_analytic <- list(
  logistic_loss = function(v) {
    y <- rep(c(1, -1), length.out = length(v))
    -y * exp(-y * v) / (1 + exp(-y * v))
  },
  gaussian_loglik = function(v) -v,
  softmax_entropy = function(v) {
    ev <- exp(v)
    p <- ev / sum(ev)
    -p * (v - sum(v * p))
  },
  nn_forward = function(v) {
    w <- .nn_weights(length(v))
    h1 <- tanh(as.numeric(w$W1 %*% v))
    h2 <- tanh(as.numeric(w$W2 %*% h1))
    delta1 <- as.numeric(t(w$W2) %*% (1 - h2^2))
    delta_Wv <- delta1 * (1 - h1^2)
    as.numeric(t(w$W1) %*% delta_Wv)
  },
  kld_normal = function(v) v
)
