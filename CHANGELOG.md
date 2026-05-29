# Changelog

All notable changes to the `dat` package are documented here.
Format loosely follows Keep a Changelog; entries reference their Spectra change.

## [Unreleased]

### Fixed

- **Hessian for length-1 (n=1) input.** The Hessian fast-path emitted
  `diag(<length-1 vector>)`, which R reinterprets as a matrix *dimension*, so
  `hessian(function(v) sum(v^2))(c(5))` returned a 2x2 identity instead of the
  1x1 `[[2]]` (and `sum(v^3))(c(2))` returned 12x12, `sum(sin(v))` returned
  0x0). All five fast-path diagonals are now sized via
  `diag(d, nrow = length(v))`; output for `n >= 2` is unchanged. `grad()` was
  never affected. (add-comprehensive-grad-hessian-tests)

### Added

- **Comprehensive cross-validated test suite** for `grad` / `hessian` /
  `jacobian` across eight dimensions: numeric-equivalence (triple ground truth
  — numDeriv + an independent central finite-difference + closed form, with a
  ground-truth-vs-ground-truth cross-check), catalog coverage, composition /
  nesting, multi-variable block Hessian, edge / numerical (n=1, moderate n,
  zeros, degenerate denominators), property-based invariants (gradient
  linearity, Hessian symmetry, cross-engine `Hessian == numerical-Jacobian(grad)`),
  boundary must-raise plus negative-of-negative, and fast-path-vs-recursive
  equivalence. Full suite at 1110 passing. (add-comprehensive-grad-hessian-tests)
