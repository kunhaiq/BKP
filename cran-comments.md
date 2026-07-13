## Release summary

This is an update to BKP 0.3.0, which is currently available on CRAN.

BKP 0.3.1 contains the following user-facing changes:

* Reordered the optional arguments of `fit_BKP()` and `fit_DKP()` so that
  kernel specification, length-scale settings, loss and effective-sample-size
  options, and optimization controls are grouped more consistently. The
  required arguments, argument names, default values, and named-argument usage
  are unchanged. Calls that pass optional arguments positionally should be
  updated to use named arguments.
* Corrected probability-scale plot labels in the BKP, DKP, TwinBKP, and
  TwinDKP plotting methods. The labels now use "Posterior Mean", "Posterior
  Variance", and "95% Credible Interval" rather than predictive terminology,
  because these panels summarize posterior uncertainty in the underlying
  probability surface.
* Added links to the companion BKP-paper reproducibility repository and
  expanded the README with a model-selection guide, worked examples, and a
  two-dimensional posterior-summary illustration.
* Updated the package citation to version 0.3.1.

No changes were made to the statistical calculations, compiled code, model
defaults, or numerical optimization procedures.

## Test environments

The GitHub Actions workflow is configured to check the package on:

* macOS, R release
* Windows, R release
* Ubuntu, R devel
* Ubuntu, R release
* Ubuntu, R oldrel-1

The workflow uses `R CMD check --as-cran`. OpenMP and common BLAS thread counts
are restricted to respect CRAN's parallel-use policies.

## R CMD check results

0 errors | 0 warnings | 0 notes

