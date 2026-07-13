
# BKP: Beta Kernel Process Modeling <img src="man/figures/logo.png" align="right" height="140"/>

<!-- badges: start -->

[![Ask
DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Jiangyan-Zhao/BKP)
[![CRAN
status](https://www.r-pkg.org/badges/version/BKP)](https://cran.r-project.org/package=BKP)
![Total downloads](https://cranlogs.r-pkg.org/badges/grand-total/BKP)
[![R-CMD-check](https://github.com/Jiangyan-Zhao/BKP/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/Jiangyan-Zhao/BKP/actions/workflows/R-CMD-check.yaml)
[![Codecov test
coverage](https://codecov.io/gh/Jiangyan-Zhao/BKP/graph/badge.svg)](https://app.codecov.io/gh/Jiangyan-Zhao/BKP)
<!-- badges: end -->

**BKP** implements Beta Kernel Process models for nonparametric
estimation of covariate-dependent binomial probabilities. It uses
kernel-weighted conjugate updates to construct closed-form, pointwise
Beta posterior summaries for binary and aggregated binomial responses,
without latent-variable augmentation, numerical posterior approximation,
or MCMC sampling.

The package also implements the **Dirichlet Kernel Process (DKP)** for
categorical and multinomial responses. Scalable global-local
approximations are available through **TwinBKP** and **TwinDKP**, which
combine twinning-selected global subsets with location-specific
nearest-neighbour updates for larger datasets.

## Features

- Bayesian-inspired kernel smoothing for binary, binomial, categorical,
  and multinomial data
- Closed-form, pointwise Beta and Dirichlet posterior updates
- Posterior means, variances, credible intervals, fitted values,
  quantiles, and simulation
- Gaussian, Matérn 5/2, Matérn 3/2, and compactly supported Wendland
  kernels
- Isotropic and anisotropic kernel hyperparameters
- LOOCV-based hyperparameter tuning using the Brier score or log-loss
- Optional Shepard effective-sample-size calibration for BKP and DKP
  models
- Scalable TwinBKP and TwinDKP approximations using twinning-based
  global-local updates
- S3 methods for `predict()`, `simulate()`, `summary()`, `plot()`,
  `print()`, `fitted()`, `parameter()`, and `quantile()`

Shepard effective-sample-size calibration is available for `fit_BKP()`
and `fit_DKP()`. It is not applied to the TwinBKP and TwinDKP
approximations.

## Installation

Install the stable version from
[CRAN](https://CRAN.R-project.org/package=BKP):

``` r
install.packages("BKP")
```

Install the development version from
[GitHub](https://github.com/Jiangyan-Zhao/BKP):

``` r
# install.packages("pak")
pak::pak("Jiangyan-Zhao/BKP")
```

The GitHub development version may contain changes that have not yet
been released on CRAN.

## Quick example

``` r
library(BKP)

set.seed(123)

true_pi_fun <- function(x) {
  (1 + exp(-x^2) * cos(10 * (1 - exp(-x)) / (1 + exp(-x)))) / 2
}

n <- 30
Xbounds <- matrix(c(-2, 2), nrow = 1)
X <- tgp::lhs(n = n, rect = Xbounds)

true_pi <- true_pi_fun(X)
m <- sample(100, n, replace = TRUE)
y <- rbinom(n, size = m, prob = true_pi)

fit <- fit_BKP(X, y, m, Xbounds = Xbounds)

summary(fit)
plot(fit)

Xnew <- matrix(seq(-2, 2, length.out = 10), ncol = 1)
pred <- predict(fit, Xnew = Xnew)
pred
```

For categorical or multinomial responses, use `fit_DKP()`. For scalable
global-local approximations, use `fit_TwinBKP()` or `fit_TwinDKP()`.

## Scalable global-local models

For larger binomial datasets, a TwinBKP model can be fitted using the
same core data inputs:

``` r
twin_fit <- fit_TwinBKP(X, y, m, Xbounds = Xbounds)

summary(twin_fit)

twin_pred <- predict(twin_fit, Xnew = Xnew)

twin_pred
```

TwinBKP and TwinDKP use a twinning-selected global subset to represent
the overall data distribution and location-specific nearest neighbours
to recover local information.

## Documentation and reproducibility

The statistical foundations, implementation details, and worked examples
are described in:

- [**BKP software paper
  (PDF)**](https://github.com/Jiangyan-Zhao/BKP-paper/blob/master/paper/TR_BKP.pdf)
- [**BKP-paper reproducibility
  repository**](https://github.com/Jiangyan-Zhao/BKP-paper)

The reproducibility repository contains the manuscript source files,
analysis scripts, data-processing code, and materials used to generate
the examples and figures in the paper.

## Citing

If you use **BKP** in your work, please cite both the software paper and
the R package.

- **Software paper** Zhao, J., Qing, K., and Xu, J. (2025). *BKP: An R
  Package for Beta Kernel Process Modeling.* arXiv:2508.10447.
  <https://arxiv.org/abs/2508.10447>.

- **R package** Zhao, J., Qing, K., and Xu, J. (2026). *BKP: Beta Kernel
  Process Modeling.* R package version 0.3.1.
  <https://cran.r-project.org/package=BKP>.

Citation information can also be obtained directly within R:

``` r
citation("BKP")
```

## Development

The BKP package is under active development. Bug reports, feature
requests, and contributions are welcome through GitHub issues or pull
requests:

- [Report an issue](https://github.com/Jiangyan-Zhao/BKP/issues)
- [Open a pull request](https://github.com/Jiangyan-Zhao/BKP/pulls)
