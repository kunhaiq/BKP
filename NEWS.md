# BKP 0.3.1 (2026-07-13)

- Reordered optional arguments in `fit_BKP()` and `fit_DKP()` to group kernel specification, length-scale settings, loss/ESS options, and optimization controls more consistently. Named-argument usage is unaffected.
- Updated probability-scale plot labels across `plot.BKP()`, `plot.DKP()`, `plot.TwinBKP()`, and `plot.TwinDKP()` to use “Posterior Mean,” “Posterior Variance,” and “95% Credible Interval” instead of “Predictive Mean,” “Predictive Variance,” and “95% CI.” This change clarifies that these panels summarize posterior uncertainty for the latent probability surface rather than the predictive distribution of future responses.
- Added links to the companion [`BKP-paper`](https://github.com/Jiangyan-Zhao/BKP-paper) reproducibility repository in the package metadata and README. The repository contains the manuscript, replication code, data-processing scripts, and materials used to generate the examples and figures in the software paper.

# BKP 0.3.0 (2026-07-01)

- Added `fit_TwinDKP()` and associated S3 methods for scalable global-local Dirichlet Kernel Process modeling.
- Added `fit_TwinBKP()` with full S3 support for scalable Twin Beta Kernel Process modeling.
- Implemented Twinning-based global subset selection and kd-tree local-neighbour search.

# BKP 0.2.4 (2026-06-16)

- Improved computational efficiency by implementing kernel evaluation, prior construction, loss evaluation, and hyperparameter optimization routines in C++.
- Added support for the compactly supported Wendland kernel via `kernel = "wendland"`.
- Added optional Shepard effective-sample-size calibration for `fit_BKP(ess = "shepard")` and `fit_DKP(ess = "shepard")`, while keeping the default `ess = "none"` behavior unchanged.
- Added **ggplot2** support for plotting; `plot.BKP(..., engine = "ggplot")` now produces ggplot2-based visualizations.
- Added a package-wide `isotropic` argument, defaulting to `TRUE`, for isotropic kernels with a shared length-scale across dimensions. Set `isotropic = FALSE` to use anisotropic kernels with dimension-specific length-scales.

# BKP 0.2.3 (2025-09-22)

- Removed vignettes to avoid redundancy with the arXiv paper and to resolve CRAN thread limit issues during vignette building.

# BKP 0.2.0 (2025-09-16)

- Added `fitted()`, `parameter()`, and `quantile()` methods.
- Updated `predict()` and `simulate()` methods: both now return results for the training data by default when `Xnew` is not provided.
- Extended `plot()` method with new `dims` argument for higher-dimensional inputs.
- Added Section 5 to the vignette, presenting a real-data application on Loa loa parasite infection in North Cameroon.
- Added argument checking with informative error messages.

# BKP 0.1.1 (2025-08-19)

- Added a vignette introducing the package.
- Fixed minor bugs and improved stability.

# BKP 0.1.0 (2025-07-23)

- Initial release on CRAN.
