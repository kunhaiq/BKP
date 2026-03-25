#include <RcppArmadillo.h>
#include <limits>
#include <cmath>
#include <algorithm>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// ---- declarations from other cpp files ----
arma::mat kernel_matrix_rcpp(SEXP X, SEXP Xprime, NumericVector theta, std::string kernel, bool isotropic);

List get_prior_bkp_noninformative_rcpp(int n);
List get_prior_bkp_fixed_rcpp(int n, double r0, double p0);
List get_prior_bkp_adaptive_rcpp(const arma::mat& K, const arma::vec& y, const arma::vec& m, double r0);

double loss_fun_brier_bkp_rcpp(
  const arma::mat& K, const arma::vec& y, const arma::vec& m,
  const arma::vec& alpha0, const arma::vec& beta0
);

double loss_fun_logloss_bkp_rcpp(
  const arma::mat& K, const arma::vec& y, const arma::vec& m,
  const arma::vec& alpha0, const arma::vec& beta0
);

static inline double clamp_double(const double x, const double lo, const double hi) {
  return std::max(lo, std::min(hi, x));
}

static double eval_bkp_loss_from_gamma(
    const arma::vec& gamma,
    const arma::mat& Xnorm,
    const arma::vec& y,
    const arma::vec& m,
    const std::string& prior,
    const double r0,
    const double p0,
    const std::string& loss,
    const std::string& kernel,
    const bool isotropic
) {
  arma::vec theta = arma::exp(std::log(10.0) * gamma);

  NumericVector theta_nv(theta.begin(), theta.end());
  arma::mat K = kernel_matrix_rcpp(wrap(Xnorm), R_NilValue, theta_nv, kernel, isotropic);
  K.diag().zeros();

  List prior_par;
  if (prior == "noninformative") {
    prior_par = get_prior_bkp_noninformative_rcpp(Xnorm.n_rows);
  } else if (prior == "fixed") {
    prior_par = get_prior_bkp_fixed_rcpp(Xnorm.n_rows, r0, p0);
  } else if (prior == "adaptive") {
    prior_par = get_prior_bkp_adaptive_rcpp(K, y, m, r0);
  } else {
    stop("Unsupported prior: " + prior);
  }

  arma::vec alpha0 = as<arma::vec>(prior_par["alpha0"]);
  arma::vec beta0  = as<arma::vec>(prior_par["beta0"]);

  double val = std::numeric_limits<double>::infinity();

  if (loss == "brier") {
    val = loss_fun_brier_bkp_rcpp(K, y, m, alpha0, beta0);
  } else if (loss == "log_loss") {
    val = loss_fun_logloss_bkp_rcpp(K, y, m, alpha0, beta0);
  } else {
    stop("Unsupported loss: " + loss);
  }

  // guard: NaN or Inf -> return large finite value so sort_index won't crash
  if (!std::isfinite(val)) return std::numeric_limits<double>::max();

  return val;
}

static List coordinate_refine(
    arma::vec gamma_init,
    const arma::mat& Xnorm,
    const arma::vec& y,
    const arma::vec& m,
    const std::string& prior,
    const double r0,
    const double p0,
    const std::string& loss,
    const std::string& kernel,
    const bool isotropic,
    const arma::vec& lower,
    const arma::vec& upper,
    const int max_iter
) {
  arma::vec g = gamma_init;
  double best_val = eval_bkp_loss_from_gamma(g, Xnorm, y, m, prior, r0, p0, loss, kernel, isotropic);

  double step = 1.0;
  const int p = static_cast<int>(g.n_elem);

  for (int it = 0; it < max_iter; ++it) {
    bool improved = false;

    for (int j = 0; j < p; ++j) {
      const double left  = std::max(lower[j], g[j] - step);
      const double right = std::min(upper[j], g[j] + step);

      double l = left;
      double r = right;

      // 1D ternary-like refinement on coordinate j
      for (int k = 0; k < 8; ++k) {
        const double m1 = l + (r - l) / 3.0;
        const double m2 = r - (r - l) / 3.0;

        arma::vec g1 = g;
        arma::vec g2 = g;
        g1[j] = m1;
        g2[j] = m2;

        const double f1 = eval_bkp_loss_from_gamma(g1, Xnorm, y, m, prior, r0, p0, loss, kernel, isotropic);
        const double f2 = eval_bkp_loss_from_gamma(g2, Xnorm, y, m, prior, r0, p0, loss, kernel, isotropic);

        if (f1 <= f2) {
          r = m2;
        } else {
          l = m1;
        }
      }

      const double cand = 0.5 * (l + r);
      arma::vec g_new = g;
      g_new[j] = clamp_double(cand, lower[j], upper[j]);

      const double val_new = eval_bkp_loss_from_gamma(g_new, Xnorm, y, m, prior, r0, p0, loss, kernel, isotropic);

      if (val_new < best_val) {
        best_val = val_new;
        g = g_new;
        improved = true;
      }
    }

    if (!improved) {
      step *= 0.6;
      if (step < 1e-3) break;
    }
  }

  return List::create(
    Named("gamma") = g,
    Named("value") = best_val
  );
}

static List generate_anisotropic_candidates(
    const int p,
    const int n_candidates,
    const double lower,
    const double upper
) {
  arma::mat cand(n_candidates, p, arma::fill::zeros);

  // first candidate = zero vector (theta = 1 for all dims)
  cand.row(0).zeros();

  if (n_candidates > 1) {
    // Latin Hypercube Sampling for remaining n_candidates-1 points
    const int n_lhs = n_candidates - 1;
    arma::mat lhs(n_lhs, p);

    for (int j = 0; j < p; ++j) {
      // generate permutation of 0..n_lhs-1
      arma::uvec perm = arma::randperm(n_lhs);
      for (int i = 0; i < n_lhs; ++i) {
        // stratified sample within each cell
        double u = (static_cast<double>(perm[i]) + R::runif(0.0, 1.0)) / static_cast<double>(n_lhs);
        lhs(i, j) = lower + u * (upper - lower);
      }
    }

    cand.rows(1, n_candidates - 1) = lhs;
  }

  return List::create(Named("cand") = cand);
}

// [[Rcpp::export]]
Rcpp::List optimize_bkp_theta_rcpp(
    const arma::mat& Xnorm,
    const arma::vec& y,
    const arma::vec& m,
    const std::string& prior,
    const double r0,
    const double p0,
    const std::string& loss,
    const std::string& kernel,
    const bool isotropic,
    const int n_grid,
    const int n_starts,
    const int max_iter,
    const double g_lower,
    const double g_upper
) {
  if (n_grid < 5) stop("n_grid must be >= 5.");
  if (n_starts < 1) stop("n_starts must be >= 1.");
  if (max_iter < 1) stop("max_iter must be >= 1.");
  if (Xnorm.n_rows == 0 || Xnorm.n_cols == 0) stop("Xnorm must be non-empty.");
  if (static_cast<int>(y.n_elem) != static_cast<int>(Xnorm.n_rows)) stop("length(y) must equal nrow(Xnorm).");
  if (static_cast<int>(m.n_elem) != static_cast<int>(Xnorm.n_rows)) stop("length(m) must equal nrow(Xnorm).");

  arma::vec gamma_opt;
  double loss_min = std::numeric_limits<double>::infinity();

  if (isotropic) {
    // ---- coarse grid in 1D ----
    arma::vec grid = arma::linspace(g_lower, g_upper, n_grid);
    arma::vec vals(n_grid, arma::fill::value(std::numeric_limits<double>::max()));

    for (int i = 0; i < n_grid; ++i) {
      arma::vec g(1);
      g[0] = grid[i];
      vals[i] = eval_bkp_loss_from_gamma(g, Xnorm, y, m, prior, r0, p0, loss, kernel, true);
    }

    arma::uvec ord = arma::sort_index(vals, "ascend");
    const int k_starts = std::min(n_starts, n_grid);

    arma::vec best_gamma(1);
    best_gamma[0] = grid[ord[0]];
    double best_val = std::numeric_limits<double>::infinity();

    for (int k = 0; k < k_starts; ++k) {
      const int idx = static_cast<int>(ord[k]);
      const int i_left  = std::max(0, idx - 1);
      const int i_right = std::min(n_grid - 1, idx + 1);

      arma::vec g0(1);
      g0[0] = grid[idx];

      arma::vec lower(1), upper(1);
      lower[0] = grid[i_left];
      upper[0] = grid[i_right];

      List ref = coordinate_refine(
        g0, Xnorm, y, m, prior, r0, p0, loss, kernel, true,
        lower, upper, max_iter
      );

      const arma::vec gk = as<arma::vec>(ref["gamma"]);
      const double vk = as<double>(ref["value"]);

      if (vk < best_val) {
        best_val = vk;
        best_gamma = gk;
      }
    }

    gamma_opt = best_gamma;
    loss_min = best_val;

  } else {
    // ---- anisotropic: random coarse candidates + multi-start local refine ----
    RNGScope scope;

    const int p = static_cast<int>(Xnorm.n_cols);
    const int n_candidates = std::max(10, n_grid);

    List cand_obj = generate_anisotropic_candidates(p, n_candidates, g_lower, g_upper);
    arma::mat cand = as<arma::mat>(cand_obj["cand"]);
    arma::vec vals(n_candidates, arma::fill::value(std::numeric_limits<double>::max()));

    for (int i = 0; i < n_candidates; ++i) {
      arma::vec g = cand.row(i).t();
      vals[i] = eval_bkp_loss_from_gamma(g, Xnorm, y, m, prior, r0, p0, loss, kernel, false);
    }

    arma::uvec ord = arma::sort_index(vals, "ascend");
    const int k_starts = std::min(n_starts, n_candidates);

    arma::vec lower(p), upper(p);
    lower.fill(g_lower);
    upper.fill(g_upper);

    arma::vec best_gamma = cand.row(ord[0]).t();
    double best_val = std::numeric_limits<double>::infinity();

    for (int k = 0; k < k_starts; ++k) {
      const int idx = static_cast<int>(ord[k]);
      arma::vec g0 = cand.row(idx).t();

      List ref = coordinate_refine(
        g0, Xnorm, y, m, prior, r0, p0, loss, kernel, false,
        lower, upper, max_iter
      );

      const arma::vec gk = as<arma::vec>(ref["gamma"]);
      const double vk = as<double>(ref["value"]);

      if (vk < best_val) {
        best_val = vk;
        best_gamma = gk;
      }
    }

    gamma_opt = best_gamma;
    loss_min = best_val;
  }

  arma::vec theta_opt = arma::exp(std::log(10.0) * gamma_opt);

  return List::create(
    Named("theta_opt") = theta_opt,
    Named("gamma_opt") = gamma_opt,
    Named("loss_min") = loss_min
  );
}