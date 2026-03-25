#include <RcppArmadillo.h>
#include <limits>
#include <cmath>
#include <algorithm>
#include <vector>

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

// ---- L-BFGS-B forward declaration (must appear before use) ----
static Rcpp::List lbfgsb_refine(
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
);

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

      List ref = lbfgsb_refine(
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

      List ref = lbfgsb_refine(
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

// ---- L-BFGS-B helpers ----
static inline arma::vec project_box(const arma::vec& x, const arma::vec& lower, const arma::vec& upper) {
  arma::vec out = x;
  for (arma::uword i = 0; i < out.n_elem; ++i) {
    out[i] = clamp_double(out[i], lower[i], upper[i]);
  }
  return out;
}

static inline double projected_grad_inf_norm(
    const arma::vec& x, const arma::vec& g,
    const arma::vec& lower, const arma::vec& upper
) {
  double mx = 0.0;
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    double pg = g[i];
    const bool at_lower = (x[i] <= lower[i] + 1e-12);
    const bool at_upper = (x[i] >= upper[i] - 1e-12);

    // Feasible projected gradient for box constraints
    if ((at_lower && g[i] > 0.0) || (at_upper && g[i] < 0.0)) {
      pg = 0.0;
    }
    mx = std::max(mx, std::abs(pg));
  }
  return mx;
}

static arma::vec finite_diff_grad(
    const arma::vec& x,
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
    const arma::vec& upper
) {
  const arma::uword p = x.n_elem;
  arma::vec g(p, arma::fill::zeros);

  const double f0 = eval_bkp_loss_from_gamma(x, Xnorm, y, m, prior, r0, p0, loss, kernel, isotropic);

  for (arma::uword j = 0; j < p; ++j) {
    const double span = std::max(1e-8, upper[j] - lower[j]);
    const double h = std::min(1e-3 * (1.0 + std::abs(x[j])), 0.25 * span);

    arma::vec xp = x, xm = x;
    xp[j] = clamp_double(x[j] + h, lower[j], upper[j]);
    xm[j] = clamp_double(x[j] - h, lower[j], upper[j]);

    if (xp[j] > x[j] && xm[j] < x[j]) {
      const double fp = eval_bkp_loss_from_gamma(xp, Xnorm, y, m, prior, r0, p0, loss, kernel, isotropic);
      const double fm = eval_bkp_loss_from_gamma(xm, Xnorm, y, m, prior, r0, p0, loss, kernel, isotropic);
      g[j] = (fp - fm) / (xp[j] - xm[j]);
    } else if (xp[j] > x[j]) {
      const double fp = eval_bkp_loss_from_gamma(xp, Xnorm, y, m, prior, r0, p0, loss, kernel, isotropic);
      g[j] = (fp - f0) / (xp[j] - x[j]);
    } else if (xm[j] < x[j]) {
      const double fm = eval_bkp_loss_from_gamma(xm, Xnorm, y, m, prior, r0, p0, loss, kernel, isotropic);
      g[j] = (f0 - fm) / (x[j] - xm[j]);
    } else {
      g[j] = 0.0;
    }
  }
  return g;
}

static Rcpp::List lbfgsb_refine(
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
  arma::vec x = project_box(gamma_init, lower, upper);
  double fx = eval_bkp_loss_from_gamma(x, Xnorm, y, m, prior, r0, p0, loss, kernel, isotropic);
  arma::vec g = finite_diff_grad(x, Xnorm, y, m, prior, r0, p0, loss, kernel, isotropic, lower, upper);

  const int m_hist = 7;
  std::vector<arma::vec> s_hist, y_hist;
  std::vector<double> rho_hist;
  s_hist.reserve(m_hist);
  y_hist.reserve(m_hist);
  rho_hist.reserve(m_hist);

  const double pg_tol = 1e-5;
  const double c1 = 1e-4;

  for (int it = 0; it < max_iter; ++it) {
    if (projected_grad_inf_norm(x, g, lower, upper) < pg_tol) break;

    // ---- two-loop recursion ----
    arma::vec q = g;
    const int k = static_cast<int>(s_hist.size());
    std::vector<double> alpha(k, 0.0);

    for (int i = k - 1; i >= 0; --i) {
      alpha[i] = rho_hist[i] * arma::dot(s_hist[i], q);
      q -= alpha[i] * y_hist[i];
    }

    arma::vec r = q;
    if (k > 0) {
      const arma::vec& sk = s_hist.back();
      const arma::vec& yk = y_hist.back();
      const double yy = arma::dot(yk, yk);
      const double sy = arma::dot(sk, yk);
      const double H0 = (yy > 0.0) ? (sy / yy) : 1.0;
      r *= H0;
    }

    for (int i = 0; i < k; ++i) {
      const double beta = rho_hist[i] * arma::dot(y_hist[i], r);
      r += s_hist[i] * (alpha[i] - beta);
    }

    arma::vec pdir = -r;
    if (arma::dot(pdir, g) >= 0.0) pdir = -g; // fallback to steepest descent

    // ---- Armijo backtracking with projection ----
    double step = 1.0;
    arma::vec x_new = x;
    double f_new = fx;
    bool accepted = false;

    const double gd = arma::dot(g, pdir);

    for (int ls = 0; ls < 25; ++ls) {
      arma::vec trial = project_box(x + step * pdir, lower, upper);

      if (arma::norm(trial - x, "inf") < 1e-12) {
        step *= 0.5;
        continue;
      }

      const double f_trial = eval_bkp_loss_from_gamma(trial, Xnorm, y, m, prior, r0, p0, loss, kernel, isotropic);
      if (std::isfinite(f_trial) && (f_trial <= fx + c1 * step * gd)) {
        x_new = trial;
        f_new = f_trial;
        accepted = true;
        break;
      }
      step *= 0.5;
    }

    if (!accepted) break;

    arma::vec g_new = finite_diff_grad(x_new, Xnorm, y, m, prior, r0, p0, loss, kernel, isotropic, lower, upper);

    arma::vec s = x_new - x;
    arma::vec yk = g_new - g;
    const double ys = arma::dot(yk, s);

    if (ys > 1e-10) {
      const double rho = 1.0 / ys;
      if (static_cast<int>(s_hist.size()) == m_hist) {
        s_hist.erase(s_hist.begin());
        y_hist.erase(y_hist.begin());
        rho_hist.erase(rho_hist.begin());
      }
      s_hist.push_back(s);
      y_hist.push_back(yk);
      rho_hist.push_back(rho);
    }

    x = x_new;
    fx = f_new;
    g = g_new;
  }

  return List::create(
    Named("gamma") = x,
    Named("value") = fx
  );
}