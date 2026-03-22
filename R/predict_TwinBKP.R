#' @name predict.TwinBKP
#'
#' @title Predict from a Fitted TwinBKP Model
#'
#' @description
#' Prediction method for TwinBKP. For each prediction point, the function runs:
#'
#' 1. **Local point selection**: use a kd-tree to find \code{l} nearest neighbors
#'    from the training set as local points.
#' 2. **Wendland kernel hyperparameter**: compute the coverage radius
#'    \eqn{\hat{\theta}_l} using Equation (14) as the single local-kernel
#'    hyperparameter.
#' 3. **Validation set**: randomly sample \code{v = 2g} points from non-global points.
#' 4. **Mixing weight \eqn{\lambda}**: minimize the loss of the mixed kernel
#'    \eqn{\lambda K_g + (1-\lambda) K_l} on the validation set to obtain the
#'    optimal \eqn{\lambda}.
#' 5. **Posterior prediction**: perform BKP posterior updating with the mixed kernel
#'    and return posterior mean/variance/confidence interval.
#'
#' @param object A \code{"TwinBKP"} object returned by \code{fit_TwinBKP()}.
#' @param Xnew Prediction input matrix (unnormalized), with dimension \code{n_new x d}.
#' @param l Positive integer. Number of local neighbors per prediction point.
#'   Default is \code{50}.
#' @param v Positive integer. Validation set size. Default is \code{2 * object$g}.
#' @param CI_level Numeric confidence level. Default is \code{0.95}.
#' @param threshold Classification threshold (only used for classification with
#'   \code{m = 1}). Default is \code{0.5}.
#' @param ... Unused.
#'
#' @return A list of class \code{"predict_TwinBKP"} containing:
#' \describe{
#'   \item{\code{mean}}{Posterior predictive mean vector (length \code{n_new}).}
#'   \item{\code{variance}}{Posterior predictive variance vector.}
#'   \item{\code{lower, upper}}{Lower/upper confidence interval vectors.}
#'   \item{\code{lambda}}{Mixing weight \eqn{\lambda} for each prediction point
#'   (length \code{n_new}).}
#'   \item{\code{theta_l}}{Local kernel hyperparameter for each prediction point
#'   (length \code{n_new}).}
#'   \item{\code{local_idx}}{List of selected local training-row indices for each
#'   prediction point.}
#'   \item{\code{Xnew}}{Original prediction coordinates.}
#'   \item{\code{Xnew_norm}}{Normalized prediction coordinates.}
#' }
#'
#' @seealso \code{\link{fit_TwinBKP}}
#' @examples
#'
#'   # ============================================================== #
#'   # ======================= TwinBKP Examples ====================== #
#'   # ============================================================== #
#'
#'   #-------------------------- 1D Example ---------------------------
#'   set.seed(123)
#'
#'   # Define true success probability function
#'   true_pi_fun <- function(x) {
#'     (1 + exp(-x^2) * cos(10 * (1 - exp(-x)) / (1 + exp(-x)))) / 2
#'   }
#'
#'   n <- 100
#'   Xbounds <- matrix(c(-2, 2), nrow = 1)
#'   X <- tgp::lhs(n = n, rect = Xbounds)
#'   true_pi <- true_pi_fun(X)
#'   m <- sample(100, n, replace = TRUE)
#'   y <- rbinom(n, size = m, prob = true_pi)
#'
#'   # Fit TwinBKP model (global stage only)
#'   model1 <- fit_TwinBKP(X, y, m, Xbounds = Xbounds, g = 10) 
#'
#'   # Prediction on new data
#'   Xnew <- matrix(seq(-2, 2, length.out = 10), ncol = 1)
#'   predict(model1, Xnew = Xnew, l = 10)
#'
#'
#'   #-------------------------- 2D Example ---------------------------
#'   set.seed(123)
#'
#'   # Define 2D latent function and probability transformation
#'   true_pi_fun <- function(X) {
#'     if (is.null(nrow(X))) X <- matrix(X, nrow = 1)
#'     m <- 8.6928
#'     s <- 2.4269
#'     x1 <- 4 * X[, 1] - 2
#'     x2 <- 4 * X[, 2] - 2
#'     a <- 1 + (x1 + x2 + 1)^2 *
#'       (19 - 14 * x1 + 3 * x1^2 - 14 * x2 + 6 * x1 * x2 + 3 * x2^2)
#'     b <- 30 + (2 * x1 - 3 * x2)^2 *
#'       (18 - 32 * x1 + 12 * x1^2 + 48 * x2 - 36 * x1 * x2 + 27 * x2^2)
#'     f <- log(a * b)
#'     f <- (f - m) / s
#'     pnorm(f)
#'   }
#'
#'   n <- 50
#'   Xbounds <- matrix(c(0, 0, 1, 1), nrow = 2)
#'   X <- tgp::lhs(n = n, rect = Xbounds)
#'   true_pi <- true_pi_fun(X)
#'   m <- sample(100, n, replace = TRUE)
#'   y <- rbinom(n, size = m, prob = true_pi)
#'
#'   # Fit TwinBKP model
#'   model2 <- fit_TwinBKP(X, y, m, Xbounds = Xbounds, g = 12) 
#'
#'   # Prediction on new data
#'   x1 <- seq(Xbounds[1, 1], Xbounds[1, 2], length.out = 8)
#'   x2 <- seq(Xbounds[2, 1], Xbounds[2, 2], length.out = 8)
#'   Xnew <- expand.grid(x1 = x1, x2 = x2)
#'   predict(model2, Xnew = Xnew, l = 12)
#' 
#' @importFrom stats optimise
#' @export
predict.TwinBKP <- function(
    object,
    Xnew,
    l         = 50L,
    v         = NULL,
    CI_level  = 0.95,
    threshold = 0.5,
    ...
) {


  Xnorm        <- object$Xnorm
  y            <- object$y
  m_train      <- object$m
  Xbounds      <- object$Xbounds
  global_idx   <- object$global_idx
  Xnorm_global <- object$Xnorm_global
  y_global     <- object$y_global
  m_global     <- object$m_global
  theta_global <- object$theta_global
  kernel       <- object$kernel
  isotropic    <- object$isotropic
  prior        <- object$prior
  r0           <- object$r0
  p0           <- object$p0
  loss_type    <- object$loss
  g            <- object$g

  n <- nrow(Xnorm)
  d <- ncol(Xnorm)
 

  if (is.null(nrow(Xnew))) Xnew <- matrix(Xnew, nrow = 1)
  Xnew <- as.matrix(Xnew)
  if (!is.numeric(Xnew)) stop("'Xnew' must be numeric.")
  if (ncol(Xnew) != d)   stop("'Xnew' must have the same number of columns as training 'X'.")
  if (anyNA(Xnew))       stop("'Xnew' contains NA values.")

  if (!is.numeric(CI_level) || CI_level <= 0 || CI_level >= 1) {
    stop("'CI_level' must be strictly between 0 and 1.")
  }
  if (!is.numeric(threshold) || threshold <= 0 || threshold >= 1) {
    stop("'threshold' must be strictly between 0 and 1.")
  }

  # 归一化 Xnew 到 [0,1]^d
  Xnew_norm <- sweep(Xnew, 2, Xbounds[, 1], "-")
  Xnew_norm <- sweep(Xnew_norm, 2, Xbounds[, 2] - Xbounds[, 1], "/")

  n_new <- nrow(Xnew_norm)


  if (!is.numeric(l) || length(l) != 1 || l <= 0) stop("'l' must be a positive integer.")
  l <- as.integer(l)
  l_eff <- min(l, n)

  if (is.null(v)) v <- 2L * g  # 默认 v = 2g
  if (!is.numeric(v) || length(v) != 1 || v <= 0) stop("'v' must be a positive integer.")
  v <- as.integer(v)

  knn_result <- get_knnx_nanoflann_rcpp(
    data = Xnorm,
    query = Xnew_norm,
    k = l_eff
  )
  local_idx_mat  <- knn_result$nn.index
  local_dist_mat <- knn_result$nn.dist

  knn_global <- get_knnx_nanoflann_rcpp(
    data = Xnorm_global,
    query = Xnorm,
    k = 1L
  )
  theta_l <- max(knn_global$nn.dist)

  q_wend <- floor(d / 2) + 3

  non_global_idx <- setdiff(seq_len(n), global_idx)

  v_eff <- min(v, length(non_global_idx))
  if (v_eff < v) {
    warning(sprintf(
      "Only %d non-global points available; validation set size reduced from %d to %d.",
      length(non_global_idx), v, v_eff
    ))
  }

  val_idx   <- sample(non_global_idx, size = v_eff, replace = FALSE)
  Xnorm_val <- Xnorm[val_idx, , drop = FALSE]
  y_val     <- y[val_idx, , drop = FALSE]
  m_val     <- m_train[val_idx, , drop = FALSE]


  K_g_val <- kernel_matrix(
    X = Xnorm_val,
    theta = theta_global,
    kernel = kernel,
    isotropic = isotropic
  )

  .wendland_kernel <- function(X1, X2, theta) {
    n1 <- nrow(X1); n2 <- nrow(X2)
    K  <- matrix(0, n1, n2)
    for (i in seq_len(n1)) {
      r <- sqrt(rowSums((X2 - matrix(X1[i, ], nrow = n2, ncol = ncol(X1), byrow = TRUE))^2))
      u <- r / theta
      K[i, ] <- (q_wend * u + 1) * pmax(0, 1 - u)^q_wend
    }
    K
  }

  .mixed_loss <- function(lambda, K_g, K_l, y_v, m_v, alpha0_v, beta0_v) {
    lambda  <- pmin(pmax(lambda, 0), 1)
    K_mix <- lambda * K_g_val + (1 - lambda) * K_l_val

    if (loss_type == "brier") {
      loss_fun_brier_bkp_rcpp(K_mix, as.numeric(y_v), as.numeric(m_v),
                               as.numeric(alpha0_v), as.numeric(beta0_v))
    } else {
      loss_fun_logloss_bkp_rcpp(K_mix, as.numeric(y_v), as.numeric(m_v),
                                 as.numeric(alpha0_v), as.numeric(beta0_v))
    }
  }

  pred_mean     <- numeric(n_new)
  pred_var      <- numeric(n_new)
  pred_lower    <- numeric(n_new)
  pred_upper    <- numeric(n_new)
  pred_lambda   <- numeric(n_new)
  pred_theta_l  <- rep(theta_l, n_new)
  local_idx_out <- vector("list", n_new)

  for (i in seq_len(n_new)) {

    loc_idx  <- local_idx_mat[i, ]
    Xnorm_loc <- Xnorm[loc_idx, , drop = FALSE]
    y_loc     <- y[loc_idx, , drop = FALSE]
    m_loc     <- m_train[loc_idx, , drop = FALSE]
    local_idx_out[[i]] <- loc_idx

    K_l_val <- .wendland_kernel(
      X1 = Xnorm_val,
      X2 = Xnorm_val,
      theta = theta_l
    )

    prior_val <- get_prior(
      prior = prior, model = "BKP",
      r0 = r0, p0 = p0,
      y = y_val, m = m_val,
      K = K_g_val
    )
    alpha0_val <- prior_val$alpha0
    beta0_val  <- prior_val$beta0

    opt_lambda <- optimise(
      f        = .mixed_loss,
      interval = c(0, 1),
      K_g      = K_g_val,
      K_l      = K_l_val,
      y_v      = y_val,
      m_v      = m_val,
      alpha0_v = alpha0_val,
      beta0_v  = beta0_val,
      maximum  = FALSE
    )
    lambda_i <- opt_lambda$minimum
    pred_lambda[i] <- lambda_i

    K_g_star <- kernel_matrix(
      X        = matrix(Xnew_norm[i, ], nrow = 1),
      Xprime   = Xnorm_global,
      theta    = theta_global,
      kernel   = kernel,
      isotropic = isotropic
    )  # (1 x g)

    K_l_star <- .wendland_kernel(
      X1    = matrix(Xnew_norm[i, ], nrow = 1),
      X2    = Xnorm_loc,
      theta = theta_l
    )  # (1 x l)
    prior_g <- get_prior(
      prior = prior, model = "BKP",
      r0 = r0, p0 = p0,
      y = y_global, m = m_global,
      K = kernel_matrix(Xnorm_global, theta = theta_global,
                        kernel = kernel, isotropic = isotropic)
    )

    K_ll <- .wendland_kernel(Xnorm_loc, Xnorm_loc, theta_l)
    prior_l <- get_prior(
      prior = prior, model = "BKP",
      r0 = r0, p0 = p0,
      y = y_loc, m = m_loc,
      K = K_ll
    )

    # alpha_n = alpha0 + lambda * K_g * y_g + (1-lambda) * K_l * y_l
    alpha_n_i <- lambda_i * (as.numeric(prior_g$alpha0) +
                               as.numeric(K_g_star %*% as.numeric(y_global))) +
      (1 - lambda_i) * (as.numeric(prior_l$alpha0) +
                          as.numeric(K_l_star %*% as.numeric(y_loc)))

    beta_n_i  <- lambda_i * (as.numeric(prior_g$beta0) +
                               as.numeric(K_g_star %*% as.numeric(m_global - y_global))) +
      (1 - lambda_i) * (as.numeric(prior_l$beta0) +
                          as.numeric(K_l_star %*% as.numeric(m_loc - y_loc)))

    eps <- 1e-10
    ab_sum <- max(alpha_n_i + beta_n_i, eps)

    pred_mean[i] <- alpha_n_i / ab_sum
    pred_mean[i] <- min(max(pred_mean[i], eps), 1 - eps)
    pred_var[i]  <- pred_mean[i] * (1 - pred_mean[i]) / (ab_sum + 1)

    pred_lower[i] <- suppressWarnings(qbeta((1 - CI_level) / 2, alpha_n_i, beta_n_i))
    pred_upper[i] <- suppressWarnings(qbeta((1 + CI_level) / 2, alpha_n_i, beta_n_i))
  }

  result <- list(
    Xnew      = Xnew,          
    Xnew_norm = Xnew_norm,   
    mean      = pred_mean,   
    variance  = pred_var,    
    lower     = pred_lower,  
    upper     = pred_upper,  
    lambda    = pred_lambda, 
    theta_l   = pred_theta_l,
    theta_global = theta_global,
    local_idx = local_idx_out, 
    CI_level  = CI_level,
    l         = l_eff,
    v         = v_eff
  )

  if (all(m_train == 1)) {
    result$class     <- ifelse(pred_mean > threshold, 1, 0)
    result$threshold <- threshold
  }

  class(result) <- "predict_TwinBKP"
  return(result)
}