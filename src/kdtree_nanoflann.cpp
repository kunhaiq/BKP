#include <Rcpp.h>

// Route nanoflann assertions to R errors (avoid stderr/abort paths)
#ifndef NANOFLANN_ASSERT
#define NANOFLANN_ASSERT(x)                                                     \
  do {                                                                          \
    if (!(x)) Rcpp::stop("nanoflann assertion failed: " #x);                   \
  } while (0)
#endif

#include "../inst/nanoflann.hpp" 

using namespace Rcpp;
 
struct MatrixRowAdaptor {
  const NumericMatrix& mat;
  MatrixRowAdaptor(const NumericMatrix& m) : mat(m) {}

  inline size_t kdtree_get_point_count() const { return mat.nrow(); }

  inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
    return mat(idx, dim);
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
  nanoflann::L2_Simple_Adaptor<double, MatrixRowAdaptor>,
  MatrixRowAdaptor,
  -1,
  size_t
>;

// [[Rcpp::export]]
List get_knnx_nanoflann_rcpp(NumericMatrix data, NumericMatrix query, int k) {
  const int n_data = data.nrow();
  const int d = data.ncol();
  const int n_query = query.nrow();

  if (n_data <= 0) stop("'data' has no rows.");
  if (query.ncol() != d) stop("'query' must have same ncol as 'data'.");
  if (k <= 0) stop("'k' must be positive.");

  const size_t k_eff = std::min((size_t)k, (size_t)n_data);

  try {
  MatrixRowAdaptor adaptor(data);
  KDTree index(d, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  index.buildIndex();

  IntegerMatrix nn_index(n_query, k_eff);
  NumericMatrix nn_dist(n_query, k_eff);

  std::vector<size_t> ret_index(k_eff);
  std::vector<double> out_dist_sqr(k_eff);
  std::vector<double> q(d);

  for (int i = 0; i < n_query; ++i) {
    for (int j = 0; j < d; ++j) q[j] = query(i, j);

    index.knnSearch(q.data(), k_eff, ret_index.data(), out_dist_sqr.data());

    for (size_t t = 0; t < k_eff; ++t) {
      nn_index(i, t) = static_cast<int>(ret_index[t]) + 1; // R 1-based
      nn_dist(i, t) = std::sqrt(out_dist_sqr[t]);          // 
    }
  }

  return List::create(
    _["nn.index"] = nn_index,
    _["nn.dist"] = nn_dist
  );
  } catch (const std::exception& e) {
    Rcpp::stop("nanoflann error: %s", e.what());
  } catch (...) {
    Rcpp::stop("nanoflann unknown error.");
}
}

