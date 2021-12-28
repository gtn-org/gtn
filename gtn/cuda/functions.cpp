#include "gtn/cuda/functions.h"

namespace gtn {
namespace cuda {

namespace detail {
  Graph compose(const Graph& g1, const Graph& g2);
  Graph shortestDistance(const Graph& g, bool tropical);
} // namespace detail

Graph concat(const std::vector<Graph>& graphs) {
  throw std::logic_error("[cuda::concat] GPU function not implemented.");
}

Graph closure(const Graph& g) {
  throw std::logic_error("[cuda::closure] GPU function not implemented.");
}

Graph union_(const std::vector<Graph>& graphs) {
  throw std::logic_error("[cuda::union_] GPU function not implemented.");
}

Graph intersect(const Graph& g1, const Graph& g2) {
  return cuda::detail::compose(g1, g2);
}

Graph compose(const Graph& g1, const Graph& g2) {
  return cuda::detail::compose(g1, g2);
}

Graph remove(const Graph& g, int ilabel, int olabel) {
  throw std::logic_error("[cuda::remove] GPU function not implemented.");
}

Graph forwardScore(const Graph& g) {
  return cuda::detail::shortestDistance(g, false);
}

Graph viterbiScore(const Graph& g) {
  return cuda::detail::shortestDistance(g, true);
}

Graph viterbiPath(const Graph& g) {
  throw std::logic_error("[cuda::viterbiPath] GPU function not implemented.");
}

} // namespace cuda
} // namespace gtn
