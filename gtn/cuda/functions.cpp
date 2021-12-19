#include "gtn/cuda/functions.h"

namespace gtn {
namespace cuda {

namespace detail {
  Graph compose(const Graph& g1, const Graph& g2);
} // namespace detail

Graph negate(const Graph& g) {
  throw std::logic_error("[cuda::negate] GPU function not implemented.");
}

Graph add(const Graph& g1, const Graph& g2) {
  throw std::logic_error("[cuda::add] GPU function not implemented.");
}

Graph subtract(const Graph& g1, const Graph& g2) {
  throw std::logic_error("[cuda::subtract] GPU function not implemented.");
}

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
  throw std::logic_error("[cuda::forwardScore] GPU function not implemented.");
}

Graph viterbiScore(const Graph& g) {
  throw std::logic_error("[cuda::viterbiScore] GPU function not implemented.");
}

Graph viterbiPath(const Graph& g) {
  throw std::logic_error("[cuda::viterbiPath] GPU function not implemented.");
}

} // namespace cuda
} // namespace gtn
