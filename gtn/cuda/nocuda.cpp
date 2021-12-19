#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>

#include "gtn/graph.h"
#include "gtn/cuda/cuda.h"
#include "gtn/cuda/functions.h"

namespace gtn {

namespace cuda {

bool isAvailable() {
  return false;
}

int deviceCount() {
  throw std::logic_error("[cuda::deviceCount] CUDA not available.");
}

int getDevice() {
  throw std::logic_error("[cuda::getDevice] CUDA not available.");
}

void setDevice(int device) {
  throw std::logic_error("[cuda::getDevice] CUDA not available.");
}

namespace detail {

void add(const float* a, const float* b, float* out, size_t size, bool isCuda) {
  if (isCuda) {
    throw std::logic_error("[cuda:add] CUDA not available.");
  } else {
    std::transform(a, a + size, b, out, std::plus<>());
  }
}

float* ones(size_t size, int device) {
  throw std::logic_error("[cuda::ones] CUDA not available.");
}

void copy(void* dst, const void* src, size_t size) {
  std::memcpy(dst, src, size);
}

void* allocate(size_t size, int device) {
  throw std::logic_error("[cuda::allocate] CUDA not available.");
}

void free(void* ptr) {
  throw std::logic_error("[cuda::free] CUDA not available.");
}

} // namespace detail

Graph negate(const Graph& g) {
  throw std::logic_error("[cuda::negate] CUDA not available.");
}

Graph add(const Graph& g1, const Graph& g2) {
  throw std::logic_error("[cuda::add] CUDA not available.");
}

Graph subtract(const Graph& g1, const Graph& g2) {
  throw std::logic_error("[cuda::subtract] CUDA not available.");
}

Graph concat(const std::vector<Graph>& graphs) {
  throw std::logic_error("[cuda::concat] CUDA not available.");
}

Graph closure(const Graph& g) {
  throw std::logic_error("[cuda::closure] CUDA not available.");
}

Graph union_(const std::vector<Graph>& graphs) {
  throw std::logic_error("[cuda::union_] CUDA not available.");
}

Graph intersect(const Graph& g1, const Graph& g2) {
  throw std::logic_error("[cuda::remove] CUDA not available.");
}

Graph compose(const Graph& g1, const Graph& g2) {
  throw std::logic_error("[cuda::remove] CUDA not available.");
}

Graph remove(const Graph& g, int ilabel, int olabel) {
  throw std::logic_error("[cuda::remove] CUDA not available.");
}

Graph forwardScore(const Graph& g) {
  throw std::logic_error("[cuda::forwardScore] CUDA not available.");
}

Graph viterbiScore(const Graph& g) {
  throw std::logic_error("[cuda::viterbiScore] CUDA not available.");
}

Graph viterbiPath(const Graph& g) {
  throw std::logic_error("[cuda::viterbiPath] CUDA not available.");
}


} // namespace cuda
} // namespace gtn
