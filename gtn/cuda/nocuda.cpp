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
  throw std::logic_error("[cuda::setDevice] CUDA not available.");
}

void synchronize() {
  throw std::logic_error("[cuda::synchronize] CUDA not available.");
}

void synchronize(int device) {
  throw std::logic_error("[cuda::synchronize] CUDA not available.");
}

void synchronizeStream() {
  throw std::logic_error("[cuda::synchronizeStream] CUDA not available.");
}

Event::Event() {
  throw std::logic_error("[cuda::Event] CUDA not available.");
}

Event::~Event() { }
void Event::record() { }
void Event::synchronize() { }
void Event::wait() { }

namespace detail {

void add(const float* a, const float* b, float* out, size_t size) {
  throw std::logic_error("[cuda::detail::add] CUDA not available.");
}

void subtract(const float* a, const float* b, float* out, size_t size) {
  throw std::logic_error("[cuda::detail::subtract] CUDA not available.");
}

void negate(const float* in, float* out, size_t size) {
  throw std::logic_error("[cuda::detail::negate] CUDA not available.");
}

void copy(void* dst, const void* src, size_t size) {
  std::memcpy(dst, src, size);
}

void* allocate(size_t size, int device) {
  throw std::logic_error("[cuda::detail::allocate] CUDA not available.");
}

void free(void* ptr) {
  throw std::logic_error("[cuda::detail::free] CUDA not available.");
}

bool equal(const float* lhs, const float* rhs, size_t size) {
  throw std::logic_error("[cuda::detail::equal] CUDA not available.");
}
bool equal(const int* lhs, const int* rhs, size_t size) {
  throw std::logic_error("[cuda::detail::equal] CUDA not available.");
}
bool equal(const bool* lhs, const bool* rhs, size_t size) {
  throw std::logic_error("[cuda::detail::equal] CUDA not available.");
}

void fill(float* dst, float val, size_t size) {
  throw std::logic_error("[cuda::detail::fill] CUDA not available.");
}
void fill(int* dst, int val, size_t size) {
  throw std::logic_error("[cuda::detail::fill] CUDA not available.");
}
void fill(bool* dst, bool val, size_t size) {
  throw std::logic_error("[cuda::detail::fill] CUDA not available.");
}


} // namespace detail

Graph negate(const Graph& g) {
  throw std::logic_error("[cuda::negate] CUDA not available.");
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

Graph scalarGraph(float val, bool calcGrad, Device device) {
  throw std::logic_error("[cuda::scalarGraph] CUDA not available.");
}

Graph linearGraph(int M, int N, bool calcGrad, Device device) {
  throw std::logic_error("[cuda::linearGraph] CUDA not available.");
}


} // namespace cuda
} // namespace gtn
