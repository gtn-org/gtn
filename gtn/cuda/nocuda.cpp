#include "cuda.h"

namespace gtn {

Graph Graph::cpu() {
  return this;
}

Graph Graph::cuda() {
  throw std::logic_error("[Graph::cuda] CUDA not available.");
}

void Graph::GraphGPU::allocate(size_t numNodes, size_t numArcs) { }

void Graph::GraphGPU::deepCopy(
    const Graph::GraphGPU& other, size_t numNodes, size_t numArcs, int device) {
}

Graph::GraphGPU::~GraphGPU() { }

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

Graph toDevice(const Graph& g) {
}

Graph toHost(const Graph& g) {
}

}

}
}
