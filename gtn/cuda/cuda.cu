#include <sstream>
#include <iostream>

#include "cuda.h"

namespace gtn {

namespace {

void copyHostDevice(int* dptr, const std::vector<int>& hvec, size_t size) {
  CUDA_CHECK(cudaMemcpyAsync(
    static_cast<void*>(dptr),
    static_cast<const void*>(hvec.data()),
    size * sizeof(int),
    cudaMemcpyHostToDevice));
}

void copyDeviceHost(std::vector<int>& hvec, const int* dptr, size_t size) {
  hvec.resize(size);
  CUDA_CHECK(cudaMemcpy(
    static_cast<void*>(hvec.data()),
    static_cast<const void*>(dptr),
    size * sizeof(int),
    cudaMemcpyDeviceToHost));
}

} // namespace

Graph Graph::cpu() {
  // No-op if already on CPU
  if (!sharedGraph_->isCuda) {
    return *this;
  }
  Graph g;
  g.sharedGraph_->startIds = this->start();
  g.sharedGraph_->acceptIds = this->accept();
  g.sharedGraph_->isCuda = false;
  g.sharedGraph_->compiled = true;
  g.sharedGraph_->numNodes = this->numNodes();
  g.sharedGraph_->numArcs = this->numArcs();
  g.setCalcGrad(this->calcGrad());
  auto& dd = this->sharedGraph_->deviceData;
  auto& hd = *(g.sharedGraph_);
  copyDeviceHost(hd.start, dd.start, g.numNodes());
  copyDeviceHost(hd.accept, dd.accept, g.numNodes());
  copyDeviceHost(hd.inArcOffset, dd.inArcOffset, g.numNodes());
  copyDeviceHost(hd.outArcOffset, dd.outArcOffset, g.numNodes());
  copyDeviceHost(hd.inArcs, dd.inArcs, g.numArcs());
  copyDeviceHost(hd.outArcs, dd.outArcs, g.numArcs());
  copyDeviceHost(hd.ilabels, dd.ilabels, g.numArcs());
  copyDeviceHost(hd.olabels, dd.olabels, g.numArcs());
  copyDeviceHost(hd.srcNodes, dd.srcNodes, g.numArcs());
  copyDeviceHost(hd.dstNodes, dd.dstNodes, g.numArcs());

  g.sharedWeights_->resize(g.numArcs());
  CUDA_CHECK(cudaMemcpy(
    static_cast<void*>(g.weights()),
    static_cast<const void*>(dd.weights),
    g.numArcs() * sizeof(int),
    cudaMemcpyDeviceToHost));
  return g;
}

Graph Graph::cuda() {
  // No-op if already on GPU
  if (sharedGraph_->isCuda) {
    return *this;
  }
  maybeCompile();

  Graph g;
  g.sharedGraph_->startIds = this->start();
  g.sharedGraph_->acceptIds = this->accept();
  g.sharedGraph_->isCuda = true;
  g.sharedGraph_->compiled = true;
  g.sharedGraph_->numNodes = this->numNodes();
  g.sharedGraph_->numArcs = this->numArcs(); 
  g.setCalcGrad(this->calcGrad());

  auto& hd = *(this->sharedGraph_);
  auto& dd = g.sharedGraph_->deviceData;
  dd.allocate(g.numNodes(), g.numArcs());
  copyHostDevice(dd.start, hd.start, g.numNodes());
  copyHostDevice(dd.accept, hd.accept, g.numNodes());
  copyHostDevice(dd.inArcOffset, hd.inArcOffset, g.numNodes());
  copyHostDevice(dd.outArcOffset, hd.outArcOffset, g.numNodes());
  copyHostDevice(dd.inArcs, hd.inArcs, g.numArcs());
  copyHostDevice(dd.outArcs, hd.outArcs, g.numArcs());
  copyHostDevice(dd.ilabels, hd.ilabels, g.numArcs());
  copyHostDevice(dd.olabels, hd.olabels, g.numArcs());
  copyHostDevice(dd.srcNodes, hd.srcNodes, g.numArcs());
  copyHostDevice(dd.dstNodes, hd.dstNodes, g.numArcs());
  CUDA_CHECK(cudaMemcpyAsync(
    static_cast<void*>(dd.weights),
    static_cast<const void*>(this->weights()),
    g.numArcs() * sizeof(float),
    cudaMemcpyHostToDevice));
  return g;
}

void Graph::GraphGPU::allocate(size_t numNodes, size_t numArcs) {
  CUDA_CHECK(cudaMalloc((void **)(&start), sizeof(int) * numNodes));
  CUDA_CHECK(cudaMalloc((void **)(&accept), sizeof(int) * numNodes));
  CUDA_CHECK(cudaMalloc((void **)(&inArcOffset), sizeof(int) * numNodes));
  CUDA_CHECK(cudaMalloc((void **)(&outArcOffset), sizeof(int) * numNodes));
  CUDA_CHECK(cudaMalloc((void **)(&inArcs), sizeof(int) * numArcs));
  CUDA_CHECK(cudaMalloc((void **)(&outArcs), sizeof(int) * numArcs));
  CUDA_CHECK(cudaMalloc((void **)(&ilabels), sizeof(int) * numArcs));
  CUDA_CHECK(cudaMalloc((void **)(&olabels), sizeof(int) * numArcs));
  CUDA_CHECK(cudaMalloc((void **)(&srcNodes), sizeof(int) * numArcs));
  CUDA_CHECK(cudaMalloc((void **)(&dstNodes), sizeof(int) * numArcs));
  CUDA_CHECK(cudaMalloc((void **)(&weights), sizeof(float) * numArcs));
}

Graph::GraphGPU::~GraphGPU() {
  if (start != nullptr) {
    CUDA_CHECK(cudaFree(start)); 
    CUDA_CHECK(cudaFree(accept)); 
    CUDA_CHECK(cudaFree(inArcOffset)); 
    CUDA_CHECK(cudaFree(outArcOffset)); 
    CUDA_CHECK(cudaFree(inArcs)); 
    CUDA_CHECK(cudaFree(outArcs)); 
    CUDA_CHECK(cudaFree(ilabels)); 
    CUDA_CHECK(cudaFree(olabels)); 
    CUDA_CHECK(cudaFree(srcNodes)); 
    CUDA_CHECK(cudaFree(dstNodes)); 
  }
}

namespace cuda {

bool isAvailable() {
#if defined(CUDA)
  return deviceCount() > 0;
#else
  return false;
#endif
}

int deviceCount() {
  int count;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  return count;
}

int getDevice() {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

void setDevice(int device) {
  CUDA_CHECK(cudaSetDevice(device));
}

namespace detail {

void cudaCheck(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::ostringstream ess;
    ess << '[' << file << ':' << line
        << "] CUDA error: " << cudaGetErrorString(err);
    throw std::runtime_error(ess.str());
  }
}

} // namespace detail
} // namespace cuda
} // namespace gtn
