#include <sstream>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "cuda.h"

namespace gtn {

namespace {

void add(const float* a, const float* b, float* out, size_t size) {
  thrust::device_ptr<const float> aPtr(a);
  thrust::device_ptr<const float> bPtr(b);
  thrust::device_ptr<float> outPtr(out);
  thrust::transform(aPtr, aPtr + size, bPtr, outPtr, thrust::plus<float>());
}

void copyDeviceDevice(int* dst, const int* src, size_t size) {
  CUDA_CHECK(cudaMemcpyAsync(
    static_cast<void*>(dst),
    static_cast<const void*>(src),
    size * sizeof(int),
    cudaMemcpyDefault));
}

void copyHostDevice(int* dst, const std::vector<int>& src, size_t size) {
  CUDA_CHECK(cudaMemcpyAsync(
    static_cast<void*>(dst),
    static_cast<const void*>(src.data()),
    size * sizeof(int),
    cudaMemcpyDefault));
}

void copyDeviceHost(std::vector<int>& dst, const int* src, size_t size) {
  dst.resize(size);
  CUDA_CHECK(cudaMemcpy(
    static_cast<void*>(dst.data()),
    static_cast<const void*>(src),
    size * sizeof(int),
    cudaMemcpyDefault));
}

} // namespace

Graph Graph::cpu() const {

  // No-op if already on CPU
  if (!sharedGraph_->isCuda) {
    return *this;
  }
  Graph g;
  auto& hd = *(g.sharedGraph_);
  hd.isCuda = false;
  hd.compiled = true;
  g.setCalcGrad(this->calcGrad());
  auto& dd = this->sharedGraph_->deviceData;
  hd.numNodes = dd.numNodes;
  hd.numArcs = dd.numArcs;
  copyDeviceHost(hd.start, dd.start, g.numNodes());
  copyDeviceHost(hd.accept, dd.accept, g.numNodes());
  copyDeviceHost(hd.inArcOffset, dd.inArcOffset, g.numNodes() + 1);
  copyDeviceHost(hd.outArcOffset, dd.outArcOffset, g.numNodes() + 1);
  copyDeviceHost(hd.inArcs, dd.inArcs, g.numArcs());
  copyDeviceHost(hd.outArcs, dd.outArcs, g.numArcs());
  copyDeviceHost(hd.ilabels, dd.ilabels, g.numArcs());
  copyDeviceHost(hd.olabels, dd.olabels, g.numArcs());
  copyDeviceHost(hd.srcNodes, dd.srcNodes, g.numArcs());
  copyDeviceHost(hd.dstNodes, dd.dstNodes, g.numArcs());
  // Get the indices of the start and accept nodes
  for (int i = 0; i < hd.start.size(); i++) {
    if (hd.start[i]) {
      hd.startIds.push_back(i);
    }
  }
  for (int i = 0; i < hd.accept.size(); i++) {
    if (hd.accept[i]) {
      hd.acceptIds.push_back(i);
    }
  }

  g.sharedWeights_->weights.resize(g.numArcs());
  CUDA_CHECK(cudaMemcpy(
    static_cast<void*>(g.weights()),
    static_cast<const void*>(this->weights()),
    g.numArcs() * sizeof(int),
    cudaMemcpyDefault));
  return g;
}

Graph Graph::cuda(int device_) const {
  // No-op if already on GPU
  if (isCuda() && device() == device_) {
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
  g.sharedGraph_->device = device_;

  auto& hd = *(this->sharedGraph_);
  auto& dd = g.sharedGraph_->deviceData;
  cuda::detail::DeviceManager dm(device_);
  if (!isCuda()) {
    dd.allocate(this->numNodes(), this->numArcs());
    copyHostDevice(dd.start, hd.start, g.numNodes());
    copyHostDevice(dd.accept, hd.accept, g.numNodes());
    copyHostDevice(dd.inArcOffset, hd.inArcOffset, g.numNodes() + 1);
    copyHostDevice(dd.outArcOffset, hd.outArcOffset, g.numNodes() + 1);
    copyHostDevice(dd.inArcs, hd.inArcs, g.numArcs());
    copyHostDevice(dd.outArcs, hd.outArcs, g.numArcs());
    copyHostDevice(dd.ilabels, hd.ilabels, g.numArcs());
    copyHostDevice(dd.olabels, hd.olabels, g.numArcs());
    copyHostDevice(dd.srcNodes, hd.srcNodes, g.numArcs());
    copyHostDevice(dd.dstNodes, hd.dstNodes, g.numArcs());
    CUDA_CHECK(cudaMalloc(
      (void**)(&g.sharedWeights_->deviceWeights),
      g.numArcs() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(
      static_cast<void*>(g.weights()),
      static_cast<const void*>(this->weights()),
      g.numArcs() * sizeof(float),
      cudaMemcpyDefault));
  } else {
    dd.deepCopy(hd.deviceData, device_);
  }
  return g;
}

Graph Graph::cuda() const {
  return cuda(cuda::getDevice());
}

void Graph::addGrad(const float* other) {
  if (!isCuda()) {
    throw std::logic_error(
      "[Graph::addGrad] This addGrad is only for GPU graphs.");
  }
  if (calcGrad()) {
    std::lock_guard<std::mutex> lock(sharedGraph_->grad_lock);
    if (isGradAvailable()) {
      add(other, grad().weights(), grad().weights(), numArcs());
    } else {
      sharedGrad_->grad = std::make_unique<Graph>(false);
      sharedGrad_->grad->sharedGraph_ = sharedGraph_;
      sharedGrad_->grad->sharedWeights_->deepCopy(other, numArcs(), device());
    }
  }
}

void Graph::GraphGPU::allocate(size_t numNodes, size_t numArcs) {
  this->numNodes = numNodes;
  this->numArcs = numArcs;
  CUDA_CHECK(cudaMalloc((void**)(&start), sizeof(int) * numNodes));
  CUDA_CHECK(cudaMalloc((void**)(&accept), sizeof(int) * numNodes));
  CUDA_CHECK(cudaMalloc((void**)(&inArcOffset), sizeof(int) * (numNodes + 1)));
  CUDA_CHECK(cudaMalloc((void**)(&outArcOffset), sizeof(int) * (numNodes + 1)));
  CUDA_CHECK(cudaMalloc((void**)(&inArcs), sizeof(int) * numArcs));
  CUDA_CHECK(cudaMalloc((void**)(&outArcs), sizeof(int) * numArcs));
  CUDA_CHECK(cudaMalloc((void**)(&ilabels), sizeof(int) * numArcs));
  CUDA_CHECK(cudaMalloc((void**)(&olabels), sizeof(int) * numArcs));
  CUDA_CHECK(cudaMalloc((void**)(&srcNodes), sizeof(int) * numArcs));
  CUDA_CHECK(cudaMalloc((void**)(&dstNodes), sizeof(int) * numArcs));
}

void Graph::GraphGPU::deepCopy(
    const Graph::GraphGPU& other, int device) {
  cuda::detail::DeviceManager dm(device);
  allocate(other.numNodes, other.numArcs);
  copyDeviceDevice(start, other.start, numNodes);
  copyDeviceDevice(accept, other.accept, numNodes);
  copyDeviceDevice(inArcOffset, other.inArcOffset, numNodes + 1);
  copyDeviceDevice(outArcOffset, other.outArcOffset, numNodes + 1);
  copyDeviceDevice(inArcs, other.inArcs, numArcs);
  copyDeviceDevice(outArcs, other.outArcs, numArcs);
  copyDeviceDevice(ilabels, other.ilabels, numArcs);
  copyDeviceDevice(olabels, other.olabels, numArcs);
  copyDeviceDevice(srcNodes, other.srcNodes, numArcs);
  copyDeviceDevice(dstNodes, other.dstNodes, numArcs);
}

void Graph::GraphGPU::free() {
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

void Graph::SharedWeights::deepCopy(
    const float *src, size_t numArcs, int device) {
  cuda::detail::DeviceManager dm(device);
  CUDA_CHECK(cudaMalloc((void**)(&deviceWeights), sizeof(float) * numArcs));
  CUDA_CHECK(cudaMemcpyAsync(
    static_cast<void*>(deviceWeights),
    static_cast<const void*>(src),
    numArcs * sizeof(float),
    cudaMemcpyDefault));
}

Graph::SharedWeights::~SharedWeights() {
  if (deviceWeights != nullptr) {
    CUDA_CHECK(cudaFree(deviceWeights));
  }
}

namespace cuda {

bool isAvailable() {
  return deviceCount() > 0;
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

float* ones(size_t size, int device) {
  DeviceManager dm(device);
  float *res;
  CUDA_CHECK(cudaMalloc((void**)(&res), size * sizeof(float)));
  thrust::device_ptr<float> dPtr(res);
  thrust::fill(dPtr, dPtr + size, 1.0f);
  return res;
}

void free(float* ptr) {
  CUDA_CHECK(cudaFree(static_cast<void*>(ptr)));
}

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
