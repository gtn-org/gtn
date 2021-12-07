#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "cuda.h"

namespace gtn {

namespace {

void copy(int* dst, const int* src, size_t size) {
  CUDA_CHECK(cudaMemcpyAsync(
    static_cast<void*>(dst),
    static_cast<const void*>(src),
    size * sizeof(int),
    cudaMemcpyDefault));
}

} // namespace

void Graph::SharedGraph::allocDevice() {
  cuda::detail::DeviceManager dm(device);
  CUDA_CHECK(cudaMalloc((void**)(&startIds), sizeof(int) * numNodes));
  CUDA_CHECK(cudaMalloc((void**)(&acceptIds), sizeof(int) * numNodes));
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

void Graph::SharedGraph::deepCopy(const Graph::SharedGraph& other) {
  numNodes = other.numNodes;
  numArcs = other.numArcs; 
  numStart = other.numStart;
  numAccept = other.numAccept;
  if (isCuda) {
    allocDevice();
  } else {
    allocHost();
  }
  copy(startIds, other.startIds, numStart);
  copy(acceptIds, other.acceptIds, numAccept);
  copy(start, other.start, numNodes);
  copy(accept, other.accept, numNodes);
  copy(inArcOffset, other.inArcOffset, numNodes + 1);
  copy(outArcOffset, other.outArcOffset, numNodes + 1);
  copy(inArcs, other.inArcs, numArcs);
  copy(outArcs, other.outArcs, numArcs);
  copy(ilabels, other.ilabels, numArcs);
  copy(olabels, other.olabels, numArcs);
  copy(srcNodes, other.srcNodes, numArcs);
  copy(dstNodes, other.dstNodes, numArcs);
}

void Graph::SharedGraph::freeDevice() {
  CUDA_CHECK(cudaFree(startIds)); 
  CUDA_CHECK(cudaFree(acceptIds)); 
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

void Graph::SharedWeights::allocDevice(size_t numArcs, int device) {
  isCuda = true;
  cuda::detail::DeviceManager dm(device);
  CUDA_CHECK(cudaMalloc((void**)(&weights), sizeof(float) * numArcs));
}

void Graph::SharedWeights::deepCopy(
    const float *src, size_t numArcs, bool isCuda, int device) {
  if (isCuda) {
    allocDevice(numArcs, device);
  } else {
    allocHost(numArcs);
  }
  CUDA_CHECK(cudaMemcpyAsync(
    static_cast<void*>(weights),
    static_cast<const void*>(src),
    numArcs * sizeof(float),
    cudaMemcpyDefault));
}

} // namespace gtn
