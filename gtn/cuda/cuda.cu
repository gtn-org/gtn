#include <algorithm>
#include <sstream>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "cuda.h"
#include "gtn/hd_span.h"

namespace gtn {

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

void add(const float* a, const float* b, float* out, size_t size, bool isCuda) {
  if (isCuda) {
    thrust::device_ptr<const float> aPtr(a);
    thrust::device_ptr<const float> bPtr(b);
    thrust::device_ptr<float> outPtr(out);
    thrust::transform(aPtr, aPtr + size, bPtr, outPtr, thrust::plus<float>());
  } else {
    std::transform(a, a + size, b, out, std::plus<>());
  }
}

void fill(bool* dst, bool val, size_t size) {
  thrust::device_ptr<bool> dPtr(dst);
  thrust::fill(dPtr, dPtr + size, val);
}

void fill(int* dst, int val, size_t size) {
  thrust::device_ptr<int> dPtr(dst);
  thrust::fill(dPtr, dPtr + size, val);
}

void fill(float* dst, float val, size_t size) {
  thrust::device_ptr<float> dPtr(dst);
  thrust::fill(dPtr, dPtr + size, val);
}

void copy(void* dst, const void* src, size_t size) {
  CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault));
}

void* allocate(size_t size, int device) {
  DeviceManager dm(device);
  void *res;
  CUDA_CHECK(cudaMalloc(&res, size));
  return res;
}

void free(void* ptr) {
  CUDA_CHECK(cudaFree(ptr));
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
