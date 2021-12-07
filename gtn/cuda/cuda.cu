#include <algorithm>
#include <sstream>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "cuda.h"

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
