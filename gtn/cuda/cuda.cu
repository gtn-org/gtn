#include <sstream>
#include <thrust/device_ptr.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

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

void synchronize() {
  CUDA_CHECK(cudaDeviceSynchronize());
}

void synchronize(int device) {
  detail::DeviceManager dm(device);
  synchronize();
}

void synchronizeStream() {
  CUDA_CHECK(cudaStreamSynchronize(0));
}

Event::Event() {
  CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
}

Event::~Event() {
  CUDA_CHECK(cudaEventDestroy(event_));
}

void Event::record() {
  CUDA_CHECK(cudaEventRecord(event_));
}

void Event::synchronize() {
  CUDA_CHECK(cudaEventSynchronize(event_));
}

void Event::wait() {
  CUDA_CHECK(cudaStreamWaitEvent(0, event_));
}

namespace detail {

void negate(const float* in, float* out, size_t size) {
  thrust::transform(
    thrust::device, in, in + size, out, thrust::negate<float>());
}

void add(const float* a, const float* b, float* out, size_t size) {
  transform(
    a, a + size, b, out,
    [] __device__ (const float lhs, const float rhs) {
      return lhs + rhs;
    });
}

void subtract(const float* a, const float* b, float* out, size_t size) {
  transform(
    a, a + size, b, out,
    [] __device__ (const float lhs, const float rhs) {
      return lhs - rhs;
    });
}

void copy(void* dst, const void* src, size_t size) {
  CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault));
}

void* allocate(size_t size, int device) {
  DeviceManager dm(device);
  void *res;
  CUDA_CHECK(cudaMallocAsync(&res, size, 0));
  return res;
}

void free(void* ptr) {
  CUDA_CHECK(cudaFreeAsync(ptr, 0));
}

void cudaCheck(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::ostringstream ess;
    ess << '[' << file << ':' << line
        << "] CUDA error: " << cudaGetErrorString(err);
    throw std::runtime_error(ess.str());
  }
}

bool equal(const float* lhs, const float* rhs, size_t size) {
  thrust::device_ptr<const float> lhsPtr(lhs);
  thrust::device_ptr<const float> rhsPtr(rhs);
  return thrust::equal(lhsPtr, lhsPtr + size, rhsPtr);
}

bool equal(const int* lhs, const int* rhs, size_t size) {
  thrust::device_ptr<const int> lhsPtr(lhs);
  thrust::device_ptr<const int> rhsPtr(rhs);
  return thrust::equal(lhsPtr, lhsPtr + size, rhsPtr);
}

bool equal(const bool* lhs, const bool* rhs, size_t size) {
  thrust::device_ptr<const bool> lhsPtr(lhs);
  thrust::device_ptr<const bool> rhsPtr(rhs);
  return thrust::equal(lhsPtr, lhsPtr + size, rhsPtr);
}

void fill(float* dst, float val, size_t size) {
  thrust::device_ptr<float> ptr(dst);
  thrust::fill(ptr, ptr + size, val);
}

void fill(int* dst, int val, size_t size) {
  thrust::device_ptr<int> ptr(dst);
  thrust::fill(ptr, ptr + size, val);
}

void fill(bool* dst, bool val, size_t size) {
  thrust::device_ptr<bool> ptr(dst);
  thrust::fill(ptr, ptr + size, val);
}

} // namespace detail
} // namespace cuda
} // namespace gtn
