#pragma once

#include <cstddef>
#include <functional>

#if defined(_CUDA_)
#include <cuda.h>
#include <cuda_runtime.h>
#define CUDA_CHECK(err) \
  cuda::detail::cudaCheck(err, __FILE__, __LINE__)
#endif

namespace gtn {
namespace cuda {

/** \addtogroup cuda
 *  @{
 */

/**
 * Check if CUDA and a GPU device are available.
 */
bool isAvailable();

/**
 * Returns the number of GPU devices.
 */
int deviceCount();

/**
 * Gets the currently active GPU device.
 */
int getDevice();

/**
 * Sets the active GPU device to `device`.
 */
void setDevice(int device);

/**
 * Waits in the currently active device until all kernels in all streams are
 * complete.
 */
void synchronize();

/**
 * Waits in the given device until all kernels in all streams are complete.
 */
void synchronize(int device);

/**
 * Synchronize the default stream.
 */
void synchronizeStream();

/**
 * A class wrapper for a CUDA event.
 */
class Event {
 public:
  Event();
  ~Event();

  /**
   * Record the event in the default stream.
   */
  void record();

  /**
   * Synchronize the host thread with the event.
   */
  void synchronize();

  /**
   * Wait for the event in the defualt stream.
   */
  void wait();
#if defined(_CUDA_)
 private:
  cudaEvent_t event_;
#endif
};

/** @} */

namespace detail {

// Resource manager for switching devices
class DeviceManager {
 public:
  DeviceManager(int device) :
    device_(getDevice()) {
      setDevice(device);
  }
  ~DeviceManager() {
    setDevice(device_);
  }
 private:
  int device_;
};

void transform(
    const float* aBegin,
    const float* aEnd,
    const float* bBegin,
    float* outBegin,
    const std::function<float>& op);
void negate(const float* in, float* out, size_t size);
void add(const float* a, const float* b, float* out, size_t size);
void subtract(const float* a, const float* b, float* out, size_t size);

void copy(void* dst, const void* src, size_t size);
void* allocate(size_t size, int device);
void free(void* ptr);

bool equal(const float* lhs, const float* rhs, size_t size);
bool equal(const int* lhs, const int* rhs, size_t size);
bool equal(const bool* lhs, const bool* rhs, size_t size);
void fill(float* dst, float val, size_t size);
void fill(int* dst, int val, size_t size);
void fill(bool* dst, bool val, size_t size);

#if defined(_CUDA_)
void cudaCheck(cudaError_t err, const char* file, int line);
#if defined(__NVCC__)
namespace {
template <typename T, typename F>
__global__ void transformKernel(
    const T* a,
    const T* aEnd,
    const T* b,
    T* out,
    const F op) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  int size = aEnd - a;
  if (gTid < size) {
    out[gTid] = op(a[gTid], b[gTid]);
  }
}
template <typename T, typename F>
void transform(
    const T* aBegin,
    const T* aEnd,
    const T* bBegin,
    T* outBegin,
    const F op) {
  int size = aEnd - aBegin;
  if (size == 0) {
    return;
  }
  const int NT = 128;
  const int blocks = (size + NT - 1) / NT;
  transformKernel<<<blocks, NT>>>(aBegin, aEnd, bBegin, outBegin, op);
}
}
#endif
#endif

} // namespace detail

} // namespace cuda
} // namespace gtn
