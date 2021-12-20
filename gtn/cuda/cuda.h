#pragma once

#if defined(CUDA)
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

void add(const float* a, const float* b, float* out, size_t size, bool isCuda);

void copy(void* dst, const void* src, size_t size);
void* allocate(size_t size, int device);
void free(void* ptr);

bool equal(const float* lhs, const float* rhs, size_t size);
bool equal(const int* lhs, const int* rhs, size_t size);
bool equal(const bool* lhs, const bool* rhs, size_t size);
void fill(float* dst, float val, size_t size);
void fill(int* dst, int val, size_t size);
void fill(bool* dst, bool val, size_t size);

#if defined(CUDA)
void cudaCheck(cudaError_t err, const char* file, int line);
#endif

} // namespace detail

} // namespace cuda
} // namespace gtn
