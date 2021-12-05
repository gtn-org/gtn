#pragma once

#if defined(CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#define CUDA_CHECK(err) \
  cuda::detail::cudaCheck(err, __FILE__, __LINE__)
#endif

#include "gtn/graph.h"

namespace gtn {
namespace cuda {

bool isAvailable();

int deviceCount();

int getDevice();

void setDevice(int device);

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

float* ones(size_t size, int device);

void free(float* ptr);

#if defined(CUDA)
void cudaCheck(cudaError_t err, const char* file, int line);
#endif

} // namespace detail

} // namespace cuda
} // namespace gtn
