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

#if defined(CUDA)
void cudaCheck(cudaError_t err, const char* file, int line);
#endif

} // namespace detail

} // namespace cuda
} // namespace gtn
