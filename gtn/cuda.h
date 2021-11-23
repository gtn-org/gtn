#pragma once

#if defined(CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#define CUDA_CHECK(err) \
      cudaCheck(err, __FILE__, __LINE__)
#endif

namespace gtn {
namespace cuda {

bool isAvailable();

#if defined(CUDA)
int deviceCount();

int getDevice();

void setDevice(int device);

void cudaCheck(cudaError_t err, const char* file, int line);
#endif
}
}

