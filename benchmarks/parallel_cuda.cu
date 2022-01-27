#include <iostream>
#include <string>

#include "gtn/gtn.h"
#include "benchmarks/time_utils.h"

using namespace gtn;

__global__
void longKernel(float *x, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    x[i] = sqrt(pow(3.14159,i));
  }
}

void launchKernel(int size) {
  float *data;
  cudaMallocAsync(&data, size * sizeof(float), 0);
  longKernel<<<1, 64>>>(data, size);
  cudaFreeAsync(data, 0);
}

void timeParallelCuda(const int B) {
  std::vector<int> sizes(B, 1 << 20);

  auto timeLongCudaKernel = [&sizes]() {
    parallelMapDevice(Device::CUDA, launchKernel, sizes); 
  };

  TIME_DEVICE(timeLongCudaKernel, Device::CUDA);
}

int main(int argc, char** argv) {
  /**
   * Usage:
   *   `./benchmark_parallel_cuda <batch_size (default 8)>`
   */

  int B = 8; // batch size
  if (argc > 1) {
    B = std::stoi(argv[1]);
  }
  std::cout << "Batch size " << B << " with " << detail::getNumViableThreads(B)
    << " threads." << std::endl;
  if (cuda::isAvailable()) {
    timeParallelCuda(B);
  }
}
