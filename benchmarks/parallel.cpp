/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string>

#include "benchmarks/time_utils.h"
#include "gtn/gtn.h"

using namespace gtn;

void timeParallelCompose(const int B, Device device = Device::CPU) {
  const int N1 = 100;
  const int N2 = 50;
  const int A1 = 20;
  const int A2 = 500;

  std::vector<Graph> graphs1;
  std::vector<Graph> graphs2;
  for (int b = 0; b < B; b++) {
    auto f = linearGraph(N1, A1);
    auto s = linearGraph(N2, A2);
    for (int i = 0; i < N2; i++) {
      for (int j = 0; j < A2; j++) {
        // add self loops so composition completes
        s.addArc(i, i, j);
      }
    }
    graphs1.push_back(f.to(device));
    graphs2.push_back(s.to(device));
  }

  auto composeParallel = [&graphs1, &graphs2]() {
    return parallelMap(compose, graphs1, graphs2);
  };

  TIME_DEVICE(composeParallel, device);

  auto out = parallelMap(compose, graphs1, graphs2);
  std::vector<bool> retainGraph({true});
  auto backwardParallel = [&out, &retainGraph]() {
    parallelMap(
        static_cast<void (*)(Graph, bool)>(&backward), out, retainGraph);
  };

  TIME_DEVICE(backwardParallel, device);
}

void timeParallelForward(const int B, Device device = Device::CPU) {
  std::vector<Graph> graphs;
  for (int b = 0; b < B; b++) {
    graphs.push_back(linearGraph(50, 1000, device));
  }

  auto forwardScoreParallel = [&graphs]() {
    parallelMap(forwardScore, graphs);
  };

  TIME_DEVICE(forwardScoreParallel, device);
}


void timeParallelClone(const int B, Device device = Device::CPU) {
  std::vector<Graph> graphs;
  for (int b = 0; b < B; b++) {
    graphs.push_back(linearGraph(10, 1000, device));
  }

  auto cloneParallel = [&graphs]() {
    parallelMap(projectInput, graphs);
  };

  TIME_DEVICE(cloneParallel, device);
}

int main(int argc, char** argv) {
  /**
   * Usage:
   *   `./benchmark_parallel <batch_size (default 8)>`
   */

  int B = 8; // batch size
  if (argc > 1) {
    B = std::stoi(argv[1]);
  }
  std::cout << "Batch size " << B << " with " << detail::getNumViableThreads(B)
    << " threads." << std::endl;
  timeParallelClone(B);
  timeParallelCompose(B);
  timeParallelForward(B);
  if (cuda::isAvailable()) {
    timeParallelClone(B, Device::CUDA);
    timeParallelCompose(B, Device::CUDA);
    timeParallelForward(B, Device::CUDA);
  }
}
