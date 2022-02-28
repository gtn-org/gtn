/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "gtn/gtn.h"

using namespace gtn;

#define TIME_DEVICE(FUNC, DEVICE) \
  { \
    bool isCuda = Device(DEVICE).isCuda(); \
    auto deviceName = isCuda ? "(cuda)" : "(cpu)"; \
    std::cout << "Timing " << #FUNC << " " << deviceName << " ...  " << std::flush; \
    std::cout << std::setprecision(5) << timeit(FUNC, isCuda) << " msec" << std::endl; \
  }

#define TIME(FUNC) \
  std::cout << "Timing " << #FUNC << " (cpu) ...  " << std::flush; \
  std::cout << std::setprecision(5) << timeit(FUNC) << " msec" << std::endl;

#define milliseconds(x) \
  (std::chrono::duration_cast<std::chrono::nanoseconds>(x).count() / 1e6)
#define timeNow() std::chrono::high_resolution_clock::now()

double timeit(std::function<void()> fn, bool isCuda = false) {
  // warmup
  for (int i = 0; i < 5; ++i) {
    fn();
  }
  if (isCuda) {
    cuda::synchronize();
  }

  int numIters = 100;
  auto start = timeNow();
  for (int i = 0; i < numIters; i++) {
    fn();
    if (isCuda) {
      cuda::synchronize();
    }
  }
  auto end = timeNow();
  return milliseconds(end - start) / static_cast<double>(numIters);
}

// *NB* num_arcs is assumed to be greater than num_nodes.
Graph makeRandomDAG(int num_nodes, int num_arcs) {
  Graph graph;
  graph.addNode(true);
  for (int n = 1; n < num_nodes; n++) {
    graph.addNode(false, n == num_nodes - 1);
  }
  for (int i = 0; i < num_arcs; i++) {
    // To preserve DAG property, select src then select dst to be
    // greater than source.
    // select from [0, num_nodes-2]:
    auto src = rand() % (num_nodes - 1);
    // then select from  [src + 1, num_nodes - 1]:
    auto dst = src + 1 + rand() % (num_nodes - src - 1);
    graph.addArc(src, dst, 0);
  }
  return graph;
}
