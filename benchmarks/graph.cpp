/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <queue>

#include "benchmarks/time_utils.h"
#include "gtn/gtn.h"

using namespace gtn;

void timeConstructDestruct(Device device = Device::CPU) {
  // Hold the reference to the graphs so we don't time destruction.
  std::vector<Graph> graphs;
  auto linearConstruction = [&graphs, device]() {
    graphs.push_back(linearGraph(1000, 1000, device));
  };
  TIME_DEVICE(linearConstruction, device);

  auto linearDestruction = [&graphs]() { graphs.pop_back(); };
  TIME_DEVICE(linearDestruction, device);
}

void timeCopy(Device device = Device::CPU) {
  auto graph = linearGraph(1000, 1000, device);
  auto copy = [&graph]() { auto copied = Graph::deepCopy(graph); };
  TIME_DEVICE(copy, device);
}

void timeTraversal() {
  auto graph = linearGraph(100000, 100);

  // A simple iterative function to visit every node in a graph.
  auto traverseForward = [&graph]() {
    std::vector<bool> visited(graph.numNodes(), false);
    std::queue<int> toExplore;
    for (auto s : graph.start()) {
      toExplore.push(s);
    }
    while (!toExplore.empty()) {
      auto curr = toExplore.front();
      toExplore.pop();
      for (auto a : graph.out(curr)) {
        auto dn = graph.dstNode(a);
        if (!visited[dn]) {
          visited[dn] = true;
          toExplore.push(dn);
        }
      }
    }
  };
  TIME(traverseForward);

  auto traverseBackward = [&graph]() {
    std::vector<bool> visited(graph.numNodes(), false);
    std::queue<int> toExplore;
    for (auto a : graph.accept()) {
      toExplore.push(a);
    }
    while (!toExplore.empty()) {
      auto curr = toExplore.front();
      toExplore.pop();
      for (auto a : graph.in(curr)) {
        auto un = graph.srcNode(a);
        if (!visited[un]) {
          visited[un] = true;
          toExplore.push(un);
        }
      }
    }
  };
  TIME(traverseBackward);
}

int main() {
  /* Various function benchmarks. */
  timeConstructDestruct();
  timeCopy();
  timeTraversal();
  if (cuda::isAvailable()) {
    timeConstructDestruct(Device::CUDA);
    timeCopy(Device::CUDA);
  }
}
